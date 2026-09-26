"""The centred-score estimator (core/estimator/engine.cpp): the surface fit at a fixed
topology, the pool variance, and the candidate scores the refinement driver (pyrect/refine.py)
selects on.

The surface solves the centred conditional-mode estimating equation: the joint-ML score for
the surface with its expectation under the generating law subtracted (pyrect/estimator/pool.py
computes those expectations by quadrature over the pool effect and, for pools up to 128 loans,
exact enumeration over the count). The pool variance solves the dispersion equation. The
gate's input is the standardized centred score with its refit- and shrinkage-corrected
sampling variance; accepted meshes solve the nonlinear centred equations.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from pyrect.mesh import K_MINIMUM_FIT_POINTS
from pyrect.estimator.design import live_system, surpluses_from
from pyrect.estimator.padded_design import PaddedDesign, as_design, digest, matvec, rmatvec, take_rows
from pyrect.estimator.pool import _dispersion_score, fit_surface, fit_variance, moments, weighted_gram
from pyrect.refine import refine
from pyrect.units import K_HEIGHT_LIMIT_UNITS


def serial(value):
    """JSON default for numpy values in the audit trace."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


class ScoreSystem:
    """First-order nuisance refit, sampling variance, and shrinkage bias."""
    def __init__(self, X, beta, penalty, expected, score):
        self.X = as_design(X)
        self.j = expected['information']
        self.v = expected['a1']**2*expected['second']
        self.score = np.asarray(score)
        self.factor = cho_factor(weighted_gram(self.X,self.j)+penalty)
        self.omega = weighted_gram(self.X,self.v)
        self.bias_direction = cho_solve(self.factor, penalty@beta)

    def candidate(self, x, rows=None):
        # Candidate columns have local support; nuisance coefficient matrices
        # remain global, so this retains the complete refit correction.
        X = self.X if rows is None else take_rows(self.X, rows)
        j = self.j if rows is None else self.j[rows]
        v = self.v if rows is None else self.v[rows]
        score = self.score if rows is None else self.score[rows]
        cross = rmatvec(X, j*x)
        adjustment = cho_solve(self.factor, cross)
        information = float(x@(j*x)-cross@adjustment)
        noise_cross = rmatvec(X, v*x)
        variance = float(x@(v*x)-2*adjustment@noise_cross+adjustment@self.omega@adjustment)
        raw_score = float(x@score)
        bias = float(cross@self.bias_direction)
        return dict(information=information, variance=variance, raw_score=raw_score,
                    shrinkage_bias=bias)


def _update_information(mesh, system, X, info):
    """Each free coefficient's support at the fit: the information and rows of its nodal column
    (the design is nodal), and its prior precision."""
    w = np.asarray(info, float)
    idx = X.index.ravel(); val = X.value.ravel()
    ww = np.repeat(w, X.width)
    p = X.shape[1]
    xwx = np.bincount(idx, weights=ww*val*val, minlength=p)
    xw2x = np.bincount(idx, weights=ww*ww*val*val, minlength=p)
    rows = np.bincount(idx, weights=(val != 0).astype(float), minlength=p)
    for j, vid in enumerate(system.free):
        vertex = mesh.vertices[vid]
        information = float(xwx[j])
        vertex.current_fit_points = int(rows[j])
        vertex.current_xTx = information
        vertex.current_keff = information*information/float(xw2x[j]) if xw2x[j] > 0 else 0.
        vertex.sigma_sq = 1./information if information > 0 else 1e300
        if vertex.has_prior():
            lv = 1./mesh.tau_for_depth(vertex.depth)
            vertex.lambda_v = lv
            vertex.sigma_pooled = 1./(information+lv) if information+lv > 0 else 1e300


def training_column(ids, values, row_lookup):
    """Select training entries and sum duplicates without a full-length vector."""
    rows = row_lookup[ids]
    keep = rows>=0
    rows, values = rows[keep], values[keep]
    if len(rows)>1 and np.any(rows[1:]<=rows[:-1]):
        unique, inverse = np.unique(rows, return_inverse=True)
        merged = np.zeros(len(unique))
        np.add.at(merged, inverse, values)
        rows, values = unique, merged
    return rows, values



def _store_second_moments(mesh, system, factor, beta):
    """The depth variance's statistic for each child (EM): its fitted surplus squared plus its
    posterior variance in the working Gaussian. The system is in nodal heights, so a child's
    surplus is its transform row r applied to them and its variance r' (X'WX + P)^-1 r, read from
    the scoring system's factor of that same matrix. Stored on the vertex; recompute_tau_sq
    averages it per depth."""
    p = len(factor[0])
    inverse = cho_solve(factor, np.eye(p))
    for c, vid in enumerate(system.children):
        cols, vals = system.transform[system.root_count + c]
        mean = 0.0; variance = 0.0
        for a, wa in zip(cols.tolist(), vals.tolist()):
            mean += wa*beta[a]
            for b, wb in zip(cols.tolist(), vals.tolist()):
                variance += wa*wb*inverse[a, b]
        mesh.vertices[vid].em_second = mean*mean + max(variance, 0.0)


class CandidateGain:
    """A candidate's profile from the engine: the centred score of its column against the frozen
    fit, its penalised gain and the surplus it proposes (core/estimator/engine.h)."""
    __slots__ = ('ok', 'null_units', 'optimum_units', 'gain_nats', 'signed_root_gain', 'raw_gain_nats',
                 'information_at_null', 'data_information_optimum', 'lambda_logit', 'fit_points', 'xTx', 'keff',
                 'centered_gate_z', 'centered_gate_sd')

    def __init__(self):
        self.ok = False; self.null_units = 0.; self.optimum_units = 0.; self.gain_nats = 0.; self.signed_root_gain = 0.
        self.raw_gain_nats = 0.; self.information_at_null = 0.; self.data_information_optimum = 0.; self.lambda_logit = 0.
        self.fit_points = 0; self.xTx = 0.; self.keff = 0.; self.centered_gate_z = 0.; self.centered_gate_sd = 0.


class Engine:
    """The estimator, configured once per run. The frozen settings: local rows within 1e-2 logits
    of their cached base, one Newton variance step per gate round held below 1e-2 (relative)
    with coarse moments first and accurate confirmation, the alternating solver for the initial
    and final variance at 1e-3 (gate rounds) and 1e-5 (final fit), intermediate surfaces
    uncertified, the scoring state reused when nothing moved."""
    ROW_THRESHOLD = 1e-2
    STEP_HOLD = 1e-2
    FINAL_TOLERANCE = 1e-5
    GATE_TOLERANCE = 1e-3

    def __init__(self, config, dataset, audit_path=None):
        self.config = config
        self.dataset = dataset
        prefix = config.output.prefix
        self.audit_path = audit_path if audit_path is not None else (Path(str(prefix) + '_centered_rounds.json') if prefix else None)
        self.trace = []
        self.last_levels = None
        self._level_sizes = None   # the factors' level counts, when there are levels
        self.mesh = None

    # ---- the run ---------------------------------------------------------------------------
    def run(self):
        """Fit the mesh end to end and write the dumps. Returns the process exit status."""
        if self.dataset.pool_count != len(self.dataset.observations):
            raise ValueError('the centered-score mesh needs one row per pool')
        result = refine(self.config, self.dataset, self)
        self.write_metadata()
        self.save()
        return result

    def save(self):
        if self.audit_path is not None:
            self.audit_path.write_text(json.dumps(self.trace, default=serial, indent=2)+'\n')

    # ---- what the driver calls -------------------------------------------------------------
    def refit(self, mesh):
        """Refit the surface at the current topology (skipped when nothing moved)."""
        self.fit(mesh)

    def score(self, frontier, context):
        """Refit at the gate tolerance, then score the frontier against the frozen fit."""
        mesh = context.mesh
        record = self.fit(mesh, estimate=True, tolerance=self.GATE_TOLERANCE)
        record['variance_update'] = 'full'
        record.update(round=context.round, cells_before=mesh.leaf_count(), frontier=len(frontier))
        self.trace.append(record)
        self.save()
        census = context.census
        pushed = 0
        if mesh.dim == 3:   # the kept dry-run columns of candidates no longer on the frontier go
            keep = {(c.at(), c.vertex, c.p0, c.p1, c.axis) for c in frontier}
            cache = mesh.__dict__.get('_column_cache', {})
            for key in [k for k in cache if k not in keep]:
                del cache[key]
        for q, candidate in enumerate(frontier):
            # three dimensions: the exact column of a dry-run move (see RectMesh.exact_candidate_column)
            exact = (self.candidate_gain(mesh, candidate) if mesh.dim == 2 else
                     self.candidate_gain_on(mesh, candidate, mesh.exact_candidate_column(candidate)))
            if exact.xTx <= 0:
                continue
            census.attempted += 1
            if not exact.ok or exact.gain_nats <= 1e-12:
                census.unscoreable += 1
                continue
            census.scored += 1
            pushed += 1
            context.push(q, exact.centered_gate_z, exact.centered_gate_sd, exact)
        print('CENTERED round %d v %s scored %d of %d [%.2f s]' % (context.round, repr(float(record['pool_variance'])),
              pushed, len(frontier), record['seconds']), flush=True)

    def note_round(self, candidates, admitted, null_status, null_sd, null_mean):
        """The round's selection, appended to the audit trace."""
        if not self.trace:
            return
        self.trace[-1].update(candidates=candidates, admitted=admitted, null_status=null_status,
                              null_sd=null_sd, null_mean=null_mean)
        self.save()

    def candidate_gain(self, mesh, candidate):
        """The candidate's profile at the live state: the column its coefficient would have after its
        move, against the frozen fit. A candidate is never free, so its surplus now and at the null
        are both zero."""
        column = mesh.candidate_column(candidate)
        precision = 1.0 / mesh.tau_for_depth(candidate.depth)
        return self._candidate(mesh, column, 0.0, 0.0, precision)

    def candidate_gain_on(self, mesh, candidate, column):
        """The same on a column already computed."""
        precision = 1.0 / mesh.tau_for_depth(candidate.depth)
        return self._candidate(mesh, column, 0.0, 0.0, precision)

    def final_refit(self, mesh):
        """The certified final fit at the frozen topology; writes the final-fit record."""
        record = self.fit(mesh, estimate=True, final=True)
        if self.config.output.prefix:
            Path(str(self.config.output.prefix) + '_final_fit.json').write_text(json.dumps(record, default=serial, indent=2)+'\n')

    def audit(self, mesh):
        """The fixed-point audit of the final fit."""
        state = mesh._centered_state
        np.testing.assert_allclose(mesh.predict_all(), state['prediction'], atol=1e-8, rtol=0)
        if state['fit']['score_max'] > 2e-5:
            raise RuntimeError('final centered equation is not stationary')
        print('CENTERED final score audit', repr(float(state['fit']['score_max'])), flush=True)

    def pool_variance(self):
        return float(getattr(self.mesh, '_centered_variance', 0.)) if self.mesh is not None else 0.

    # ---- the fit ---------------------------------------------------------------------------
    def fit(self, mesh, estimate=False, tolerance=FINAL_TOLERANCE, final=False):
        """Fit the surface (and, with estimate, the pool variance) at the mesh's topology, install
        the fitted heights and the working data on the mesh, and freeze the scoring state."""
        self.mesh = mesh
        config = self.config
        row_threshold = self.ROW_THRESHOLD
        row_cache = mesh.__dict__.setdefault('_row_cache', {})
        accurate_cache = mesh.__dict__.setdefault('_accurate_row_cache', {})
        started = time.monotonic()
        offset_all, n, k = mesh.obs_arrays()
        groups, offset, level_count = self._load_level_groups(mesh, offset_all)
        train = np.asarray(mesh.training_mask, bool) if config.fit.heldout_split else np.ones(mesh.n_points, bool)
        # Only the training rows of the design are ever needed; the consistency check against
        # the mesh's predictions runs on them.
        system = live_system(mesh, np.flatnonzero(train))
        Xt, initial = system.X, system.initial.copy()
        nt, kt = n[train], k[train]
        ot = offset[train]
        surface_count = len(system.free)
        penalty = system.penalty
        if level_count:
            # one more padded entry per row for each factor: the row's indicator in that factor's block
            # every factor's column in one copy of the design (as the binary builds it)
            start = 0; extra_index = []; extra_value = []
            for f, size in enumerate(mesh._level_sizes):
                if size == 0:
                    continue
                g = groups[train, f]
                extra_index.append(np.where(g > 0, surface_count + start + g - 1, 0)[:, None])
                extra_value.append((g > 0).astype(float)[:, None])
                start += size
            Xt = PaddedDesign(np.hstack([Xt.index] + extra_index), np.hstack([Xt.value] + extra_value), surface_count + level_count)
            initial = np.r_[initial, mesh._level_values]
            # the depth prior in nodal coordinates; the level columns are unpenalised
            wide = np.zeros((surface_count + level_count, surface_count + level_count))
            wide[:surface_count, :surface_count] = penalty
            penalty = wide
        variance = getattr(mesh, '_centered_variance', 0.)
        # The same fitted problem can be requested again by a refit: hash all training inputs
        # and the current prior, not just the topology.
        key = hashlib.sha256()
        digest(Xt, key)
        for array in (nt, kt, ot, penalty):
            key.update(memoryview(np.ascontiguousarray(array)).cast('B'))
            key.update(str(array.shape).encode())
        signature = key.hexdigest()
        variance, initial, stepped, variance_fit, reused = self._step_or_start_variance(
            mesh, estimate, final, tolerance, Xt, nt, kt, ot, penalty, initial, variance,
            row_cache, accurate_cache, len(system.free), signature)
        fit_key = (signature, float(variance))
        previous_state = mesh.__dict__.get('_centered_state')
        surface_reused = (variance_fit is None and previous_state is not None
                          and previous_state['fit_key'] == fit_key)
        if previous_state is not None and not surface_reused:
            # the previous round's scoring system holds a full training design; this fit builds
            # its own, so release the old one before the new one exists (one design at peak)
            previous_state['system'] = None
        if (surface_reused
                and previous_state.get('w_version') == mesh.__dict__.get('_w_version', 0)
                and previous_state.get('refresh_epoch') == mesh.refresh_epoch
                and np.array_equal(previous_state['all_heights'],
                                   np.fromiter((v.height for v in mesh.vertices), float, len(mesh.vertices)))):
            record = dict(previous_state['record'])
            record.update(seconds=time.monotonic()-started, surface_evaluations=0, surface_fit_reused=True,
                          skipped_unchanged=True)
            return record
        fitted = (variance_fit['surface'] if variance_fit is not None else
                  previous_state['fit'] if surface_reused else
                  fit_surface(Xt, nt, kt, variance, initial, penalty=penalty, offset=ot,
                              row_cache=row_cache, row_threshold=row_threshold, certify=final))
        if level_count:
            levels = fitted['beta'][surface_count:]
            mesh._level_values = levels.copy()
            row_level = np.zeros(len(groups)); start = 0
            for f, size in enumerate(mesh._level_sizes):
                g = groups[:, f]
                row_level = row_level + np.where(g > 0, levels[np.maximum(start + g - 1, 0)], 0.)
                start += size
            for o, value in zip(mesh.obs_store, (mesh._base_offsets + row_level).tolist()):
                o.offset = value
            mesh.__dict__.pop('_obs_arrays', None)
            offset = mesh._base_offsets + row_level      # what the working data sees
        self._commit_surface_heights(mesh, system, fitted, surface_count)
        pred = mesh.predict_all_array().copy()      # the mesh's array is updated in place later; the round's frozen predictions must not be
        row_shift = matvec(Xt, fitted['beta']) - (offset[train]-mesh._base_offsets[train] if level_count else 0.)
        np.testing.assert_allclose(pred[train], row_shift, atol=1e-8, rtol=0)
        expected = fitted['moments']
        score = expected['a1']*fitted['centered_scaled_modes']
        info = expected['information']
        if np.any(info <= 0) or not np.isfinite(info).all():
            raise RuntimeError('nonpositive centered working information')
        mesh.refresh_epoch += 1
        mesh._w_version = getattr(mesh, '_w_version', 0)+1
        _update_information(mesh, system, Xt, info)
        row_lookup = np.full(mesh.n_points, -1, dtype=np.intp)
        row_lookup[train] = np.arange(len(nt))
        live = system
        score_system = ScoreSystem(Xt, fitted['beta'], penalty, expected, score)
        _store_second_moments(mesh, live, score_system.factor, fitted['beta'])
        mesh._centered_state = dict(fit_key=fit_key, row_lookup=row_lookup, system=score_system,
                                    train=train, prediction=pred, info=info,
                                    expected=expected, fit=fitted,
                                    w_version=mesh._w_version, refresh_epoch=mesh.refresh_epoch,
                                    all_heights=np.fromiter((v.height for v in mesh.vertices), float, len(mesh.vertices)))
        if level_count:
            self.last_levels = mesh._level_values.tolist()
            self._level_sizes = list(mesh._level_sizes)
        record = dict(seconds=time.monotonic()-started, coefficients=len(system.free), pool_variance=variance,
                      levels=(mesh._level_values.tolist() if level_count else None),
                      score_max=fitted['score_max'], surface_evaluations=0 if surface_reused else fitted['evaluations'],
                      surface_fit_reused=surface_reused, exact_evaluations=fitted.get('exact_evaluations'),
                      partial_rows=fitted.get('partial_rows'), certified=fitted.get('certified', True))
        if variance_fit is not None:
            record['variance_fit'] = {key: value for key, value in variance_fit.items()
                                      if key not in ('beta', 'surface')}
            record['variance_fit_reused'] = False if stepped is not None else reused
        mesh._centered_state['record'] = record
        return record

    # fit()'s phases -----------------------------------------------------------------------

    def _load_level_groups(self, mesh, offset_all):
        """Joint population levels: DMESH_LEVEL_GROUPS names a file with one integer per row
        (0 = the reference group). Each other group gets an unpenalised indicator column of
        the design, solved with the surface; the fitted levels are folded into the observation
        offsets after every fit (prediction, dumps, deviance all see them), while the engine
        keeps working from the base offsets it was given. Returns (groups, offset, level_count)."""
        groups = mesh.__dict__.get('_level_groups')
        if groups is None and self._groups_file():
            # one line per row, one column per factor (0 each factor's reference)
            groups = np.loadtxt(self._groups_file(), dtype=int, ndmin=2)
            if groups.shape[0] != mesh.n_points or groups.size == 0 or groups.min() < 0:
                raise ValueError('DMESH_LEVEL_GROUPS needs one nonnegative integer per row')
            mesh._level_groups = groups
            mesh._level_sizes = [int(s) for s in groups.max(axis=0)]
            mesh._level_count = sum(mesh._level_sizes)
            mesh._level_values = np.zeros(mesh._level_count)
            mesh._base_offsets = offset_all.copy()
        if groups is not None:
            return groups, mesh._base_offsets, mesh._level_count
        return None, offset_all, 0

    def _step_or_start_variance(self, mesh, estimate, final, tolerance, Xt, nt, kt, ot, penalty, initial, variance,
                                row_cache, accurate_cache, n_free, signature):
        """The pool variance this fit uses, and the surface fitted while finding it. Returns
        (variance, initial, stepped, variance_fit, reused)."""
        row_threshold = self.ROW_THRESHOLD
        stepped = None
        variance_fit = None
        reused = False
        if estimate and not final and variance == 0 and np.any(nt > 1):
            # One Newton step from zero: the surface at zero variance is the exact-binomial fit
            # (no quadrature); the accurate score and its slope there give the first variance,
            # clipped to the solver's range.
            print('CENTERED initial variance step', n_free, 'coefficients', flush=True)
            zero = fit_surface(Xt, nt, kt, 0., initial, penalty=penalty, offset=ot)
            score0 = _dispersion_score(Xt, nt, zero['beta'], 0., penalty, ot, zero)
            high = moments(ot+matvec(Xt, zero['beta']), nt, 0., 8, 32)
            sensitivity = float(high['weight']@high['sensitivity'])
            first = float(np.clip(score0/sensitivity, 2e-4, 1.9)) if sensitivity > 0 and score0 > 0 else 0.
            if first > 0:
                surface = fit_surface(Xt, nt, kt, first, zero['beta'], penalty=penalty, offset=ot,
                                      row_cache=row_cache, row_threshold=row_threshold, certify=False)
                variance = first
                stepped = dict(variance=variance, status='initial_step', score=score0, variance_step=first,
                               variance_tolerance=None, moved=True, surface=surface,
                               surface_evaluations_total=zero['evaluations']+surface['evaluations'], high_evaluations=1,
                               coarse_evaluations=0, evaluations=1, relative_tolerance=None, solver_used='initial_step',
                               accurate_evaluated=True)
                variance_fit = stepped
                initial = surface['beta']
                mesh._centered_variance = variance
        if stepped is None and estimate and not final and variance > 0:
            # One surface at the held variance, one Newton step on the dispersion equation with
            # the slope from the moments' sensitivity: coarse moments first, accurate ones when
            # the coarse step is small. Convergence of the variance is only claimed by the final
            # fit's full solve.
            print('CENTERED variance step', n_free, 'coefficients', flush=True)
            surface = fit_surface(Xt, nt, kt, variance, initial, penalty=penalty, offset=ot,
                                  row_cache=row_cache, row_threshold=row_threshold, certify=False)
            def newton_step(score, moments_used):
                sensitivity = float(moments_used['weight']@moments_used['sensitivity'])
                return score/sensitivity if sensitivity > 0 else 0.
            step_tolerance = self.STEP_HOLD*max(variance, .01)
            accurate_rows = None
            coarse_score = _dispersion_score(Xt, nt, surface['beta'], variance, penalty, ot, surface, expected=surface['moments'])
            coarse_step = newton_step(coarse_score, surface['moments'])
            if abs(coarse_step) <= step_tolerance:
                score = _dispersion_score(Xt, nt, surface['beta'], variance, penalty, ot, surface,
                                          accurate_cache=accurate_cache, row_threshold=row_threshold)
                accurate_rows = (accurate_cache.get('rows_resolved'), accurate_cache.get('rows_total'))
                step = newton_step(score, accurate_cache.get('moments', surface['moments']))
            else:
                score, step = coarse_score, coarse_step
            moved = abs(step) > step_tolerance
            if moved:
                new_variance = float(np.clip(variance+step, .5*variance, 2.*variance))
                surface = fit_surface(Xt, nt, kt, new_variance, surface['beta'], penalty=penalty, offset=ot,
                                      row_cache=row_cache, row_threshold=row_threshold, certify=False)
                variance = new_variance
            stepped = dict(variance=variance, status='step', score=score, variance_step=step,
                           variance_tolerance=step_tolerance, moved=moved, surface=surface,
                           coarse_step=coarse_step, accurate_evaluated=accurate_rows is not None,
                           surface_evaluations_total=surface['evaluations'], high_evaluations=1,
                           coarse_evaluations=0, evaluations=1, relative_tolerance=tolerance,
                           solver_used='step', accurate_rows_resolved=accurate_rows)
            variance_fit = stepped
            initial = surface['beta']
            mesh._centered_variance = variance
        elif estimate and stepped is None:
            print('CENTERED variance start', n_free, 'coefficients', flush=True)
            cache = mesh.__dict__.setdefault('_centered_variance_cache', {})
            # A cached solve serves a request only at its own precision or looser; the final fit
            # never inherits a gate round's loose root.
            reused = (signature in cache and
                      cache[signature].get('relative_tolerance', self.FINAL_TOLERANCE) <= tolerance)
            if reused:
                variance_fit = cache[signature]
            else:
                variance_fit = fit_variance(Xt, nt, kt, initial, penalty=penalty, offset=ot,
                                            return_surface=True, initial_variance=variance, relative_tolerance=tolerance)
                cache.clear()
                cache[signature] = variance_fit
            if variance_fit.get('variance') is None:
                raise RuntimeError(f'pool variance update unresolved: {variance_fit}')
            variance = variance_fit['variance']
            initial = variance_fit['beta']
            mesh._centered_variance = variance
            row_cache.clear()   # solved at a new variance; bases are stale
            accurate_cache.clear()
        return variance, initial, stepped, variance_fit, reused

    def _commit_surface_heights(self, mesh, system, fitted, surface_count):
        """The fitted coefficients are the free vertices' heights; the mesh keeps surpluses (R h), and
        every height follows from them by the birth-parent recursion. Every height is checked finite
        and within the runaway guard (a near-empty region can honestly reach the proposal limit, but
        never three times over it)."""
        heights = np.asarray(fitted['beta'][:surface_count], float)
        surplus = surpluses_from(system, heights)
        previous = [mesh.vertices[vid].surplus for vid in system.free]
        for vid, value in zip(system.free, surplus.tolist()):
            mesh.vertices[vid].surplus = value
        mesh.recompute_heights()
        worst = max(mesh.vertices, key=lambda v: abs(v.height) if np.isfinite(v.height) else np.inf)
        if not np.isfinite(worst.height) or abs(worst.height) > 3*K_HEIGHT_LIMIT_UNITS:
            for vid, value in zip(system.free, previous):
                mesh.vertices[vid].surplus = value
            mesh.recompute_heights()
            raise RuntimeError('centered surface exceeds existing height safeguard: vertex %d depth %d height %s at (%s, %s)'
                               % (worst._id, worst.depth, worst.height, worst.x, worst.y))

    def _candidate(self, mesh, column, current, null, lam_units):
        out = CandidateGain()
        out.null_units = null
        state = getattr(mesh, '_centered_state', None)
        if state is None or state.get('system') is None:
            raise RuntimeError('no centered state to score against')
        ids, vals = np.asarray(column[0], int), np.asarray(column[1], float)
        big = np.abs(vals) > 1e-12
        rows, x = training_column(ids[big], vals[big], state['row_lookup'])
        out.fit_points = int((abs(x) > 1e-12).sum())
        info_rows = state['info'][rows]
        out.xTx = float(np.sum(info_rows*x*x))
        xT2x = float(np.sum(info_rows*info_rows*x*x))
        out.keff = out.xTx*out.xTx/xT2x if xT2x > 0 else 0.
        if out.fit_points < K_MINIMUM_FIT_POINTS or out.xTx <= 0:
            return out
        stats = state['system'].candidate(x, rows)
        h, variance, score = stats['information'], stats['variance'], stats['raw_score']
        if h <= 1e-12 or variance <= 1e-12:
            return out
        # The displacement is needed on the candidate's rows only: the surface now against the
        # round's frozen predictions.
        train_idx = state.get('train_idx')
        if train_idx is None:
            train_idx = state['train_idx'] = np.flatnonzero(state['train'])
        global_rows = train_idx[rows]
        current_pred = mesh.predict_all_array()          # incremental; read-only
        change_rows = current_pred[global_rows]-state['prediction'][global_rows]
        score -= float(x@(state['info'][rows]*change_rows))
        score -= h*(null-current)
        lam = lam_units
        step = score/(h+lam)
        out.lambda_logit = lam
        out.information_at_null = h+lam
        out.data_information_optimum = h
        out.optimum_units = null+step
        out.gain_nats = out.raw_gain_nats = .5*score*step
        out.signed_root_gain = float(np.sign(score)*np.sqrt(2*out.gain_nats))
        out.ok = True
        out.centered_gate_z = (score-stats['shrinkage_bias'])/np.sqrt(variance)
        out.centered_gate_sd = np.sqrt(variance)/h
        return out

    # ---- metadata --------------------------------------------------------------------------
    @staticmethod
    def _groups_file():
        return os.environ.get('DMESH_LEVEL_GROUPS') or None

    def write_metadata(self):
        """The model file beside the dumps: the domain, the baseline and the estimator's settings
        (the binary's write_metadata, key for key)."""
        prefix = self.config.output.prefix
        if not prefix:
            return
        dataset = self.dataset
        metadata = dict(schema_version=1, baseline_logit=dataset.baseline_logit,
                        wala_min=dataset.bounds[0], wala_max=dataset.bounds[1], wac_min=dataset.bounds[2], wac_max=dataset.bounds[3])
        if dataset.dim() == 3:   # a three-dimensional mesh: the third axis and its range
            metadata.update(axis3=dataset.axis3, axis3_min=dataset.axis3_bounds[0], axis3_max=dataset.axis3_bounds[1])
        metadata.update(height_scale=1, baseline_source=dataset.baseline_source, blas_threads=1)
        inner = dict(fitting='penalized centered conditional-mode equation',
                     gate='first-order refit- and shrinkage-bias-corrected score',
                     proposal='local quadratic; nonlinear joint refit after admission',
                     variance_components='not separately decomposed in vertex dump',
                     selection='existing empirical-null rule; no adaptive FDR guarantee',
                     depth_variance='em',
                     surface_quadrature=dict(count_order=6, pool_order_base=24),   # pool_order_for(variance, 24) per solve
                     variance_quadrature=dict(count_order=8, pool_order=32,
                                              early_search='surface moments; accurate secant reset before acceptance'),
                     surface_initializer='direct', variance_schedule='full update every gate round',
                     variance_solver='alternating', design='padded', hierarchy='rectangular hierarchical surplus')
        metadata['experimental_centered_score'] = inner
        if self.last_levels is not None:
            metadata['centered_levels'] = dict(groups_file=self._groups_file() or '', levels=self.last_levels)
            sizes = self._level_sizes
            if sizes and len(sizes) > 1:   # several factors: the levels above are their blocks in order
                metadata['centered_levels']['factor_sizes'] = list(sizes)
        inner['local_rows'] = dict(row_threshold_logits=self.ROW_THRESHOLD, certify_intermediate=False,
                                   variance_schedule='step', step_hold=self.STEP_HOLD, step_moments='coarse', step_slope='newton',
                                   initial_variance='step', skip_unchanged=True, trim_caches=True)
        inner['variance_tolerance'] = dict(gate_rounds=self.GATE_TOLERANCE, final_fit=self.FINAL_TOLERANCE,
                                           note='relative step in variance units; alternating solver only')
        Path(str(prefix) + '_model.json').write_text(json.dumps(metadata, indent=2) + '\n')
