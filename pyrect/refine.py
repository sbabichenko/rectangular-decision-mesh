"""core/fit/refine.cpp: the refinement driver. It builds the coarse mesh, runs the gate rounds
(score the candidate frontier against the frozen fit, calibrate an empirical null, admit the
selected prefix), refits, and writes the report and the dumps. Every fit is the engine's. The
triangular driver's rounds, null and selection, with the rectangular mesh's candidates: an
existing constrained vertex, or the midpoint of a leaf edge whose admission splits the cells around
it (a hanging vertex's release is the same move)."""
from __future__ import annotations

import sys
import time

import numpy as np

from .gate import LfdrResult, STATUS_NAMES, lindsey_lfdr, normal_survival
from .mesh import K_SIDE, RectMesh
from .report import report_heldout, write_candidates, write_dumps
from .stdlib import fabs, isfinite, sqrt, std_sort
from .units import K_HEIGHT_LIMIT_UNITS

# The refinement budget and the stopping rules of the gate rounds.
K_PRE_REFINE_LEVELS = 3          # the coarse geometry: an 8 x 8 grid of cells
K_PRE_REFINE_LEVELS_AXIS3 = 2    # three dimensions: 8 x 8 x 4, the third axis starting coarser
K_BACKBONE_DEPTH = 1             # pre-refined vertices this deep start as free coefficients
K_MINIMUM_CELL_CAP = 600         # cell budget: max(this, N // K_ROWS_PER_CELL)
K_ROWS_PER_CELL = 24             # about one cell per this many rows: the data-density ceiling
K_FDR_LEVEL = 0.10               # the gate's false discovery rate
K_MAX_ROUNDS = 80
K_STALL_LIMIT = 3                # consecutive rounds admitting nothing
K_REFRESH_PERIOD = 3             # rounds between refits of the whole surface
K_LINDSEY_MINIMUM_CANDIDATES = 50   # below this the round uses direct BH

err = sys.stderr.write
out = sys.stdout.write


class CandidateCensus:
    """The run's census of candidate profiles, reported once at the end."""
    def __init__(self):
        self.attempted = 0; self.scored = 0; self.unscoreable = 0
        self.reprofiled = 0; self.reprofile_rejected = 0


class ScoreContext:
    """What one round's scoring reads and writes: the mesh, the round number, the census, and
    the sink for each scoreable candidate (its frontier index, gate coordinate, scale, profile)."""
    def __init__(self, mesh, rnd, census, push):
        self.mesh = mesh; self.round = rnd; self.census = census; self.push = push


class RoundCandidates:
    """One round's scored candidates, one row per scoreable candidate."""
    def __init__(self):
        self.candidates = []; self.z = []; self.z_sd = []; self.exact = []

    def push(self, candidate, z_value, sd, profile):
        self.candidates.append(candidate); self.z.append(z_value); self.z_sd.append(sd); self.exact.append(profile)


def height_is_valid(h):
    return isfinite(h) and fabs(h) <= K_HEIGHT_LIMIT_UNITS


def pre_refine_mesh(mesh):
    """The coarse geometry; the roots and the depth-1 vertices (a bilinear surface and its midline
    corrections) start free, the deeper pre-refined vertices stay constrained on the frontier."""
    if mesh.dim == 3:
        mesh.refine_uniform((K_PRE_REFINE_LEVELS, K_PRE_REFINE_LEVELS, K_PRE_REFINE_LEVELS_AXIS3))
    else:
        mesh.refine_uniform(K_PRE_REFINE_LEVELS)
    for v in mesh.vertices:
        if v.has_prior() and v.depth > K_BACKBONE_DEPTH:
            continue
        v.free_coefficient = True
        v.backbone = True
    mesh.coefficient_version += 1
    mesh.recompute_heights()


def admit_selected_candidates(rnd, kk, order, scored, c_round, mesh, engine, census, admitted_flags):
    """Admit the selected candidates, best first, each re-profiled at the actual commit state on
    its realised geometry (see the binary). Returns how many were admitted."""
    admitted = 0
    for i in range(kk):
        index = order[i]
        scored_exact = scored.exact[index]
        candidate = mesh.refresh(scored.candidates[index])
        if candidate.vertex >= 0 and mesh.vertices[candidate.vertex].free_coefficient:
            continue
        if not scored_exact.ok or scored_exact.raw_gain_nats < -1e-8 or not (scored_exact.gain_nats > 1e-12):
            continue
        existed = candidate.vertex >= 0
        topology_before = mesh.topology_version
        vid = mesh.realise(candidate)
        splits = mesh.topology_version - topology_before
        mesh.recompute_heights()
        candidate = mesh.refresh(candidate)
        census.reprofiled += 1
        exact = engine.candidate_gain(mesh, candidate)
        if not exact.ok or exact.raw_gain_nats < -1e-8 or not (exact.gain_nats > 1e-12):
            census.reprofile_rejected += 1
            continue
        v = mesh.vertices[vid]
        if not height_is_valid(v.height + exact.optimum_units):
            census.reprofile_rejected += 1
            continue
        v.free_coefficient = True
        v.gate_admitted = True
        v.surplus = exact.optimum_units
        v.lambda_v = exact.lambda_logit
        v.sigma_sq = 1.0 / exact.data_information_optimum if exact.data_information_optimum > 0.0 else 1e300
        v.sigma_pooled = (1.0 / (exact.data_information_optimum + v.lambda_v)
                          if exact.data_information_optimum + v.lambda_v > 0.0 else 1e300)
        v.admit_round = v.cand_round = rnd
        v.admit_beta = scored_exact.optimum_units - scored_exact.null_units
        v.admit_sd = scored.z_sd[index]
        v.admit_z = scored.z[index]
        v.admit_c = c_round
        v.admit_xTx = scored_exact.xTx; v.admit_keff = scored_exact.keff
        v.admit_gain = scored_exact.gain_nats; v.admit_signed_root = scored_exact.signed_root_gain
        v.admit_info_null = scored_exact.information_at_null
        v.admit_move = 0 if not existed else (1 if splits == 0 else 2)
        v.admit_splits = splits
        v.admit_chord_lo, v.admit_chord_hi = mesh.chord_ends(vid)
        mesh.coefficient_version += 1
        mesh.recompute_heights()
        admitted_flags[index] = 1
        admitted += 1
    return admitted


def report_calibration(rnd, scored, lfdr_result, lindsey_fallback, kk, engine):
    """The round's calibration line: score moments, the fitted empirical null, the selection
    method and the admitted prefix."""
    zmean = 0.0; zsq = 0.0
    for z in scored.z:
        zmean += z
    zmean /= max(len(scored.z), 1)
    for z in scored.z:
        zsq += (z - zmean) * (z - zmean)
    round_z_sd = sqrt(zsq / (len(scored.z) - 1)) if len(scored.z) > 1 else 0.0
    err("[calibration] round %d M %d score mean/sd %.6g/%.6g emp-null mean/sd/pi0 %.6g/%.6g/%.6g method %s admitted-prefix %d lindsey-status %s iter %d mass %.9g curvature %.9g\n"
        % (rnd, len(scored.z), zmean, round_z_sd, lfdr_result.empirical_mean, lfdr_result.empirical_sd, lfdr_result.pi0,
           "lindsey" if lfdr_result.used_lindsey else (("BH-fallback" if lfdr_result.status == "invalid_central_fit" else "defer-invalid-null") if lindsey_fallback else "BH"), kk,
           STATUS_NAMES[lfdr_result.status], lfdr_result.iterations, lfdr_result.fitted_mass, lfdr_result.central_curvature))
    engine.note_round(len(scored.z), kk, lfdr_result.status, lfdr_result.empirical_sd, lfdr_result.empirical_mean)


def run_gate_rounds(mesh, cell_cap, engine, census, records=None):
    """The gate rounds to convergence: score the frontier, calibrate an empirical null, select
    under it, admit. The loop ends when the cell budget is spent, when a round finds no
    scoreable candidate, or after K_STALL_LIMIT consecutive rounds that admit nothing."""
    stalls = 0
    for rnd in range(K_MAX_ROUNDS):
        if mesh.leaf_count() > cell_cap:
            break
        mesh.recompute_tau_sq()
        engine.refit(mesh)
        scored = RoundCandidates()
        frontier = mesh.frontier()
        engine.score(frontier, ScoreContext(mesh, rnd, census,
                                            lambda q, z, sd, e: scored.push(frontier[q], z, sd, e)))
        if not scored.candidates:
            break
        for q, candidate in enumerate(scored.candidates):
            if candidate.vertex >= 0:
                v = mesh.vertices[candidate.vertex]
                e = scored.exact[q]
                v.cand_round = rnd; v.cand_z = scored.z[q]; v.cand_sd = scored.z_sd[q]
                v.cand_beta = e.optimum_units - e.null_units
                v.cand_xTx = e.xTx; v.cand_keff = e.keff
                v.cand_gain = e.gain_nats; v.cand_signed_root = e.signed_root_gain
                v.cand_info_null = e.information_at_null
        lf = []
        lfdr_result = None
        direct_bh_order = []
        direct_bh_k = -1
        lindsey_fallback = False

        def run_direct_bh():
            nonlocal lf, direct_bh_order, direct_bh_k
            n = len(scored.z)
            pv = [2 * normal_survival(fabs(z)) for z in scored.z]
            op = list(range(n))
            std_sort(op, lambda a, b: pv[a] < pv[b] or (pv[a] == pv[b] and a < b))   # ties by index, as the binary
            kbh = 0
            for i in range(len(op)):
                if pv[op[i]] <= K_FDR_LEVEL * (i + 1) / len(op):
                    kbh = i + 1
            direct_bh_order = op
            direct_bh_k = kbh
            lf = [1.0] * n
            for i in range(kbh):
                lf[direct_bh_order[i]] = 0.0

        if len(scored.z) >= K_LINDSEY_MINIMUM_CANDIDATES:
            lfdr_result = lindsey_lfdr(scored.z, True)
            if lfdr_result.used_lindsey:
                lf = lfdr_result.lfdr
            else:
                lindsey_fallback = True
                if lfdr_result.status == "invalid_central_fit":
                    # a converged density without a central peak keeps the theoretical-null path
                    run_direct_bh()
                else:
                    # numerical failure does not establish a standard-normal null: the stall
                    # logic refits and retries
                    lf = [1.0] * len(scored.z)
        else:
            run_direct_bh()
        if lfdr_result is None:
            lfdr_result = LfdrResult()
        if direct_bh_k >= 0:
            order = direct_bh_order
            kk = direct_bh_k
        else:
            order = list(range(len(lf)))
            std_sort(order, lambda a, b: lf[a] < lf[b] or (lf[a] == lf[b] and a < b))
            cum = 0.0
            kk = 0
            for i in range(len(order)):
                cum += lf[order[i]]
                if cum / (i + 1) <= K_FDR_LEVEL:
                    kk = i + 1
                else:
                    break
        report_calibration(rnd, scored, lfdr_result, lindsey_fallback, kk, engine)
        admitted = 0
        admitted_flags = [0] * len(scored.z)
        if kk > 0:
            c_round = 1e300
            for i in range(kk):
                c_round = min(c_round, fabs(scored.z[order[i]]))
            admitted = admit_selected_candidates(rnd, kk, order, scored, c_round, mesh, engine, census, admitted_flags)
        if records is not None:   # the round's candidates, as scored, for the star plots
            selected = [0] * len(scored.z)
            for i in range(kk):
                selected[order[i]] = 1
            # the round as one float block (the integers exact in doubles): tuples of boxed numbers
            # for tens of thousands of candidates cost several times more
            block = np.empty((len(scored.candidates), 15))
            for q, c in enumerate(scored.candidates):
                e = scored.exact[q]
                block[q] = (rnd, c.ix / K_SIDE, c.iy / K_SIDE, c.iz / K_SIDE, c.vertex, c.axis, c.depth, c.p0, c.p1, scored.z[q],
                            scored.z_sd[q], e.optimum_units - e.null_units, e.gain_nats, selected[q], admitted_flags[q])
            records.append(block)
        # a round is a stall when nothing was admitted (see the binary)
        if not admitted:
            stalls += 1
            if stalls >= K_STALL_LIMIT:
                break
            engine.refit(mesh)
            continue
        stalls = 0
        if rnd % K_REFRESH_PERIOD == K_REFRESH_PERIOD - 1:
            engine.refit(mesh)


def refine(config, dataset, engine):
    """Fit the mesh: the coarse geometry, the gate rounds, the final refit, the report and the
    dumps. Returns the process exit status."""
    obs = dataset.observations
    N = len(obs)
    out("pool-months %d, pools %d\n" % (N, dataset.pool_count))
    t_start = time.monotonic()
    heldout_split = config.fit.heldout_split
    mask = list(dataset.training_mask) if heldout_split else [1] * N
    mesh = RectMesh(obs, mask, dataset.dim())

    pre_refine_mesh(mesh)
    mesh.recompute_tau_sq()
    engine.refit(mesh)
    out("  [diag pre-rounds] cells %d, vertices %d, pred(0.5,0.5)=%.3f\n" % (mesh.leaf_count(), len(mesh.vertices), mesh.predict(0.5, 0.5)))
    cell_cap = max(K_MINIMUM_CELL_CAP, N // K_ROWS_PER_CELL)
    census = CandidateCensus()
    candidate_rows = []
    run_gate_rounds(mesh, cell_cap, engine, census, candidate_rows)
    engine.refit(mesh)
    free_count = sum(1 for v in mesh.vertices if v.free_coefficient)
    out("refinement done: cells %d, vertices %d, free %d  [%.1f s]\n" % (mesh.leaf_count(), len(mesh.vertices), free_count, time.monotonic() - t_start))

    engine.final_refit(mesh)
    engine.audit(mesh)

    report_heldout(config, mesh, obs)
    err("[candidate census] attempted %d scored %d unscoreable %d | reprofiled %d rejected %d\n"
        % (census.attempted, census.scored, census.unscoreable, census.reprofiled, census.reprofile_rejected))
    out("TOTAL %.1f s\n" % (time.monotonic() - t_start))
    sx = [o.x for o in obs]; sy = [o.y for o in obs]
    write_dumps(config.output, mesh, obs, sx, sy, dataset.pool_count, engine.pool_variance())
    write_candidates(config.output, candidate_rows, mesh.dim)
    return 0
