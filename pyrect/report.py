"""The held-out report and the dumps (core/fit/refine.cpp's report_heldout and core/fit/dumps.cpp)."""
from __future__ import annotations

import sys

import numpy as np
from scipy.special import expit

from .mesh import K_SIDE
from .stdlib import g10

out = sys.stdout.write


def row_terms(mesh, pred, lo=None, hi=None):
    """The probability, information and Pearson residual of every row at the predictions pred
    (clamped to [lo, hi] when given), as three arrays."""
    offset, exposure, events = mesh.obs_arrays()
    p = expit(offset + np.asarray(pred, dtype=float))
    if lo is not None:
        p = np.clip(p, lo, hi)
    info = exposure * p * (1.0 - p)
    return p, info, (events - exposure * p) / np.sqrt(info)


def report_heldout(config, mesh, obs):
    """The held-out deviance of the final fit."""
    if not config.fit.heldout_split:
        return
    pred = mesh.predict_all()
    ph, _, _ = row_terms(mesh, pred, 1e-6, 1 - 1e-6)
    held = np.asarray(mesh.training_mask) == 0
    _, n, k = mesh.obs_arrays()
    m = int(held.sum())
    if m > 0:
        p = ph[held]; n = n[held]; k = k[held]
        dev = np.where(k > 0, 2 * k * np.log(np.where(k > 0, k, 1) / (n * p)), 0.0)
        dev = dev + np.where(n - k > 0, 2 * (n - k) * np.log(np.where(n - k > 0, n - k, 1) / (n * (1 - p))), 0.0)
        inf = n * p * (1 - p)
        x2 = float(np.sum((k - n * p) ** 2 / inf)); wsum = float(inf.sum())
        if wsum > 0:
            out("HELDOUT: mean deviance/pool %.3f | X2/info %.4f | %d pools\n" % (float(dev.sum()) / m, x2 / wsum, m))
            return
    out("HELDOUT: no held-out pools\n")


def write_candidates(output, rows, dim=2):
    """Every scored candidate of every gate round (core/fit/dumps.cpp write_candidates). z is the
    gate's coordinate; a three-dimensional mesh's third position is x3."""
    if output.prefix is None:
        return
    with open(output.prefix + "_candidates.csv", "w") as f:
        f.write(("round,x,y,x3," if dim == 3 else "round,x,y,") + "vertex,axis,depth,parent0,parent1,z,sd,beta,gain,selected,admitted\n")
        for block in rows:   # one float block per round (refine.run_gate_rounds)
            for r in block.tolist():
                rnd, x, y, x3, vertex, axis, depth, p0, p1, z, sd, beta, gain, selected, admitted = r
                f.write(",".join([str(int(rnd)), g10(x), g10(y)] + ([g10(x3)] if dim == 3 else []) + [str(int(vertex)), str(int(axis)), str(int(depth)),
                                  str(int(p0)), str(int(p1)), g10(z), g10(sd), g10(beta), g10(gain), str(int(selected)), str(int(admitted))]) + "\n")


def write_dumps(output, mesh, observations, x, y, pool_count, pool_variance):
    """The dumps: the surface on a grid, the observations with their fitted rates, the vertex
    table, the cell table and the pool-effect posteriors at the fitted pool variance."""
    if output.prefix is None:
        return
    pred = mesh.predict_all()
    base = output.prefix
    three = mesh.dim == 3
    with open(base + "_surface.csv", "w") as f:
        f.write("gx,gy,gz,g_hat\n" if three else "gx,gy,g_hat\n")
        grid_size = 60
        # three dimensions: the same grid on five slices of the third axis
        for gz in ((0.0, 0.25, 0.5, 0.75, 1.0) if three else (0.0,)):
            for a in range(grid_size):
                for b in range(grid_size):
                    gx = (a + 0.5) / grid_size
                    gy = (b + 0.5) / grid_size
                    if three:
                        f.write(",".join([g10(gx), g10(gy), g10(gz), g10(mesh.predict(gx, gy, gz))]) + "\n")
                    else:
                        f.write(",".join([g10(gx), g10(gy), g10(mesh.predict(gx, gy))]) + "\n")
    probability, information, residual = row_terms(mesh, pred)
    with open(base + "_obs.csv", "w") as f:
        f.write("x,y,z,pid,n,k,p_hat,resid,info,g_hat,is_train,offset\n" if three else "x,y,pid,n,k,p_hat,resid,info,g_hat,is_train,offset\n")
        has_mask = len(mesh.training_mask) == len(observations)
        xs = list(x); ys = list(y); preds = list(pred)
        probs = probability.tolist(); infos = information.tolist(); resids = residual.tolist()
        for i, o in enumerate(observations):
            f.write(",".join([g10(xs[i]), g10(ys[i])] + ([g10(o.z)] if three else []) + [str(o.pool_id), str(o.exposure), str(o.events), g10(probs[i]),
                              g10(resids[i]), g10(infos[i]), g10(preds[i]),
                              str(int(mesh.training_mask[i]) if has_mask else 1), g10(o.offset)]) + "\n")
    with open(base + "_hier_vertices.csv", "w") as f:
        f.write(("id,x,y,z," if three else "id,x,y,") + "height,surplus,depth,axis,parent0,parent1,gate_admitted,free_coefficient,backbone,"
                "sigma_sq,sigma_pooled,lambda_v,em_second,current_fit_points,current_xTx,current_keff,"
                "admit_round,admit_beta,admit_sd,admit_z,admit_c,admit_xTx,admit_keff,"
                "admit_gain,admit_signed_root,admit_info_null,"
                "cand_round,cand_beta,cand_sd,cand_z,cand_keff,admit_move,admit_splits,admit_chord_lo,admit_chord_hi\n")
        for v in mesh.vertices:
            row = [str(v._id), g10(v.x), g10(v.y)] + ([g10(v.z)] if three else []) + [g10(v.height), g10(v.surplus), str(v.depth), str(v.axis), str(v.p0), str(v.p1),
                   "1" if v.gate_admitted else "0", "1" if v.free_coefficient else "0", "1" if v.backbone else "0",
                   g10(v.sigma_sq), g10(v.sigma_pooled), g10(v.lambda_v), g10(v.em_second),
                   str(v.current_fit_points), g10(v.current_xTx), g10(v.current_keff),
                   str(v.admit_round), g10(v.admit_beta), g10(v.admit_sd), g10(v.admit_z), g10(v.admit_c), g10(v.admit_xTx), g10(v.admit_keff),
                   g10(v.admit_gain), g10(v.admit_signed_root), g10(v.admit_info_null),
                   str(v.cand_round), g10(v.cand_beta), g10(v.cand_sd), g10(v.cand_z), g10(v.cand_keff),
                   str(v.admit_move), str(v.admit_splits), str(v.admit_chord_lo), str(v.admit_chord_hi)]
            f.write(",".join(row) + "\n")
    with open(base + "_mesh.csv", "w") as f:
        # corner k's name spells its side per axis in axis order (h10: x1, y0; h011: x0, y1, z1)
        names = ["".join("1" if (k >> d) & 1 else "0" for d in range(mesh.dim)) for k in range(mesh.corner_count())]
        f.write(("x0,y0,z0,x1,y1,z1" if three else "x0,y0,x1,y1") + "".join(",h" + n for n in names) + "".join(",v" + n for n in names) + ",n_pts\n")
        for c in mesh.cells:
            if not c.leaf():
                continue
            row = [g10(c.lo[d] / K_SIDE) for d in range(mesh.dim)] + [g10(c.hi[d] / K_SIDE) for d in range(mesh.dim)]
            row += [g10(mesh.vertices[k].height) for k in c.corner] + [str(k) for k in c.corner] + [str(len(c.rows))]
            f.write(",".join(row) + "\n")
    # the pool-effect posterior of every pool at the fitted variance (one row per pool here)
    floored_variance = max(pool_variance, 1e-4)
    _, n_obs, k_obs = mesh.obs_arrays()
    pools = np.fromiter((o.pool_id for o in observations), dtype=np.int64, count=len(observations))
    working_residual = (k_obs - n_obs * probability) / information
    weighted_error = np.bincount(pools, weights=information * working_residual, minlength=pool_count)
    pool_information = np.bincount(pools, weights=information, minlength=pool_count)
    with open(base + "_pools.csv", "w") as f:
        f.write("pid,u_hat,W\n")
        keep = np.flatnonzero(pool_information > 0.0)
        estimate = weighted_error[keep] / (pool_information[keep] + 1.0 / floored_variance)
        for pool, est, info in zip(keep.tolist(), estimate.tolist(), pool_information[keep].tolist()):
            f.write(",".join([str(pool), g10(est), g10(info)]) + "\n")
