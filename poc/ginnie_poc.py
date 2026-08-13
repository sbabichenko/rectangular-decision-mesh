"""Binomial rectangular-mesh fits of the frozen Ginnie designs:
cumulative attrition proxy (data/ginnie_design.csv, pooled rate 0.7832)
and monthly SMM (data/smm_design_202606.csv, rate 0.0085).

Exact per-pool binomial likelihood k ~ Bin(n, expit(f)), fitted by
penalized IRLS on the hierarchical basis (design built once per
topology; IRLS/EM are linear algebra), per-depth tau by EM on the
working model, and the e-BH gate rounds with score-test statistics
under the triangular pool-variance pricing:
    Var(beta_hat) = 1/xTwx + GEV * sum(w^2 b^2) / xTwx^2   (GEV = 0.35)
with w = n p(1-p) the binomial information. This replaces the earlier
Gaussian empirical-logit surrogate, whose SMM breakdown (66.5% of pools
have zero events) motivated the switch. Pools are subsampled for PoC
runtime; caveats in the results JSON.
"""

import json
import os
import numpy as np

from rect_mesh import RectMesh, S
from run_poc import heatmap_svg, mesh_svg
from gate_poc import ebh_admit, bh_admit, normal_sf, exact_column

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# Production triangular config (scripts/run_ginnie_benchmark.sh), ported:
GEV = 0.35            # DMESH_GATE_EXTRA_VAR
TAU = 0.5             # EM init only
TAU_FLOOR = 0.005     # DMESH_TAU_FLOOR_LOGIT_SD
Q = 0.1               # FDR level (CLI arg 2)
MAX_ADMIT = 60        # PoC runtime cap (production has no such cap)
MAX_ROUNDS = 80       # DMESH_MAX_GATE_ROUNDS
MIN_PTS = 10          # Vertex::kMinimumFitPoints
MIN_KEFF = 0.5        # Vertex::minimum_effective_support()
PARENT_VAR_SCALE = 0.25  # DMESH_PARENT_VAR_SCALE
IRLS_STEP_CAP = 4.0   # DMESH_IRLS_STEP_CAP
SIGMA0_MIN, SIGMA0_MAX = 0.3, 6.0   # DMESH_SIGMA0_MIN/MAX
PI0_MIN, D0_MAX = 0.05, 0.5         # DMESH_PI0_MIN, delta0 cap
N_SUB = 12000
HOLD_FRAC = 0.25

WALA_LO, WALA_HI = 0.0, 360.0
WAC_LO, WAC_HI = 1.8, 9.0


def load(fname, rng):
    d = np.genfromtxt(os.path.join(DATA, fname), delimiter=",", names=True)
    idx = rng.permutation(len(d))[:N_SUB]
    d = d[idx]
    x = (d["wala"] - WALA_LO) / (WALA_HI - WALA_LO)
    y = (d["wac"] - WAC_LO) / (WAC_HI - WAC_LO)
    n, k = d["n0"], d["k_term"]
    # empirical logit + weight kept for the observed-panel DISPLAY only
    pt = (k + 0.5) / (n + 1.0)
    z = np.log((k + 0.5) / (n - k + 0.5))
    w = n * pt * (1 - pt)
    # Locally balanced holdout (DMESH_SPLIT_MODE=local): hold out 25%
    # within each WALA/WAC block rather than by unrestricted rows, so
    # train and holdout are spatially matched.
    hold = np.zeros(len(d), bool)
    bx = np.clip((x * 12).astype(int), 0, 11)
    by = np.clip((y * 12).astype(int), 0, 11)
    for b in set(zip(bx, by)):
        idx_b = np.where((bx == b[0]) & (by == b[1]))[0]
        nh = int(round(HOLD_FRAC * len(idx_b)))
        hold[rng.choice(idx_b, nh, replace=False)] = True
    return x, y, z, w, n, k, hold


def expit(f):
    return 1.0 / (1.0 + np.exp(-np.clip(f, -12.0, 12.0)))


def lindsey_lfdr(z):
    """Python port of core/lfdr.cpp (Lindsey's method: Poisson GLM
    histogram fit, degree 6; empirical null by central matching) with
    the production clamps sigma0 in [0.3, 6], pi0 >= 0.05, |d0| <= 0.5.
    Returns (lfdr array, ok flag); fail-closed lfdr = 1 on any failure,
    caller falls back to BH (as the pipeline does)."""
    z = np.clip(np.asarray(z, float), -9.0, 9.0)
    NB, DEG = 72, 6
    lo, hi = -9.5, 9.5
    dbin = (hi - lo) / NB
    mids = lo + (np.arange(NB) + 0.5) * dbin
    counts = np.zeros(NB)
    for v in z:
        counts[min(max(int((v - lo) / dbin), 0), NB - 1)] += 1
    t = (mids - mids.mean()) / mids.std(ddof=1)
    B = np.vander(t, DEG + 1, increasing=True)
    beta = np.zeros(DEG + 1)
    conv = False
    for _ in range(100):
        eta = np.clip(B @ beta, -30, 30)
        mu = np.exp(eta)
        zw = eta + (counts - mu) / np.maximum(mu, 1e-10)
        A = B.T @ (mu[:, None] * B) + 1e-8 * np.eye(DEG + 1)
        nb = np.linalg.solve(A, B.T @ (mu * zw))
        if not np.all(np.isfinite(nb)):
            return np.ones(len(z)), False
        if np.max(np.abs(nb - beta)) < 1e-9:
            beta = nb
            conv = True
            break
        beta = nb
    if not conv:
        return np.ones(len(z)), False
    fbins = np.exp(np.clip(B @ beta, -30, 30)) / (len(z) * dbin)
    if not np.all(np.isfinite(fbins)) or np.any(fbins <= 0):
        return np.ones(len(z)), False
    mass = float(fbins.sum() * dbin)
    if not (0.98 <= mass <= 1.02):
        return np.ones(len(z)), False
    cen = np.abs(mids) <= 2.0
    X3 = np.vander(mids[cen], 3, increasing=True)
    th = np.linalg.lstsq(X3, np.log(np.maximum(fbins[cen], 1e-300)),
                         rcond=None)[0]
    a0, a1, a2 = th
    if not np.all(np.isfinite(th)) or not (a2 < -1e-10):
        return np.ones(len(z)), False
    s0 = np.clip(np.sqrt(-1.0 / (2 * a2)), SIGMA0_MIN, SIGMA0_MAX)
    d0 = np.clip(a1 * s0 * s0, -D0_MAX, D0_MAX)
    pi0 = np.clip(np.exp(a0 + d0 * d0 / (2 * s0 * s0))
                  * np.sqrt(2 * np.pi) * s0, PI0_MIN, 1.0)
    fh = fbins[np.clip(((z - lo) / dbin).astype(int), 0, NB - 1)]
    f0 = np.exp(-0.5 * ((z - d0) / s0) ** 2) / (s0 * np.sqrt(2 * np.pi))
    return np.clip(pi0 * f0 / np.maximum(fh, 1e-300), 0.0, 1.0), True


def lfdr_admit(zs, q):
    """Production family decision: Lindsey lfdr, admit the prefix with
    running mean lfdr <= q; BH fallback when Lindsey fails or the
    family is small (< 50), as in core/pipeline.cpp."""
    zs = np.asarray(zs, float)
    if len(zs) >= 50:
        lf, ok = lindsey_lfdr(zs)
        if ok:
            order = np.argsort(lf)
            cum = np.cumsum(lf[order])
            kk = int(np.sum(cum / (np.arange(len(order)) + 1) <= q))
            return list(order[:kk])
    p = np.array([2 * normal_sf(abs(v)) for v in zs])
    return bh_admit(p, q)


def ebf(beta, info, tau):
    """e-value: Bayes factor for beta_hat ~ N(delta, 1/info),
    delta ~ N(0, tau^2) vs delta = 0."""
    t2 = tau * tau
    return np.sqrt(1.0 / (1.0 + t2 * info)) * np.exp(
        min(700.0, t2 * info * info * beta * beta / (2 * (1.0 + t2 * info))))


def fit_binom_em(mesh, xs, ys, nn, kk, em_iters=3, irls_iters=8):
    """Penalized binomial IRLS on the hierarchical basis with per-depth
    tau by EM on the working model (REGRESSION_DESIGN §3, binomial
    bridge). The design is built once per topology; each IRLS step
    reuses it with refreshed working weights w = n p(1-p) and response
    z = f + (k - n p)/w. The EM M-step uses posterior moments
    delta^2 + S_vv from the working-model covariance A^{-1}."""
    keys, X = mesh.design_hier(xs, ys)
    depths = np.array([mesh.verts[k].depth for k in keys])
    taus = {int(d): TAU for d in set(depths) if d > 0}
    delta = np.zeros(len(keys))
    Ainv = None
    for _ in range(em_iters):
        lam = np.array([0.0 if d == 0 else 1.0 / taus[int(d)] ** 2
                        for d in depths])
        for _ in range(irls_iters):
            f = X @ delta
            p = expit(f)
            w = np.maximum(nn * p * (1 - p), 1e-8)
            zw = f + (kk - nn * p) / w
            XtW = (X * w[:, None]).T
            Ainv = np.linalg.inv(XtW @ X + np.diag(lam))
            new_delta = Ainv @ (XtW @ zw)
            # DMESH_IRLS_STEP_CAP: damp the update so no pool's eta
            # moves more than IRLS_STEP_CAP logits in one step
            step = X @ (new_delta - delta)
            mx = float(np.max(np.abs(step))) if len(step) else 0.0
            frac = 1.0 if mx <= IRLS_STEP_CAP else IRLS_STEP_CAP / mx
            delta = delta + frac * (new_delta - delta)
        svv = np.diag(Ainv)
        for d in taus:
            m2 = delta[depths == d] ** 2 + svv[depths == d]
            taus[d] = max(float(np.sqrt(m2.mean())), TAU_FLOOR)
    mesh.resolve_heights_hier(dict(zip(keys, delta)))
    return taus, dict(zip(keys, np.diag(Ainv)))


def tau_at(taus, d):
    if not taus:
        return TAU
    return taus.get(d, taus[min(taus, key=lambda k: abs(k - d))])


def cand_stats(col, r, w, tau=TAU, parent_var=0.0):
    xTwx = float((w * col) @ col)
    if xTwx <= 0:
        return None
    w2b2 = float((w * w * col) @ col)
    keff = xTwx * xTwx / w2b2 if w2b2 > 0 else 0.0
    if keff < MIN_KEFF:
        return None
    xTwr = float((w * col) @ r)
    beta = xTwr / xTwx
    # gate_var (core/pipeline.cpp): data + GEV pool pricing + parent
    # posterior variance at PARENT_VAR_SCALE
    var = (1.0 / xTwx + GEV * w2b2 / xTwx ** 2
           + PARENT_VAR_SCALE * parent_var)
    info = 1.0 / var
    return {"beta": beta, "z": beta / np.sqrt(var), "info": info,
            "e": ebf(beta, info, tau)}


def seed_uniform(mesh, levels=3):
    """Unthresholded coarse base mesh (2^levels x 2^levels), promoted
    without gate decisions — the port of the triangular pipeline's
    'deliberately unthresholded coarse mesh is one statistical model'
    (core/pipeline.cpp): the gate governs refinements beyond it."""
    for _ in range(levels):
        for leaf in list(mesh.leaves):
            mesh.split(leaf, "x", origin="seed")
        for leaf in list(mesh.leaves):
            mesh.split(leaf, "y", origin="seed")
    for k, v in list(mesh.verts.items()):
        if v.state in ("weld", "hanging"):
            mesh.promote(k)


def run_gate_w(xs, ys, nn, kk):
    mesh = RectMesh()
    seed_uniform(mesh, levels=4)
    taus, svv = fit_binom_em(mesh, xs, ys, nn, kk)

    def parent_var(pkeys):
        return sum(svv.get(pk, 0.0) for pk in pkeys)

    admitted = []
    for _ in range(MAX_ROUNDS):
        if len(admitted) >= MAX_ADMIT:
            break
        pred = np.array([mesh.eval(x, y) for x, y in zip(xs, ys)])
        # binomial score-test working residuals and weights at the fit
        p_cur = expit(pred)
        ws = np.maximum(nn * p_cur * (1 - p_cur), 1e-8)
        r = (kk - nn * p_cur) / ws
        cands = {}
        for leaf in mesh.leaves:
            inside = ((xs * S > leaf.x0) & (xs * S <= leaf.x1)
                      & (ys * S > leaf.y0) & (ys * S <= leaf.y1))
            if inside.sum() < MIN_PTS:
                continue
            u = (xs[inside] * S - leaf.x0) / (leaf.x1 - leaf.x0)
            v = (ys[inside] * S - leaf.y0) / (leaf.y1 - leaf.y0)
            for axis in ("x", "y"):
                tent = 1 - np.abs(2 * (u if axis == "x" else v) - 1)
                other = v if axis == "x" else u
                for end, trans in ((0, 1 - other), (1, other)):
                    pos = (((leaf.x0 + leaf.x1) // 2,
                            leaf.y0 if end == 0 else leaf.y1) if axis == "x"
                           else (leaf.x0 if end == 0 else leaf.x1,
                                 (leaf.y0 + leaf.y1) // 2))
                    vx = mesh.verts.get(pos)
                    if vx is not None and vx.state == "free":
                        continue
                    if axis == "x":
                        pkeys = [(leaf.x0, pos[1]), (leaf.x1, pos[1])]
                    else:
                        pkeys = [(pos[0], leaf.y0), (pos[0], leaf.y1)]
                    st = cand_stats(tent * trans, r[inside], ws[inside],
                                    tau_at(taus, leaf.lx + leaf.ly + 1),
                                    parent_var(pkeys))
                    if st is None:
                        continue
                    prev = cands.get(pos)
                    if prev is None or abs(st["z"]) > abs(prev["z"]):
                        st.update({"leaf": leaf, "axis": axis})
                        cands[pos] = st
        for key, vx in list(mesh.verts.items()):
            if vx.state == "free":
                continue
            col = exact_column(mesh, key, xs, ys)
            if np.count_nonzero(col) < MIN_PTS:
                continue
            pkeys = [(m.ix, m.iy) for m, _ in mesh.masters(vx)]
            st = cand_stats(col, r, ws, tau_at(taus, vx.depth),
                            parent_var(pkeys))
            if st is None:
                continue
            st.update({"leaf": None, "axis": None})
            cands[key] = st
        if not cands:
            break
        keys = list(cands)
        # Production family decision: Lindsey lfdr with clamped
        # empirical null, mean-lfdr prefix; BH fallback (lfdr_admit).
        admit_idx = lfdr_admit([cands[k]["z"] for k in keys], Q)
        admit_idx.sort(key=lambda i: -abs(cands[keys[i]]["z"]))
        committed = 0
        for i in admit_idx:
            if len(admitted) >= MAX_ADMIT:
                break
            pos, c = keys[i], cands[keys[i]]
            vx = mesh.verts.get(pos)
            if vx is not None and vx.state == "free":
                continue
            if vx is None:
                if c["leaf"] is None or not c["leaf"].leaf:
                    continue
                mesh.split(c["leaf"], c["axis"], origin="selected")
            mesh.promote(pos)
            pred_live = np.array([mesh.eval(x, y) for x, y in zip(xs, ys)])
            col = exact_column(mesh, pos, xs, ys)
            p_live = expit(pred_live)
            w_live = np.maximum(nn * p_live * (1 - p_live), 1e-8)
            vlive = mesh.verts[pos]
            pk_live = [(m.ix, m.iy) for m, _ in mesh.masters(vlive)] \
                if vlive.state != "free" else \
                [(m.ix, m.iy) for m, _ in vlive.birth_parents]
            st = cand_stats(col, (kk - nn * p_live) / w_live, w_live,
                            tau_at(taus, vlive.depth), parent_var(pk_live))
            if st is None or np.sign(st["beta"]) != np.sign(c["beta"]) \
                    or st["e"] <= 1.0:
                mesh.verts[pos].state = "weld"
                continue
            taus, svv = fit_binom_em(mesh, xs, ys, nn, kk)
            admitted.append(pos)
            committed += 1
        if committed == 0:
            break
    return mesh, admitted


def binned(xs, ys, vals, ws, nb=48):
    num = np.zeros((nb, nb))
    den = np.zeros((nb, nb))
    ix = np.clip((xs * nb).astype(int), 0, nb - 1)
    iy = np.clip((ys * nb).astype(int), 0, nb - 1)
    np.add.at(num, (iy, ix), ws * vals)
    np.add.at(den, (iy, ix), ws)
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def deviance_per_pool(n, k, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(k > 0, k * np.log(k / (n * p)), 0.0)
        t2 = np.where(n - k > 0, (n - k) * np.log((n - k) / (n - n * p)), 0.0)
    return float(np.mean(2 * (t1 + t2)))


def run_dataset(fname, tag, title):
    rng = np.random.default_rng(7)
    x, y, z, w, n, k, hold = load(fname, rng)
    tr = ~hold
    mesh, admitted = run_gate_w(x[tr], y[tr], n[tr], k[tr])
    pred = np.array([mesh.eval(xi, yi) for xi, yi in zip(x, y)])
    p_hat = expit(pred)
    rate0 = k[tr].sum() / n[tr].sum()
    res = {
        "n_pools_subsampled": int(len(x)), "n_train": int(tr.sum()),
        "n_heldout": int(hold.sum()), "n_admitted": len(admitted),
        "n_leaves": len(mesh.leaves),
        "heldout_deviance_per_pool": deviance_per_pool(
            n[hold], k[hold], p_hat[hold]),
        "heldout_deviance_per_pool_constant_baseline": deviance_per_pool(
            n[hold], k[hold], np.full(int(hold.sum()), rate0)),
        "stack": "binomial IRLS",
        "caveats": "Exact-binomial PoC on a 12k-pool subsample, "
                   "unthresholded 16x16 coarse seed, per-depth EM tau (floor "
                   "0.005), production gate config: Lindsey lfdr family "
                   "decision (sigma0 [0.3,6], pi0>=0.05, BH fallback), 80 "
                   "rounds, GEV=0.35 + 0.25x parent posterior variance in "
                   "the null, keff>=0.5, >=10 fit points, IRLS step cap 4, "
                   "locally balanced holdout; no pool effects in the fit, "
                   "no law machinery.",
    }
    obs = binned(x, y, z, w)
    gg = (np.arange(96) + 0.5) / 96
    fgrid = np.array([[mesh.eval(xi, yi) for xi in gg] for yi in gg])
    # mask the fitted/residual panels where no data constrains the fit
    presence = ~np.isnan(obs)
    mask96 = np.repeat(np.repeat(presence, 2, 0), 2, 1)
    fgrid = np.where(mask96, fgrid, np.nan)
    lo, hi = float(np.nanmin(obs)), float(np.nanmax(obs))
    heatmap_svg(obs, f"{tag}_obs.svg",
                f"{title}: observed logit (binned, weighted)", lo, hi)
    heatmap_svg(fgrid, f"{tag}_fitted.svg",
                f"{title}: fitted logit surface ({len(admitted)} admitted)",
                lo, hi)
    resid = binned(x, y, z - pred, w)
    heatmap_svg(resid, f"{tag}_resid.svg", f"{title}: binned residual logit")
    mesh_svg(mesh, f"{tag}_mesh.svg",
             f"{title}: {len(mesh.leaves)} leaves, {len(admitted)} admissions")
    print(tag, res, flush=True)
    return res


def main():
    gauss_ref = {}
    ref_path = os.path.join(OUT, "ginnie_results.json")
    if os.path.exists(ref_path):
        prev = json.load(open(ref_path))
        for kk_ in ("cumulative", "smm"):
            if kk_ in prev and "stack" not in prev[kk_]:
                gauss_ref[kk_] = prev[kk_]["heldout_deviance_per_pool"]
    results = {
        "cumulative": run_dataset("ginnie_design.csv", "ginnie",
                                  "Ginnie cumulative attrition"),
        "smm": run_dataset("smm_design_202606.csv", "smm",
                           "Ginnie monthly SMM"),
    }
    for kk_, v in gauss_ref.items():
        results[kk_]["gaussian_surrogate_reference"] = v
    with open(os.path.join(OUT, "ginnie_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
