"""First-look rectangular-mesh fits of the frozen Ginnie designs:
cumulative attrition proxy (data/ginnie_design.csv, pooled rate 0.7832)
and monthly SMM (data/smm_design_202606.csv, rate 0.0085).

This is the Gaussian PoC stack on empirical logits, NOT the production
binomial pipeline: response z = log((k+.5)/(n-k+.5)), weight
w = n*p~(1-p~), fitted by the weighted penalized hierarchical solve, and
gated by the e-BH round structure with the triangular pool-variance
pricing ported into the candidate null:
    Var(beta_hat) = 1/xTwx + GEV * sum(w^2 b^2) / xTwx^2   (GEV = 0.35)
(core/pipeline.cpp's gate_var, weighted-Gaussian form). Pools are
subsampled for PoC runtime; all caveats printed into the results JSON.
"""

import json
import os
import numpy as np

from rect_mesh import RectMesh, S
from run_poc import heatmap_svg, mesh_svg
from gate_poc import ebh_admit, exact_column

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

GEV = 0.35
TAU = 0.5
Q = 0.1
MAX_ADMIT = 60
MAX_ROUNDS = 8
MIN_PTS = 40
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
    pt = (k + 0.5) / (n + 1.0)
    z = np.log((k + 0.5) / (n - k + 0.5))
    w = n * pt * (1 - pt)
    n_hold = int(HOLD_FRAC * len(d))
    hold = np.zeros(len(d), bool)
    hold[rng.permutation(len(d))[:n_hold]] = True
    return x, y, z, w, n, k, hold


def ebf(beta, info, tau):
    """e-value: Bayes factor for beta_hat ~ N(delta, 1/info),
    delta ~ N(0, tau^2) vs delta = 0."""
    t2 = tau * tau
    return np.sqrt(1.0 / (1.0 + t2 * info)) * np.exp(
        min(700.0, t2 * info * info * beta * beta / (2 * (1.0 + t2 * info))))


def fit_w(mesh, xs, ys, zs, ws, tau):
    keys, X = mesh.design_hier(xs, ys)
    lam = np.array([0.0 if mesh.verts[k].depth == 0 else 1.0 / tau ** 2
                    for k in keys])
    A = (X * ws[:, None]).T @ X + np.diag(lam)
    delta = np.linalg.solve(A, (X * ws[:, None]).T @ zs)
    mesh.resolve_heights_hier(dict(zip(keys, delta)))


def cand_stats(col, r, w):
    xTwx = float((w * col) @ col)
    if xTwx <= 0:
        return None
    xTwr = float((w * col) @ r)
    beta = xTwr / xTwx
    var = 1.0 / xTwx + GEV * float((w * w * col) @ col) / xTwx ** 2
    info = 1.0 / var
    return {"beta": beta, "z": beta / np.sqrt(var), "info": info,
            "e": ebf(beta, info, TAU)}


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


def run_gate_w(xs, ys, zs, ws):
    mesh = RectMesh()
    seed_uniform(mesh)
    fit_w(mesh, xs, ys, zs, ws, TAU)
    admitted = []
    for _ in range(MAX_ROUNDS):
        if len(admitted) >= MAX_ADMIT:
            break
        pred = np.array([mesh.eval(x, y) for x, y in zip(xs, ys)])
        r = zs - pred
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
                    st = cand_stats(tent * trans, r[inside], ws[inside])
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
            st = cand_stats(col, r, ws)
            if st is None:
                continue
            st.update({"leaf": None, "axis": None})
            cands[key] = st
        if not cands:
            break
        keys = list(cands)
        admit_idx = ebh_admit([cands[k]["e"] for k in keys], Q)
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
            st = cand_stats(col, zs - pred_live, ws)
            if st is None or np.sign(st["beta"]) != np.sign(c["beta"]) \
                    or st["e"] <= 1.0:
                mesh.verts[pos].state = "weld"
                continue
            fit_w(mesh, xs, ys, zs, ws, TAU)
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
    out = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    fill = np.nansum(num) / max(np.sum(den), 1e-300)
    return np.where(np.isnan(out), fill, out)


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
    mesh, admitted = run_gate_w(x[tr], y[tr], z[tr], w[tr])
    pred = np.array([mesh.eval(xi, yi) for xi, yi in zip(x, y)])
    p_hat = 1 / (1 + np.exp(-pred))
    rate0 = k[tr].sum() / n[tr].sum()
    res = {
        "n_pools_subsampled": int(len(x)), "n_train": int(tr.sum()),
        "n_heldout": int(hold.sum()), "n_admitted": len(admitted),
        "n_leaves": len(mesh.leaves),
        "heldout_deviance_per_pool": deviance_per_pool(
            n[hold], k[hold], p_hat[hold]),
        "heldout_deviance_per_pool_constant_baseline": deviance_per_pool(
            n[hold], k[hold], np.full(int(hold.sum()), rate0)),
        "caveats": "Gaussian empirical-logit PoC on a 12k-pool subsample, "
                   "unthresholded 8x8 coarse seed, fixed tau, GEV=0.35; "
                   "not the production binomial pipeline.",
    }
    obs = binned(x, y, z, w)
    gg = (np.arange(96) + 0.5) / 96
    fgrid = np.array([[mesh.eval(xi, yi) for xi in gg] for yi in gg])
    lo, hi = float(obs.min()), float(obs.max())
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
    results = {
        "cumulative": run_dataset("ginnie_design.csv", "ginnie",
                                  "Ginnie cumulative attrition"),
        "smm": run_dataset("smm_design_202606.csv", "smm",
                           "Ginnie monthly SMM"),
    }
    with open(os.path.join(OUT, "ginnie_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
