"""Gate proof of concept: FDR machinery ported from the triangular mesh
(docs/FDR_PORT.md), Gaussian instantiation, with measured FDP.

Round structure (triangular-faithful):
  batch-score candidates -> family decision (BH on p / e-BH on Bayes
  factors) -> sequential commits with composite materialization (split +
  release, all welded) and a live re-profile on the exact hierarchical
  column (B1 invariant: live Bayes factor > 1 or no admission) -> global
  penalized refit (shrinkage only through Lambda). Rounds until a round
  admits nothing.

Worlds (docs/FDR_PORT.md §5): W0 complete null (every admission false),
W1 coarse bilinear truth (exact nulls beyond depth 2, exact labels via
nestedness), W2 chirp (power/adaptivity only).

PoC simplifications, recorded: new-cut batch scores use the
within-cell tent column (the triangular loc_regress analogue) while
existing welds/hanging vertices are scored on their exact live columns;
sigma is known to the fitter; tau is fixed, not EM-estimated; rejected
candidates' inert geometry is not yet garbage-collected. Parked welds
MUST be candidates: leaving them unpromotable strands their true signal
in the residual and the gate then buys the same structure as a
staircase of false admissions nearby (measured on W1, seed 4: 20 false
admissions collapsing to 1 once welds are candidates).
"""

import json
import os
import numpy as np

from rect_mesh import RectMesh, S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------- truth worlds

def chirp(x, y):
    """Triangular core/surface.h chirp, unit-square inputs."""
    y_sim = -4.0 + 8.0 * y
    z_warp = (0.2 + 0.8 * x) * x
    xr = -4.0 + 40.0 * z_warp
    coarse = np.exp(0.80 * np.cos(0.72 * xr + 0.44 * y_sim + 0.25)) - 1.1665149229
    fine = np.exp(0.58 * np.cos(2.65 * (-0.36 * xr + 0.93 * y_sim) - 0.40)) - 1.0859474
    return 2.0 * (0.62 * coarse + 0.38 * fine)


def make_coarse_truth(rng):
    """Random bilinear truth on a depth-2 mesh: exactly representable at
    every finer topology, so all deeper deltas are exactly zero."""
    t = RectMesh()
    t.split(t.root, "x")
    for leaf in list(t.leaves):
        t.split(leaf, "y")
    for k, v in list(t.verts.items()):
        if v.state in ("weld", "hanging"):
            t.promote(k)
    for v in t.verts.values():
        v.height = rng.normal(0, 0.7)
    # nodal heights are free draws; the surface is what it is
    t.resolve_heights()
    return t


# ------------------------------------------------------------- gate pieces

def bayes_factor(xTr, xTx, sigma, tau):
    s2, t2 = sigma * sigma, tau * tau
    denom = s2 + t2 * xTx
    return np.sqrt(s2 / denom) * np.exp(
        min(700.0, t2 * xTr * xTr / (2 * s2 * denom)))


def normal_sf(z):
    from math import erfc, sqrt
    return 0.5 * erfc(z / sqrt(2.0))


def bh_admit(pvals, q):
    """Indices admitted by Benjamini-Hochberg."""
    order = np.argsort(pvals)
    n = len(pvals)
    k = 0
    for i, oi in enumerate(order):
        if pvals[oi] <= q * (i + 1) / n:
            k = i + 1
    return list(order[:k])


def ebh_admit(evals, q):
    """Indices admitted by e-BH: k* = max k with e_(k) >= n/(q k)."""
    order = np.argsort(-np.asarray(evals))
    n = len(evals)
    k = 0
    for i, oi in enumerate(order):
        if evals[oi] >= n / (q * (i + 1)):
            k = i + 1
    return list(order[:k])


def exact_column(mesh, key, xs, ys):
    saved = {k: mesh.verts[k].height for k in mesh.verts}
    mesh.resolve_heights_hier({key: 1.0})
    col = np.array([mesh.eval(x, y) for x, y in zip(xs, ys)])
    for k, h in saved.items():
        mesh.verts[k].height = h
    return col


def fit_hier(mesh, xs, ys, zs, sigma, tau):
    """Penalized joint solve in the hierarchical basis: roots flat, all
    other free surpluses ridged at sigma^2/tau^2."""
    keys, X = mesh.design_hier(xs, ys)
    lam = np.array([0.0 if mesh.verts[k].depth == 0 else (sigma / tau) ** 2
                    for k in keys])
    A = X.T @ X + np.diag(lam)
    delta = np.linalg.solve(A, X.T @ np.asarray(zs))
    mesh.resolve_heights_hier(dict(zip(keys, delta)))


# ------------------------------------------------------------- gate rounds

def run_gate(truth_f, rng, arm="ebh", n=2000, sigma=0.1, q=0.1, tau=0.5,
             max_rounds=15, min_pts=12):
    xs = rng.random(n)
    ys = rng.random(n)
    f_true = truth_f(xs, ys)
    zs = f_true + sigma * rng.normal(size=n)

    mesh = RectMesh()
    fit_hier(mesh, xs, ys, zs, sigma, tau)
    admitted = []  # (key, depth, delta_true)

    for _ in range(max_rounds):
        pred = np.array([mesh.eval(x, y) for x, y in zip(xs, ys)])
        r = zs - pred
        # ---- batch scoring: chord-endpoint candidates of every leaf cut
        cands = {}
        for leaf in mesh.leaves:
            inside = ((xs * S > leaf.x0) & (xs * S <= leaf.x1)
                      & (ys * S > leaf.y0) & (ys * S <= leaf.y1))
            if inside.sum() < min_pts:
                continue
            u = (xs[inside] * S - leaf.x0) / (leaf.x1 - leaf.x0)
            w = (ys[inside] * S - leaf.y0) / (leaf.y1 - leaf.y0)
            ri = r[inside]
            for axis in ("x", "y"):
                tent = 1 - np.abs(2 * (u if axis == "x" else w) - 1)
                other = w if axis == "x" else u
                for end, trans in ((0, 1 - other), (1, other)):
                    if axis == "x":
                        pos = ((leaf.x0 + leaf.x1) // 2,
                               leaf.y0 if end == 0 else leaf.y1)
                    else:
                        pos = (leaf.x0 if end == 0 else leaf.x1,
                               (leaf.y0 + leaf.y1) // 2)
                    v = mesh.verts.get(pos)
                    if v is not None and v.state == "free":
                        continue
                    col = tent * trans
                    xTx = float(col @ col)
                    if xTx <= 0:
                        continue
                    xTr = float(col @ ri)
                    zst = xTr / (sigma * np.sqrt(xTx))
                    prev = cands.get(pos)
                    if prev is None or abs(zst) > abs(prev["z"]):
                        cands[pos] = {"leaf": leaf, "axis": axis, "z": zst,
                                      "xTx": xTx, "xTr": xTr}
        # ---- existing non-free vertices (parked welds, hanging) are
        # candidates too, scored on their exact live columns; without this
        # their true signal is stranded in the residual and re-bought as
        # false admissions nearby.
        for key, v in list(mesh.verts.items()):
            if v.state == "free":
                continue
            col = exact_column(mesh, key, xs, ys)
            npts = int(np.count_nonzero(col))
            if npts < min_pts:
                continue
            xTx = float(col @ col)
            if xTx <= 0:
                continue
            xTr = float(col @ r)
            zst = xTr / (sigma * np.sqrt(xTx))
            cands[key] = {"leaf": None, "axis": None, "z": zst,
                          "xTx": xTx, "xTr": xTr}
        if not cands:
            break
        keys = list(cands)
        # ---- family decision
        if arm == "bh":
            p = [2 * normal_sf(abs(cands[k]["z"])) for k in keys]
            admit_idx = bh_admit(np.array(p), q)
        else:
            e = [bayes_factor(cands[k]["xTr"], cands[k]["xTx"], sigma, tau)
                 for k in keys]
            admit_idx = ebh_admit(e, q)
        admit_idx.sort(key=lambda i: -abs(cands[keys[i]]["z"]))
        # ---- sequential commits with live re-profile (B1: live BF > 1)
        committed = 0
        for i in admit_idx:
            pos, c = keys[i], cands[keys[i]]
            v = mesh.verts.get(pos)
            if v is not None and v.state == "free":
                continue
            if v is None:
                if c["leaf"] is None or not c["leaf"].leaf:
                    continue  # stale leaf, nothing materialized: next round
                mesh.split(c["leaf"], c["axis"], origin="selected")
            mesh.promote(pos)  # composite: release splits as needed
            pred_live = np.array([mesh.eval(x, y) for x, y in zip(xs, ys)])
            col = exact_column(mesh, pos, xs, ys)
            xTx = float(col @ col)
            xTr = float(col @ (zs - pred_live))
            live_ok = (xTx > 0
                       and np.sign(xTr) == np.sign(c["xTr"])
                       and bayes_factor(xTr, xTx, sigma, tau) > 1.0)
            if not live_ok:
                mesh.verts[pos].state = "weld"  # demote; geometry stays inert
                continue
            # Refit after every accepted commit so later live re-profiles
            # in this round see deconfounded residuals (the triangular
            # "re-profile at the actual commit state", made global).
            fit_hier(mesh, xs, ys, zs, sigma, tau)
            vv = mesh.verts[pos]
            d_true = abs(truth_f(np.array([vv.x]), np.array([vv.y]))[0]
                         - 0.5 * sum(truth_f(np.array([m.x]), np.array([m.y]))[0]
                                     for m, _ in vv.birth_parents))
            admitted.append((pos, vv.depth, float(d_true)))
            committed += 1
        if committed == 0:
            break
        fit_hier(mesh, xs, ys, zs, sigma, tau)

    return mesh, admitted, (xs, ys, zs, f_true)


# ------------------------------------------------------------- experiments

def replicate(world, arm, reps, seed0, **kw):
    out = {"fdp": [], "n_admit": [], "n_false": [], "n_true": [],
           "false_depths": []}
    for rep in range(reps):
        rng = np.random.default_rng(seed0 + rep)
        if world == "w0":
            truth = lambda x, y: np.zeros_like(x)
            n_signal = 0
        else:
            t = make_coarse_truth(rng)
            truth = lambda x, y: np.array(
                [t.eval(xi, yi) for xi, yi in zip(np.atleast_1d(x),
                                                  np.atleast_1d(y))])
            n_signal = sum(1 for v in t.verts.values()
                           if v.depth > 0 and v.state == "free")
        _, admitted, _ = run_gate(truth, rng, arm=arm, **kw)
        false = [a for a in admitted if a[2] < 1e-9]
        true = [a for a in admitted if a[2] >= 1e-9]
        out["fdp"].append(len(false) / max(1, len(admitted)))
        out["n_admit"].append(len(admitted))
        out["n_false"].append(len(false))
        out["n_true"].append(len(true))
        out["false_depths"] += [a[1] for a in false]
        if world == "w1":
            out.setdefault("power", []).append(
                len(true) / max(1, n_signal))
    res = {"reps": reps,
           "realized_FDR": float(np.mean(out["fdp"])),
           "mean_admissions": float(np.mean(out["n_admit"])),
           "mean_false": float(np.mean(out["n_false"]))}
    if "power" in out:
        res["mean_power"] = float(np.mean(out["power"]))
        res["false_admission_depths"] = {
            str(d): out["false_depths"].count(d)
            for d in sorted(set(out["false_depths"]))}
    return res


def main():
    q = 0.1
    results = {"q_nominal": q}
    for arm in ("bh", "ebh"):
        results[f"w0_{arm}"] = replicate(
            "w0", arm, reps=40, seed0=1000, n=2000, sigma=0.1, q=q, tau=0.5)
        print(f"w0 {arm}: {results[f'w0_{arm}']}", flush=True)
    for arm in ("bh", "ebh"):
        results[f"w1_{arm}"] = replicate(
            "w1", arm, reps=20, seed0=5000, n=2000, sigma=0.15, q=q, tau=0.5)
        print(f"w1 {arm}: {results[f'w1_{arm}']}", flush=True)
    # W2: chirp power/adaptivity (no FDR claims: no true nulls)
    for arm in ("bh", "ebh"):
        rng = np.random.default_rng(7)
        mesh, admitted, (xs, ys, zs, f_true) = run_gate(
            chirp, rng, arm=arm, n=4000, sigma=0.3, q=q, tau=1.0,
            max_rounds=25)
        gx, gy = np.meshgrid((np.arange(96) + 0.5) / 96,
                             (np.arange(96) + 0.5) / 96)
        tgrid = chirp(gx.ravel(), gy.ravel())
        fgrid = np.array([mesh.eval(x, y)
                          for x, y in zip(gx.ravel(), gy.ravel())])
        results[f"w2_{arm}"] = {
            "n_admitted": len(admitted),
            "n_leaves": len(mesh.leaves),
            "rmse_vs_truth_grid": float(np.sqrt(np.mean((fgrid - tgrid) ** 2))),
        }
        print(f"w2 {arm}: {results[f'w2_{arm}']}", flush=True)
    with open(os.path.join(OUT, "gate_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
