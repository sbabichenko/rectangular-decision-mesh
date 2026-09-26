#!/usr/bin/env python3
"""The standard 2x2 experiment figure (the triangular repo's analysis/make_2x2_logodds.py) for a
rectangular run: observed log-odds / the fitted bilinear surface on its cells / held-out error /
the mesh. Coordinates come from the design csv (row-aligned with the obs dump). Usage:
  python3 -m analysis.make_2x2_logodds <design.csv> <dump_prefix> <title> <out.png>"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection

from pyrect.estimator.neutral import mean_neutral


def lo_(p):
    return np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))


def hanging_vertices(cells, verts):
    """Vertices on the interior of some leaf's edge: their height is that edge's average."""
    vx = verts.x.values[:, None]; vy = verts.y.values[:, None]; e = 1e-12
    x0, x1, y0, y1 = (cells[c].values for c in ("x0", "x1", "y0", "y1"))
    on_v = ((np.abs(vx - x0) < e) | (np.abs(vx - x1) < e)) & (vy > y0 + e) & (vy < y1 - e)
    on_h = ((np.abs(vy - y0) < e) | (np.abs(vy - y1) < e)) & (vx > x0 + e) & (vx < x1 - e)
    return (on_v | on_h).any(axis=1)


def bilinear_raster(cells, nx, ny):
    """The fitted surface on an nx x ny pixel grid over the unit square, bilinear within each cell
    (no triangulation: splitting a rectangle into triangles would draw diagonals the model lacks)."""
    out = np.full((ny, nx), np.nan)
    cx = (np.arange(nx) + 0.5) / nx; cy = (np.arange(ny) + 0.5) / ny
    for r in cells.itertuples():
        i0, i1 = np.searchsorted(cx, r.x0), np.searchsorted(cx, r.x1, side="right")
        j0, j1 = np.searchsorted(cy, r.y0), np.searchsorted(cy, r.y1, side="right")
        if i1 <= i0 or j1 <= j0: continue
        u = (cx[i0:i1] - r.x0) / (r.x1 - r.x0); t = (cy[j0:j1] - r.y0) / (r.y1 - r.y0)
        U, T = np.meshgrid(u, t)
        out[j0:j1, i0:i1] = (1 - U) * (1 - T) * r.h00 + U * (1 - T) * r.h10 + (1 - U) * T * r.h01 + U * T * r.h11
    return out


def trilinear_slice(cells, z, nx, ny):
    """The fitted surface on an nx x ny pixel grid over the unit square at third coordinate z:
    trilinear in each cell the slice cuts."""
    out = np.full((ny, nx), np.nan)
    cx = (np.arange(nx) + 0.5) / nx; cy = (np.arange(ny) + 0.5) / ny
    for r in cells[(cells.z0 <= z) & (cells.z1 >= z)].itertuples():
        i0, i1 = np.searchsorted(cx, r.x0), np.searchsorted(cx, r.x1, side="right")
        j0, j1 = np.searchsorted(cy, r.y0), np.searchsorted(cy, r.y1, side="right")
        if i1 <= i0 or j1 <= j0: continue
        u = (cx[i0:i1] - r.x0) / (r.x1 - r.x0); t = (cy[j0:j1] - r.y0) / (r.y1 - r.y0); s = (z - r.z0) / (r.z1 - r.z0)
        U, T = np.meshgrid(u, t)
        h = [getattr(r, f"h{a}{b}{c}") for c in "01" for b in "01" for a in "01"]
        out[j0:j1, i0:i1] = ((1 - U) * (1 - T) * (1 - s) * h[0] + U * (1 - T) * (1 - s) * h[1] + (1 - U) * T * (1 - s) * h[2] + U * T * (1 - s) * h[3]
                             + (1 - U) * (1 - T) * s * h[4] + U * (1 - T) * s * h[5] + (1 - U) * T * s * h[6] + U * T * s * h[7])
    return out


def main3d(d, obs, model, prefix, title, out):
    """A three-dimensional run: one row per balance tercile (by loans), each with the observed
    log-odds, the fitted surface sliced at the band's median balance, the held-out z and the mesh
    slice there."""
    H = model["height_scale"]; eta0 = model["baseline_logit"]
    w0, w1, c0, c1 = model["wala_min"], model["wala_max"], model["wac_min"], model["wac_max"]
    a0, a1 = model["axis3_min"], model["axis3_max"]
    to_x = lambda s: w0 + s * (w1 - w0); to_y = lambda s: c0 + s * (c1 - c0)
    wala, wac = d.wala.values, d.wac.values
    n, k, phat = d.n0.values.astype(float), d.k_term.values.astype(float), obs.p_hat.values
    z = np.clip((d[model["axis3"]].values - a0) / (a1 - a0), 0, 1)
    cells = pd.read_csv(f"{prefix}_mesh.csv"); verts = pd.read_csv(f"{prefix}_hier_vertices.csv")
    MINMASS = 300.0
    lo_w, hi_w, lo_c, hi_c = 0.0, wala.max() * 1.02, wac.min() - 0.2, wac.max() + 0.2
    BX, BY = 72, 56; rng = [[lo_w, hi_w], [lo_c, hi_c]]; ext = [lo_w, hi_w, lo_c, hi_c]
    order = np.argsort(z); cum = np.cumsum(n[order]) / n.sum()
    cuts = [z[order][np.searchsorted(cum, q)] for q in (1 / 3, 2 / 3)]
    band = np.searchsorted(cuts, z, side="right")
    held = ~obs["is_train"].astype(bool).values
    s2 = json.load(open(f"{prefix}_final_fit.json"))["pool_variance"]
    A = mean_neutral(lo_(phat), np.full(len(phat), np.sqrt(s2)), 40)[0]
    gx, gw = np.polynomial.hermite.hermgauss(40)
    pk = 1 / (1 + np.exp(-(A[:, None] + np.sqrt(2 * s2) * gx[None, :])))
    m1 = (gw[None, :] / np.sqrt(np.pi) * pk).sum(axis=1); m2 = (gw[None, :] / np.sqrt(np.pi) * pk * pk).sum(axis=1)
    var_all = n * m1 * (1 - m1) + n * (n - 1) * (m2 - m1 * m1)
    fig, ax = plt.subplots(3, 4, figsize=(26, 19), dpi=150, gridspec_kw=dict(hspace=0.3, wspace=0.25))
    def hist(m, weights): return np.histogram2d(wala[m], wac[m], bins=[BX, BY], range=rng, weights=weights[m])[0]
    all_obs = []; all_fit = []
    for b in range(3):
        m = band == b; mass = hist(m, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            o = lo_(hist(m, k) / mass); f = lo_(hist(m, n * phat) / mass)
        o[(mass < MINMASS) | ~np.isfinite(o)] = np.nan; f[mass < MINMASS] = np.nan
        all_obs.append(o); all_fit.append(f)
    v0o, v1o = np.nanpercentile(np.concatenate([x.ravel() for x in all_obs]), [2, 98]); v0f, v1f = np.nanpercentile(np.concatenate([x.ravel() for x in all_fit]), [1, 99])
    for b in range(3):
        m = band == b; mass = hist(m, n)
        zb = float(np.median(z[m])); bal = np.exp(a0 + zb * (a1 - a0))
        lo_b = np.exp(a0 + z[m].min() * (a1 - a0)); hi_b = np.exp(a0 + z[m].max() * (a1 - a0))
        name = f"balance ${lo_b / 1e3:,.0f}k to ${hi_b / 1e3:,.0f}k ({100 * n[m].sum() / n.sum():.0f}% of loans)".replace("$", r"\$")
        im0 = ax[b, 0].imshow(all_obs[b].T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=v0o, vmax=v1o)
        ax[b, 0].set_title(f"Observed log-odds, {name}", fontsize=10)
        surf = eta0 + trilinear_slice(cells, zb, 700, 540) / H
        im1 = ax[b, 1].imshow(surf, origin="lower", extent=[w0, w1, c0, c1], aspect="auto", cmap="viridis", vmin=v0f, vmax=v1f, interpolation="nearest")
        ax[b, 1].set_xlim(lo_w, hi_w); ax[b, 1].set_ylim(lo_c, hi_c)
        ax[b, 1].set_title(f"Fitted surface sliced at the band's median balance, " + f"${bal / 1e3:,.0f}k".replace("$", r"\$"), fontsize=10)
        num, den = hist(m & held, k - n * phat), hist(m & held, var_all)
        zmap = np.where(den > 0, num / np.sqrt(np.maximum(den, 1e-12)), np.nan); zmap[mass < MINMASS] = np.nan
        im2 = ax[b, 2].imshow(zmap.T, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
        ax[b, 2].set_title("Held-out miss over the model's sd, per bin (z)", fontsize=10)
        cut = cells[(cells.z0 <= zb) & (cells.z1 >= zb)]
        boxes = [[(to_x(r.x0), to_y(r.y0)), (to_x(r.x1), to_y(r.y0)), (to_x(r.x1), to_y(r.y1)), (to_x(r.x0), to_y(r.y1))] for r in cut.itertuples()]
        empty = cut.n_pts.values == 0
        a = ax[b, 3]
        a.add_collection(PolyCollection([x for x, e in zip(boxes, empty) if not e], facecolors="none", edgecolors="#14213d", linewidths=0.35, alpha=0.7))
        a.add_collection(PolyCollection([x for x, e in zip(boxes, empty) if e], facecolors="#f2f2f2", edgecolors="#9a9a9a", linewidths=0.25, alpha=0.8))
        # the vertices lying in a cell face at this height, or nearest in the band: those at the cut cells' own z levels
        levels = np.unique(np.r_[cut.z0.values, cut.z1.values])
        near = levels[np.argmin(np.abs(levels - zb))]
        vs = verts[np.abs(verts.z - near) < 1e-9]
        free = vs.free_coefficient.values == 1; adm = vs.gate_admitted.values == 1
        a.scatter(to_x(vs.x.values[~free]), to_y(vs.y.values[~free]), s=6, facecolors="none", edgecolors="#7a7a7a", linewidths=0.4)
        a.scatter(to_x(vs.x.values[free & ~adm]), to_y(vs.y.values[free & ~adm]), s=30, c="#e0a100", edgecolors="white", linewidths=0.3)
        a.scatter(to_x(vs.x.values[free & adm]), to_y(vs.y.values[free & adm]), s=14, c="#14213d", edgecolors="white", linewidths=0.3)
        a.set_xlim(lo_w, hi_w); a.set_ylim(lo_c, hi_c)
        a.set_title(f"Mesh slice: {len(cut)} cells; vertices at " + f"${np.exp(a0 + near * (a1 - a0)) / 1e3:,.0f}k".replace("$", r"\$") + " (dark: admitted, gold: backbone)", fontsize=10)
        for c_ in range(4): ax[b, c_].set_xlabel("WALA (months)"); ax[b, c_].set_ylabel("WAC (%)")
    for col, im, lab in ((0, im0, "observed log-odds"), (1, im1, "fitted log-odds"), (2, im2, "z (fast +, slow -)")):
        fig.colorbar(im, ax=ax[:, col], orientation="horizontal", shrink=0.85, pad=0.04, aspect=35).set_label(lab)
    fig.suptitle(title + f"   (3-D mesh: {len(cells)} cells, {int((verts.free_coefficient == 1).sum())} free coefficients)", fontsize=17)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}: {len(cells)} cells, {len(verts)} vertices")


def main(argv=None):
    data_csv, prefix, title, out = (argv or sys.argv[1:])[:4]
    d = pd.read_csv(data_csv)
    obs = pd.read_csv(f"{prefix}_obs.csv")
    model = json.load(open(f"{prefix}_model.json"))
    H = model["height_scale"]
    assert len(d) == len(obs), "row alignment"
    if "axis3" in model:
        return main3d(d, obs, model, prefix, title, out)
    wala, wac = d.wala.values, d.wac.values
    n, k, phat = d.n0.values.astype(float), d.k_term.values.astype(float), obs.p_hat.values
    # the run's own unit square, so the surface and the mesh sit where the fit put them
    w0, w1, c0, c1 = model["wala_min"], model["wala_max"], model["wac_min"], model["wac_max"]
    to_x = lambda s: w0 + s * (w1 - w0); to_y = lambda s: c0 + s * (c1 - c0)
    MINMASS = 300.0   # loans per bin below which a bin is masked (kills small-sample speckle)
    # the triangular script's bins exactly, so the two figures' binned panels compare bin for bin
    lo_w, hi_w, lo_c, hi_c = 0.0, wala.max() * 1.02, wac.min() - 0.2, wac.max() + 0.2
    BX, BY = 72, 56
    rng = [[lo_w, hi_w], [lo_c, hi_c]]
    def hist(weights): return np.histogram2d(wala, wac, bins=[BX, BY], range=rng, weights=weights)[0]
    def binned(num, den):
        with np.errstate(invalid="ignore", divide="ignore"): return hist(num) / hist(den)
    ext = [lo_w, hi_w, lo_c, hi_c]

    fig, ax = plt.subplots(2, 2, figsize=(19.5, 12), dpi=200)
    mass = hist(n)
    obs_surf = lo_(binned(k, n)); obs_surf[(mass < MINMASS) | ~np.isfinite(obs_surf)] = np.nan
    fit_surf = lo_(binned(n * phat, n)); fit_surf[mass < MINMASS] = np.nan
    v0o, v1o = np.percentile(obs_surf[np.isfinite(obs_surf)], [2, 98]); v0o, v1o = np.floor(v0o * 2) / 2, np.ceil(v1o * 2) / 2
    v0f, v1f = np.percentile(fit_surf[np.isfinite(fit_surf)], [1, 99]); v0f, v1f = np.floor(v0f * 10) / 10, np.ceil(v1f * 10) / 10
    im0 = ax[0, 0].imshow(obs_surf.T, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=v0o, vmax=v1o)
    ax[0, 0].set_title("Observed termination (log-odds)")

    # the actual element function: bilinear on each rectangle, rastered finely, cell edges over it
    cells = pd.read_csv(f"{prefix}_mesh.csv")
    eta0 = model["baseline_logit"]
    surf = eta0 + bilinear_raster(cells, 900, 700) / H
    im1 = ax[0, 1].imshow(surf, origin="lower", extent=[w0, w1, c0, c1], aspect="auto", cmap="viridis", vmin=v0f, vmax=v1f,
                          interpolation="nearest")
    edges = [[(to_x(r.x0), to_y(r.y0)), (to_x(r.x1), to_y(r.y0)), (to_x(r.x1), to_y(r.y1)), (to_x(r.x0), to_y(r.y1)), (to_x(r.x0), to_y(r.y0))]
             for r in cells.itertuples()]
    ax[0, 1].add_collection(LineCollection(edges, colors="white", linewidths=0.15, alpha=0.5))
    ax[0, 1].set_xlim(lo_w, hi_w); ax[0, 1].set_ylim(lo_c, hi_c)
    ax[0, 1].set_title("Adaptive-mesh fitted: the bilinear element function on its rectangles")
    fig.colorbar(im0, ax=ax[0, 0], shrink=0.9, pad=0.02).set_label("observed log-odds")
    fig.colorbar(im1, ax=ax[0, 1], shrink=0.9, pad=0.02).set_label("fitted log-odds")

    # per-pool held-out miss k - n p_hat against the fitted model's own variance: binomial plus the
    # fitted pool effect (Gaussian on the logit scale around the mean-neutral location)
    held = ~obs["is_train"].astype(bool).values
    s2 = json.load(open(f"{prefix}_final_fit.json"))["pool_variance"]
    A = mean_neutral(lo_(phat), np.full(len(phat), np.sqrt(s2)), 40)[0]
    gx, gw = np.polynomial.hermite.hermgauss(40)
    pk = 1 / (1 + np.exp(-(A[:, None] + np.sqrt(2 * s2) * gx[None, :])))
    m1 = (gw[None, :] / np.sqrt(np.pi) * pk).sum(axis=1); m2 = (gw[None, :] / np.sqrt(np.pi) * pk * pk).sum(axis=1)
    var_i = np.where(held, n * m1 * (1 - m1) + n * (n - 1) * (m2 - m1 * m1), 0.0)
    num, den = hist(np.where(held, k - n * phat, 0.0)), hist(var_i)
    zmap = np.where(den > 0, num / np.sqrt(np.maximum(den, 1e-12)), np.nan); zmap[mass < MINMASS] = np.nan
    im2 = ax[1, 0].imshow(zmap.T, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax[1, 0].set_title("Held-out miss (k - n p) over the model's sd, per bin (z)")
    fig.colorbar(im2, ax=ax[1, 0], shrink=0.9, pad=0.02).set_label("z (fast +, slow -)")

    # Bottom right: the geometry and the coefficients. Every leaf rectangle is drawn (leaves
    # without data lighter). A free coefficient is a filled dot (dark: admitted by the gate, gold:
    # the corners and the depth-1 backbone); a constrained vertex is hollow: a circle where it is a
    # corner of every rectangle around it, a small square where it hangs on a coarser rectangle's
    # edge (its height that edge's average: the nonconformity the rectangular mesh allows).
    verts = pd.read_csv(f"{prefix}_hier_vertices.csv")
    empty = cells.n_pts.values == 0
    boxes = [[(to_x(r.x0), to_y(r.y0)), (to_x(r.x1), to_y(r.y0)), (to_x(r.x1), to_y(r.y1)), (to_x(r.x0), to_y(r.y1))] for r in cells.itertuples()]
    a = ax[1, 1]
    a.add_collection(PolyCollection([b for b, e in zip(boxes, empty) if not e], facecolors="none", edgecolors="#14213d", linewidths=0.35, alpha=0.7))
    a.add_collection(PolyCollection([b for b, e in zip(boxes, empty) if e], facecolors="#f2f2f2", edgecolors="#9a9a9a", linewidths=0.25, alpha=0.8))
    hx, hy = to_x(verts.x.values), to_y(verts.y.values)
    free = verts.free_coefficient.values == 1; adm = verts.gate_admitted.values == 1
    hang = hanging_vertices(cells, verts)
    weld = ~free & ~hang; hung = ~free & hang
    a.scatter(hx[weld], hy[weld], s=6, facecolors="none", edgecolors="#7a7a7a", linewidths=0.4, alpha=0.8, label=f"constrained corner, no coefficient ({int(weld.sum())})")
    a.scatter(hx[hung], hy[hung], s=9, marker="s", facecolors="none", edgecolors="#c0392b", linewidths=0.5, alpha=0.8, label=f"hanging on a coarser edge, no coefficient ({int(hung.sum())})")
    a.scatter(hx[free & ~adm], hy[free & ~adm], s=34, c="#e0a100", edgecolors="white", linewidths=0.3, label=f"corner / backbone coefficient ({int((free & ~adm).sum())})")
    a.scatter(hx[free & adm], hy[free & adm], s=16, c="#14213d", edgecolors="white", linewidths=0.3, label=f"admitted coefficient ({int((free & adm).sum())})")
    a.legend(loc="upper left", bbox_to_anchor=(0.0, -0.09), ncol=2, fontsize=9, frameon=False)
    a.set_xlim(lo_w, hi_w); a.set_ylim(lo_c, hi_c)
    a.set_title(f"Mesh: {len(cells)} leaf rectangles ({int(empty.sum())} without data, grey); vertices by role")
    for a in ax.flat: a.set_xlabel("WALA (months)"); a.set_ylabel("WAC (%)")
    fig.suptitle(title, fontsize=17)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}: {len(cells)} rectangles, {len(verts)} vertices ({int(free.sum())} free, {int(hung.sum())} hanging)")


if __name__ == "__main__":
    main()
