"""Run the rectangular-mesh proof of concept and write results + SVG figures.

Outputs to poc/output/: results.json, truth.svg, fitted.svg, residual.svg,
mesh.svg. Everything is deterministic (fixed seed).

Three demonstrations:
  1. Inert-split lemma, numerically (GEOMETRY_DESIGN §2b).
  2. Composite release split = exactly one new column (§2, §7.3).
  3. Adaptive anisotropic fit on synthetic data: both axis cuts offered
     everywhere, greedy local scoring (PoC stand-in for the gate), welds,
     hanging nodes, and release splits all exercised.
"""

import json
import os
import numpy as np

from rect_mesh import RectMesh, S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ------------------------------------------------------------------ SVG

_VIRIDIS = np.array([
    (0.267, 0.005, 0.329), (0.283, 0.141, 0.458), (0.254, 0.265, 0.530),
    (0.207, 0.372, 0.553), (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
    (0.135, 0.659, 0.518), (0.267, 0.749, 0.441), (0.478, 0.821, 0.318),
    (0.741, 0.873, 0.150), (0.993, 0.906, 0.144)])


def _color(v):
    t = np.clip(v, 0, 1) * (len(_VIRIDIS) - 1)
    i = int(t)
    j = min(i + 1, len(_VIRIDIS) - 1)
    c = _VIRIDIS[i] + (t - i) * (_VIRIDIS[j] - _VIRIDIS[i])
    return "#%02x%02x%02x" % tuple(int(255 * x) for x in c)


def heatmap_svg(grid, fname, title, vmin=None, vmax=None):
    n = grid.shape[0]
    vmin = grid.min() if vmin is None else vmin
    vmax = grid.max() if vmax is None else vmax
    span = (vmax - vmin) or 1.0
    px, w = 6, 6 * n
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {w+30}" '
             f'font-family="sans-serif">',
             f'<text x="4" y="18" font-size="16">{title}  '
             f'[{vmin:.2f}, {vmax:.2f}]</text>',
             f'<g transform="translate(0,26)">']
    for iy in range(n):
        for ix in range(n):
            c = _color((grid[iy, ix] - vmin) / span)
            parts.append(f'<rect x="{ix*px}" y="{(n-1-iy)*px}" width="{px}" '
                         f'height="{px}" fill="{c}"/>')
    parts.append("</g></svg>")
    with open(os.path.join(OUT, fname), "w") as f:
        f.write("".join(parts))


STATE_COLORS = {"free": "#eab308", "weld": "#22c55e", "hanging": "#ef4444"}


def mesh_svg(mesh, fname, title):
    w = 640
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {w+58}" '
             f'font-family="sans-serif">',
             f'<text x="4" y="18" font-size="16">{title}</text>',
             '<text x="4" y="38" font-size="13">'
             '<tspan fill="#b48a00">&#9679; free</tspan>  '
             '<tspan fill="#15803d">&#9679; weld (species 1)</tspan>  '
             '<tspan fill="#dc2626">&#9679; hanging (species 2)</tspan></text>',
             f'<g transform="translate(0,50)">',
             f'<rect x="0" y="0" width="{w}" height="{w}" fill="#f8fafc"/>']
    for c in mesh.leaves:
        x, y = c.x0 / S * w, (1 - c.y1 / S) * w
        cw, ch = (c.x1 - c.x0) / S * w, (c.y1 - c.y0) / S * w
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" '
                     f'height="{ch:.1f}" fill="none" stroke="#64748b" '
                     f'stroke-width="1"/>')
    for v in mesh.verts.values():
        parts.append(f'<circle cx="{v.x*w:.1f}" cy="{(1-v.y)*w:.1f}" r="3.4" '
                     f'fill="{STATE_COLORS[v.state]}"/>')
    parts.append("</g></svg>")
    with open(os.path.join(OUT, fname), "w") as f:
        f.write("".join(parts))


# ------------------------------------------- demo 1: inert-split lemma

def demo_lemma():
    m = RectMesh()
    # random topology with promotions of regular chord vertices
    for _ in range(10):
        leaf = m.leaves[rng.integers(len(m.leaves))]
        a, b = m.split(leaf, rng.choice(["x", "y"]))
        for k, v in list(m.verts.items()):
            if v.state == "weld" and rng.random() < 0.6:
                m.promote(k)
    for k in m.free_keys():
        m.verts[k].height = rng.normal()
    g0 = m.eval_grid(96)
    # welded-only splits must be invisible
    for _ in range(6):
        leaf = m.leaves[rng.integers(len(m.leaves))]
        m.split(leaf, rng.choice(["x", "y"]))  # chords stay welded
    g1 = m.eval_grid(96)
    return {"grid_n": 96, "n_setup_splits": 10, "n_welded_splits": 6,
            "max_abs_surface_change": float(np.abs(g1 - g0).max())}


# ------------------------------ demo 2: release split = exactly one column

def demo_release():
    m = RectMesh()
    m.split(m.root, "x")
    left = next(l for l in m.leaves if l.x0 == 0)
    m.split(left, "x")
    lr = next(l for l in m.leaves if l.x0 == S // 4)
    m.split(lr, "y")
    key = (S // 4, S // 2)  # strictly inside the [0,1/4] leaf's right edge
    assert m.verts[key].state == "hanging"
    for k in m.free_keys():
        m.verts[k].height = rng.normal()
    # probe points; designs in both parameterizations, pre-release
    px = rng.random(600)
    py = rng.random(600)
    keys0, X0 = m.design(px, py)          # nodal basis
    keys0h, X0h = m.design_hier(px, py)   # hierarchical (surplus) basis
    g0 = m.eval_grid(96)
    n_rel = m.promote(key)  # composite move: release split + free
    g1 = m.eval_grid(96)  # released vertex still AT its weld value: inert
    keys1, X1 = m.design(px, py)
    keys1h, X1h = m.design_hier(px, py)
    coln = {k: X1[:, i] for i, k in enumerate(keys1)}
    colh = {k: X1h[:, i] for i, k in enumerate(keys1h)}
    max_nodal_change = max(
        float(np.abs(coln[k] - X0[:, i]).max()) for i, k in enumerate(keys0))
    max_hier_change = max(
        float(np.abs(colh[k] - X0h[:, i]).max()) for i, k in enumerate(keys0h))
    # give the released vertex a surplus: exactly its column moves
    m.verts[key].height += 0.5
    g2 = m.eval_grid(96)
    support_frac = float(np.mean(np.abs(g2 - g1) > 1e-12))
    return {"n_release_splits": n_rel,
            "surface_change_from_release_geometry": float(np.abs(g1 - g0).max()),
            "preexisting_column_change_hierarchical_basis": max_hier_change,
            "preexisting_column_change_nodal_basis": max_nodal_change,
            "new_column_support_fraction_of_domain": support_frac}


# ----------------------------------------- demo 3: adaptive anisotropic fit

def truth(x, y):
    return (0.5 * np.tanh(14 * (x - 0.30))
            + 0.4 * np.tanh(18 * (y - 0.70)) / (1 + np.exp(-12 * (x - 0.55)))
            + 0.6 * np.exp(-((x - 0.80) ** 2 + (y - 0.25) ** 2) / 0.008))


def _local_gain(m, leaf, axis, xs, ys, r):
    inside = ((xs * S > leaf.x0) & (xs * S <= leaf.x1)
              & (ys * S > leaf.y0) & (ys * S <= leaf.y1))
    if inside.sum() < 12:
        return -1.0
    u = (xs[inside] * S - leaf.x0) / (leaf.x1 - leaf.x0)
    w = (ys[inside] * S - leaf.y0) / (leaf.y1 - leaf.y0)
    tent = 1 - np.abs(2 * (u if axis == "x" else w) - 1)
    other = w if axis == "x" else u
    Xc = np.column_stack([tent * (1 - other), tent * other])
    A = Xc.T @ Xc + 1e-10 * np.eye(2)
    b = Xc.T @ r[inside]
    return float(b @ np.linalg.solve(A, b))


def demo_fit(n_splits=40):
    n_train, n_test = 4000, 1500
    xs = rng.random(n_train + n_test)
    ys = rng.random(n_train + n_test)
    zs = truth(xs, ys) + 0.08 * rng.normal(size=xs.shape)
    tr = slice(0, n_train)
    te = slice(n_train, None)

    m = RectMesh()
    m.fit(xs[tr], ys[tr], zs[tr])
    n_release_total = 0
    n_selected = 0
    for _ in range(n_splits):
        pred = np.array([m.eval(x, y) for x, y in zip(xs[tr], ys[tr])])
        r = zs[tr] - pred
        best = None
        for leaf in m.leaves:
            for axis in ("x", "y"):  # both cuts always offered (§7.1)
                g = _local_gain(m, leaf, axis, xs[tr], ys[tr], r)
                if best is None or g > best[0]:
                    best = (g, leaf, axis)
        gain, leaf, axis = best
        if gain <= 0:
            break
        m.split(leaf, axis, origin="selected")
        n_selected += 1
        # PoC stand-in for the gate: admit both chord coefficients of the
        # selected cut, composite release splits included.
        d = leaf.lx + leaf.ly  # children level sum minus 1 == parent's
        for k, v in list(m.verts.items()):
            if v.depth == d + 1 and v.state in ("weld", "hanging") \
                    and v.birth_origin == "selected":
                n_release_total += m.promote(k)
        m.fit(xs[tr], ys[tr], zs[tr])

    pred_te = np.array([m.eval(x, y) for x, y in zip(xs[te], ys[te])])
    pred_tr = np.array([m.eval(x, y) for x, y in zip(xs[tr], ys[tr])])
    states = {}
    origins = {}
    for v in m.verts.values():
        states[v.state] = states.get(v.state, 0) + 1
        origins[v.birth_origin] = origins.get(v.birth_origin, 0) + 1
    axes = {"x": 0, "y": 0}
    stack = [m.root]
    while stack:
        c = stack.pop()
        if not c.leaf:
            axes[c.axis] += 1
            stack += list(c.children)
    g = (np.arange(96) + 0.5) / 96
    gx, gy = np.meshgrid(g, g)
    return m, {
        "n_train": n_train, "n_test": n_test, "noise_sd": 0.08,
        "n_selected_splits": n_selected,
        "n_splits_total": axes["x"] + axes["y"],
        "n_balance_splits": axes["x"] + axes["y"] - n_selected - n_release_total,
        "axis_counts": axes,
        "n_release_splits": n_release_total,
        "vertex_states": states, "birth_origins": origins,
        "n_free_coefficients": len(m.free_keys()),
        "train_rmse": float(np.sqrt(np.mean((zs[tr] - pred_tr) ** 2))),
        "test_rmse": float(np.sqrt(np.mean((zs[te] - pred_te) ** 2))),
        "test_rmse_vs_noiseless_truth": float(np.sqrt(np.mean(
            (truth(xs[te], ys[te]) - pred_te) ** 2))),
    }, truth(gx, gy)


def main():
    results = {"lemma": demo_lemma(), "release": demo_release()}
    m, fit_stats, tgrid = demo_fit()
    results["fit"] = fit_stats
    fgrid = m.eval_grid(96)
    lo, hi = float(tgrid.min()), float(tgrid.max())
    heatmap_svg(tgrid, "truth.svg", "Truth surface", lo, hi)
    heatmap_svg(fgrid, "fitted.svg", "Fitted bilinear surface "
                f"({results['fit']['n_free_coefficients']} free coefficients)",
                lo, hi)
    heatmap_svg(np.abs(fgrid - tgrid), "residual.svg", "|fit − truth|")
    mesh_svg(m, "mesh.svg",
             f"Adaptive mesh: {len(m.leaves)} leaves, "
             f"{results['fit']['axis_counts']['x']} x-cuts / "
             f"{results['fit']['axis_counts']['y']} y-cuts")
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
