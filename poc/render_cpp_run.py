"""Render page assets for the C++ full-data runs: obs/fitted/resid/mesh
SVGs per dataset from the ginnie_rect dumps + raw CSVs."""
import json
import os
import numpy as np
from run_poc import heatmap_svg, _color
from ginnie_poc import binned, WALA_LO, WALA_HI, WAC_LO, WAC_HI

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
STATE_COLORS = {0: "#eab308", 1: "#22c55e", 2: "#ef4444"}  # free/weld/hanging

def mesh_svg_cpp(tag, title):
    leaves = np.genfromtxt(f"{OUT}/cpp_{tag}/mesh.csv", delimiter=",", names=True)
    verts = np.genfromtxt(f"{OUT}/cpp_{tag}/verts.csv", delimiter=",", names=True)
    w = 640
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {w+58}" font-family="sans-serif">',
             f'<text x="4" y="18" font-size="16">{title}</text>',
             '<text x="4" y="38" font-size="13"><tspan fill="#b48a00">&#9679; free</tspan>  '
             '<tspan fill="#15803d">&#9679; weld</tspan>  <tspan fill="#dc2626">&#9679; hanging</tspan></text>',
             '<g transform="translate(0,50)">', f'<rect width="{w}" height="{w}" fill="#f8fafc"/>']
    for row in leaves:
        parts.append(f'<rect x="{row["x0"]*w:.1f}" y="{(1-row["y1"])*w:.1f}" '
                     f'width="{(row["x1"]-row["x0"])*w:.1f}" height="{(row["y1"]-row["y0"])*w:.1f}" '
                     f'fill="none" stroke="#64748b" stroke-width="0.8"/>')
    for row in verts:
        parts.append(f'<circle cx="{row["x"]*w:.1f}" cy="{(1-row["y"])*w:.1f}" r="2.6" '
                     f'fill="{STATE_COLORS[int(row["state"])]}"/>')
    parts.append("</g></svg>")
    open(f"{OUT}/cpp_{tag}_mesh.svg", "w").write("".join(parts))

def render(tag, csv, title):
    d = np.genfromtxt(os.path.join(DATA, csv), delimiter=",", names=True)
    x = (d["wala"] - WALA_LO) / (WALA_HI - WALA_LO)
    y = (d["wac"] - WAC_LO) / (WAC_HI - WAC_LO)
    n, k = d["n0"], d["k_term"]
    pt = (k + .5) / (n + 1)
    z = np.log((k + .5) / (n - k + .5))
    w = n * pt * (1 - pt)
    obs = binned(x, y, z, w)
    g = np.genfromtxt(f"{OUT}/cpp_{tag}/surface_grid.csv", delimiter=",", names=True)
    fgrid = g["f"].reshape(96, 96)
    presence = ~np.isnan(obs)
    mask96 = np.repeat(np.repeat(presence, 2, 0), 2, 1)
    fgrid = np.where(mask96, fgrid, np.nan)
    lo, hi = float(np.nanmin(obs)), float(np.nanmax(obs))
    heatmap_svg(obs, f"cpp_{tag}_obs.svg", f"{title}: observed logit (full data)", lo, hi)
    res = json.load(open(f"{OUT}/cpp_{tag}/results.json"))
    heatmap_svg(fgrid, f"cpp_{tag}_fitted.svg",
                f"{title}: C++ full-data fit ({res['n_admitted']} admitted)", lo, hi)
    fx = np.array([np.interp(0,[0],[0])])  # noop
    # binned residual vs fitted surface
    fi = fgrid[np.clip((y*96).astype(int),0,95)*0 + np.clip((y*96).astype(int),0,95),
               np.clip((x*96).astype(int),0,95)]
    resid = binned(x, y, z - np.nan_to_num(fi), w)
    heatmap_svg(resid, f"cpp_{tag}_resid.svg", f"{title}: binned residual logit")
    mesh_svg_cpp(tag, f"{title}: {res['n_leaves']} leaves, {res['n_admitted']} admissions")

render("cumulative", "ginnie_design.csv", "Ginnie cumulative attrition")
render("smm", "smm_design_202606.csv", "Ginnie monthly SMM")
print("rendered")
