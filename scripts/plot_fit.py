"""Draw a rectangular mesh fit from its dumps: the fitted surface, the leaf cells, the pools and
the gate's admissions, in loan age and coupon.

    python3 scripts/plot_fit.py <dump prefix, e.g. out/run> [--out fit.png] [--limit 1.5]

Reads <prefix>_model.json (the coordinate bounds), _surface.csv, _mesh.csv, _obs.csv and
_hier_vertices.csv, as the binary and the port write them.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection


def draw(ax, prefix, limit=1.5):
    prefix = str(prefix)
    model = json.loads(Path(prefix + "_model.json").read_text())
    x0, x1, y0, y1 = model["wala_min"], model["wala_max"], model["wac_min"], model["wac_max"]
    sx = lambda u: x0 + np.asarray(u) * (x1 - x0)
    sy = lambda u: y0 + np.asarray(u) * (y1 - y0)
    surface = pd.read_csv(prefix + "_surface.csv")
    n = int(round(np.sqrt(len(surface))))
    image = ax.imshow(surface.g_hat.values.reshape(n, n).T, origin="lower", extent=(x0, x1, y0, y1), aspect="auto",
                      cmap="RdBu_r", vmin=-limit, vmax=limit)
    cells = pd.read_csv(prefix + "_mesh.csv")
    polys = [[(sx(c.x0), sy(c.y0)), (sx(c.x1), sy(c.y0)), (sx(c.x1), sy(c.y1)), (sx(c.x0), sy(c.y1))]
             for c in cells.itertuples()]
    ax.add_collection(PolyCollection(polys, facecolors="none", edgecolors="k", linewidths=0.35))
    obs = pd.read_csv(prefix + "_obs.csv")
    ax.scatter(sx(obs.x), sy(obs.y), s=0.3, c="0.35", alpha=0.25, linewidths=0)
    vertices = pd.read_csv(prefix + "_hier_vertices.csv")
    admitted = vertices[vertices.gate_admitted == 1]
    ax.scatter(sx(admitted.x), sy(admitted.y), s=9, facecolors="gold", edgecolors="k", linewidths=0.4, zorder=5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel("loan age (months)"); ax.set_ylabel("coupon (%)")
    ax.set_title(f"{len(cells)} cells, {int(vertices.free_coefficient.sum())} free coefficients, "
                 f"{len(admitted)} gate admissions", fontsize=10)
    return image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix")
    parser.add_argument("--out", type=Path, default=Path("fit.png"))
    parser.add_argument("--limit", type=float, default=1.5, help="colour scale limit, logits")
    args = parser.parse_args(argv)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = draw(ax, args.prefix, args.limit)
    fig.colorbar(image, ax=ax, label="fitted correction (logit)")
    fig.savefig(args.out, dpi=150)
    print(args.out)


if __name__ == "__main__":
    main()
