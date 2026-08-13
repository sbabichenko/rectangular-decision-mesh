"""Regenerate the W2 (chirp) e-BH gate run and write its 2x2 figures:
gate_fitted.svg, gate_residual.svg, gate_mesh.svg (truth.svg is shared
with the greedy demo). Same seed and parameters as the battery run, so
the figures match gate_results.json's w2_ebh numbers exactly."""

import numpy as np

from gate_poc import run_gate, chirp
from run_poc import heatmap_svg, mesh_svg

rng = np.random.default_rng(7)
mesh, admitted, _ = run_gate(chirp, rng, arm="ebh", n=4000, sigma=0.3,
                             q=0.1, tau=1.0, max_rounds=25)

g = (np.arange(96) + 0.5) / 96
gx, gy = np.meshgrid(g, g)
tgrid = chirp(gx, gy)
fgrid = np.array([[mesh.eval(x, y) for x in g] for y in g])
rmse = float(np.sqrt(np.mean((fgrid - tgrid) ** 2)))
lo, hi = float(tgrid.min()), float(tgrid.max())

heatmap_svg(fgrid, "gate_fitted.svg",
            f"Gated fit, e-BH q=0.1 ({len(admitted)} admitted)", lo, hi)
heatmap_svg(np.abs(fgrid - tgrid), "gate_residual.svg",
            f"|gated fit − truth|  (RMSE {rmse:.3f})")
mesh_svg(mesh, "gate_mesh.svg",
         f"Gated mesh: {len(mesh.leaves)} leaves, {len(admitted)} admissions")
print("done", len(admitted), rmse)
