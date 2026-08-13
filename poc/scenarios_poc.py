"""Rectangular analogues of the triangular /scenarios validation battery
(decision-mesh-audit/scenarios/run_scenarios.sh):

  1. chirp at production noise  -> already run (gate_results.json w2_*).
  2. chirp nearly clean, gate MATCHED to the regime (sigma 0.05 / 0.05).
  3. chirp nearly clean, gate MISMATCHED (assumes production 0.30 on
     0.05 data) — the documented triangular gotcha: "the gate's assumed
     pool variance must match the data regime — at 0.35 on a nearly-
     clean simulation the gate under-refines to the root mesh."
  4. complete null at production noise — the gate should shut at the
     root mesh (triangular scenario 'null').

Outputs: scenario_results.json + clean/mismatch fitted+mesh SVGs.
"""

import json
import os
import numpy as np

from gate_poc import run_gate, chirp, replicate
from run_poc import heatmap_svg, mesh_svg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

g = (np.arange(96) + 0.5) / 96
gx, gy = np.meshgrid(g, g)
TGRID = chirp(gx, gy)
LO, HI = float(TGRID.min()), float(TGRID.max())


def chirp_run(name, assumed, true, fig_prefix):
    rng = np.random.default_rng(7)
    mesh, admitted, _ = run_gate(
        chirp, rng, arm="ebh", n=3000, sigma=assumed, sigma_true=true,
        q=0.1, tau=1.0, max_rounds=12, max_admit=120)
    fgrid = np.array([[mesh.eval(x, y) for x in g] for y in g])
    rmse = float(np.sqrt(np.mean((fgrid - TGRID) ** 2)))
    heatmap_svg(fgrid, f"{fig_prefix}_fitted.svg",
                f"{name}: gated fit ({len(admitted)} admitted)", LO, HI)
    mesh_svg(mesh, f"{fig_prefix}_mesh.svg",
             f"{name}: {len(mesh.leaves)} leaves, {len(admitted)} admissions")
    res = {"sigma_assumed": assumed, "sigma_true": true,
           "n_admitted": len(admitted), "n_leaves": len(mesh.leaves),
           "rmse_vs_truth_grid": rmse}
    print(name, res, flush=True)
    return res


def main():
    results = {}
    # 4. complete null at production noise, both arms
    for arm in ("bh", "ebh"):
        results[f"null_prod_{arm}"] = replicate(
            "w0", arm, reps=40, seed0=9000, n=2000, sigma=0.30, q=0.1,
            tau=0.5)
        print(f"null_prod {arm}: {results[f'null_prod_{arm}']}", flush=True)
    # 3. mismatched gate on nearly-clean data (run before the matched one:
    # it is fast and fails loudly if something is off)
    results["chirp_clean_mismatched"] = chirp_run(
        "Clean chirp, gate assumes production noise", 0.30, 0.05, "mismatch")
    # 2. matched nearly-clean gate
    results["chirp_clean_matched"] = chirp_run(
        "Clean chirp, matched gate", 0.05, 0.05, "clean")
    # Weak-signal pair: amplitude 0.15x. At full amplitude the chirp's
    # signal overwhelms even a 6x-overstated noise assumption, so the
    # triangular under-refinement gotcha only reproduces where the
    # mismatch actually bites: weak signal on clean data.
    weak = lambda x, y: 0.15 * chirp(x, y)
    for name, assumed, prefix in (
            ("weak_chirp_matched", 0.05, "weak_matched"),
            ("weak_chirp_mismatched", 0.30, "weak_mismatch")):
        rng = np.random.default_rng(7)
        mesh, admitted, _ = run_gate(
            weak, rng, arm="ebh", n=3000, sigma=assumed, sigma_true=0.05,
            q=0.1, tau=0.5, max_rounds=12, max_admit=120)
        wt = 0.15 * TGRID
        fgrid = np.array([[mesh.eval(x, y) for x in g] for y in g])
        results[name] = {
            "sigma_assumed": assumed, "sigma_true": 0.05,
            "amplitude_scale": 0.15,
            "n_admitted": len(admitted), "n_leaves": len(mesh.leaves),
            "rmse_vs_truth_grid": float(np.sqrt(np.mean((fgrid - wt) ** 2)))}
        mesh_svg(mesh, f"{prefix}_mesh.svg",
                 f"Weak chirp, {'matched' if assumed == 0.05 else 'MISMATCHED'} "
                 f"gate: {len(mesh.leaves)} leaves, {len(admitted)} admissions")
        print(name, results[name], flush=True)
    with open(os.path.join(OUT, "scenario_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
