# Rectangular Decision Mesh

Adaptive rectangular decision mesh with **bilinear interpolation** — the
sibling of the triangular/barycentric mesh in
[`triangular-decision-mesh`](https://github.com/sbabichenko/triangular-decision-mesh).
The two designs are inspired by each other but intentionally not
code-shared: the rectangular structure revisits core design decisions
rather than swapping the interpolation stencil under the triangular
pipeline.

**Status: design phase.** No solver code yet.

## Contents

- `data/` — the frozen benchmark inputs, copied byte-for-byte from the
  triangular repo so cross-mesh comparisons run on identical data:
  - `ginnie_design.csv` — 120,595-row frozen Ginnie benchmark design
    (pooled rate 0.783241).
  - `smm_design_202606.csv` — June 2026 SMM design.
- `docs/ESTIMATOR_LESSONS.md` — distilled failure modes from the
  triangular mesh's audit sessions (tau-floor welds, zero-information
  hyperparameter voters, the DL estimator audit, honest-se findings,
  footprint certification, holdout design, reproducibility rules).
  These live in the statistical layer, not the geometry; read before
  designing the rectangular estimator.
- `scripts/run_ginnie_benchmark.sh` — the triangular repo's frozen
  benchmark harness, kept as a template for the rectangular equivalent
  (env-flag conventions, dump layout, wall-clock capture). It references
  triangular-mesh env flags and will not run against this repo yet.

## Baselines to beat (triangular mesh, frozen Ginnie CSV)

Seed 7, lindsey, locally balanced split: marginal NLL 2.7129
(2.7413 under historical parity). Seeds 101-120 local-split average
2.7191 (SD 0.0093). See the triangular repo's README and
`docs/LOCAL_BALANCED_SPLIT_BOOTSTRAP_2026-07-30.md` for the exact
environment.
