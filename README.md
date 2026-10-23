# Rectangular Decision Mesh

Adaptive rectangular decision mesh with **bilinear interpolation**: the sibling of the
triangular (barycentric) mesh in
[`triangular-decision-mesh`](https://github.com/sbabichenko/triangular-decision-mesh). It fits a
smooth logit surface over (loan age, coupon) to pool prepayment counts: binomial counts with a
mean-neutral pool effect, a hierarchical surplus basis on an adaptively refined mesh, and an
empirical-null FDR gate that decides where to refine.

**Status (2026-09-23):** a working estimator in C++ (`core/`, the reference) and in Python
(`pyrect/`, the port, agreeing with the binary to 1e-9). The C++ engine is out of the tree for
now (2026-10-23; it is in the history up to 190e8db); this tree is the Python port alone. It runs the triangular mesh's current
estimator with a rectangular geometry underneath. On the 2020 year battery it matches the
triangular binary on single-issuer fits (held-out deviance -0.24%, se 0.14%), and it is level when
the newest cohorts are held out and in the combined predictor, fitting in 55-65% of the time and
two thirds of the memory.
See [`docs/RECT_ESTIMATOR.md`](docs/RECT_ESTIMATOR.md).

## Running it

    DMESH_DATA=design.csv DMESH_DUMP=out/run DMESH_SPLIT=1 python3 -m pyrect 7
    python3 -m pytest -q pyrect/tests                   # the checks against the binary skip without one

    python3 -m analysis.starplot <run dir> <design.csv> --vertex <id> --out star.png   # one admission, drawn
    python3 -m analysis.starplot <run dir> <design.csv> --pick giant|keff|surplus --held-out \
        [--compare <another run of the same design, e.g. the triangular engine's>]
    python3 scripts/plot_fit.py <run dir>/run --out fit.png                              # the whole fit
    python3 -m analysis.make_2x2_logodds <design.csv> <run dir>/run <title> 2x2.png     # the standard 2x2

The design file is the triangular engine's (`wala,wac,n0,k_term[,offset][,pool_id]`), as are the
environment (`DMESH_DATA`, `DMESH_DUMP`, `DMESH_SPLIT`, `DMESH_SPLIT_MASK`, `DMESH_BASELINE`,
`DMESH_Q`, `DMESH_LEVEL_GROUPS`, `OMP_NUM_THREADS`) and the dumps the benchmark reads, so the triangular
repo's harness runs the port unchanged:

    cd ../triangular-decision-mesh
    DMESH_BIN=../rectangular-decision-mesh/scripts/pyrect_engine python3 -m bench.base_fits 1 --month 202002

## Contents

- `pyrect/`: the estimator in Python (`pyrect/README.md`).
- `analysis/starplot.py`: star plots, the triangular repo's tool with the rectangular geometry
  underneath (the coefficient's hierarchical column on its cells, the two sections along and
  across its parent segment, the accounting card); it reads the binary's and the port's dumps
  alike. `scripts/plot_fit.py` draws a whole fit.
- `docs/RECT_ESTIMATOR.md`: the design as built, the decisions and the results.
- `docs/RECT_3D.md`: the three-dimensional mesh (loan age, coupon, a third column such as log average
  balance; `DMESH_AXIS3`), its design, results and performance.
- `docs/GEOMETRY_DESIGN.md`, `docs/FDR_PORT.md`, `docs/REGRESSION_DESIGN.md`,
  `docs/ESTIMATOR_LESSONS.md`: the August design documents (the geometry note is what was built,
  with the departures listed in RECT_ESTIMATOR.md).
- `poc/`: the August proof of concept (Python geometry, gate experiments, the Ginnie first look).
- `data/`: the frozen Ginnie and SMM benchmark inputs, copied from the triangular repo.
