# Rectangular Decision Mesh

Adaptive rectangular decision mesh with **bilinear interpolation**: the sibling of the
triangular (barycentric) mesh in
[`triangular-decision-mesh`](https://github.com/sbabichenko/triangular-decision-mesh). It fits a
smooth logit surface over (loan age, coupon) to pool prepayment counts: binomial counts with a
mean-neutral pool effect, a hierarchical surplus basis on an adaptively refined mesh, and an
empirical-null FDR gate that decides where to refine.

**Status (2026-09-23):** a working estimator in C++ (`core/`, the reference) and in Python
(`pyrect/`, the port, agreeing with the binary to 1e-9). It runs the triangular mesh's current
estimator with a rectangular geometry underneath. On the 2020 year battery it matches the
triangular binary on single-issuer fits (held-out deviance -0.24%, se 0.14%), and it is level when
the newest cohorts are held out and in the combined predictor, fitting in 55-65% of the time and
two thirds of the memory.
See [`docs/RECT_ESTIMATOR.md`](docs/RECT_ESTIMATOR.md).

## Running it

    cmake -S core -B build-release -DCMAKE_BUILD_TYPE=Release && cmake --build build-release -j
    ctest --test-dir build-release                      # the geometry contracts
    DMESH_DATA=design.csv DMESH_DUMP=out/run DMESH_SPLIT=1 ./build-release/rect_mesh 7

    python3 -m pyrect 7                                 # the port, same environment and dumps
    python3 -m pytest -q pyrect/tests                   # the port against the binary

    python3 -m analysis.starplot <run dir> <design.csv> --vertex <id> --out star.png   # one admission, drawn
    python3 -m analysis.starplot <run dir> <design.csv> --pick giant|keff|surplus --held-out \
        [--compare <another run of the same design, e.g. the triangular engine's>]
    python3 scripts/plot_fit.py <run dir>/run --out fit.png                              # the whole fit
    python3 -m analysis.make_2x2_logodds <design.csv> <run dir>/run <title> 2x2.png     # the standard 2x2

The design file is the triangular engine's (`wala,wac,n0,k_term[,offset][,pool_id]`), as are the
environment (`DMESH_DATA`, `DMESH_DUMP`, `DMESH_SPLIT`, `DMESH_SPLIT_MASK`, `DMESH_BASELINE`,
`DMESH_Q`, `DMESH_LEVEL_GROUPS`, `OMP_NUM_THREADS`) and the dumps the benchmark reads, so the triangular
repo's harness runs either engine unchanged:

    cd ../triangular-decision-mesh
    DMESH_BIN=../rectangular-decision-mesh/build-release/rect_mesh python3 -m bench.base_fits 1 2 3 --month 202002
    DMESH_BIN=../rectangular-decision-mesh/scripts/pyrect_engine  python3 -m bench.base_fits 1 --month 202002
    scripts/run_leveled_benchmark.sh ../rectangular-decision-mesh/build-release/rect_mesh out 1   # the combined predictor

## Contents

- `core/`: the C++ estimator. `mesh/rect.*` is the rectangular geometry; `estimator/` and `fit/`
  are the triangular repo's estimator and gate (copied 2026-09-22) with the mesh-facing parts
  rewritten. `tests/test_rect_geometry.cpp` checks the geometry's contracts.
- `pyrect/`: the Python port (`pyrect/README.md`).
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
- `poc/`: the August proof of concept (Python geometry, gate experiments, the Ginnie first look);
  its C++ core is kept in `poc/core_2026-08-13/`.
- `data/`: the frozen Ginnie and SMM benchmark inputs, copied from the triangular repo.
