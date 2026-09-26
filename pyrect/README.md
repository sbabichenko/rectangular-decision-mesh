# pyrect

The rectangular mesh estimator in Python: numpy and scipy, nothing compiled. The same command
line, the same `DMESH_*` environment and the same dump files as `build-release/rect_mesh`, which
is the reference implementation; the two implement one algorithm, and their dumps must agree to
a relative 1e-9 (`pyrect/tests/test_port.py`, `CENTERED_TOLERANCE`). On the pinned February 2020
benchmark they agree to the last printed digit on every table of the single-issuer fit and on
the mesh, vertex and surface tables of the pooled fit, and to 1e-9 on the rest.

    python3 -m pyrect [seed=7]
    DMESH_DATA=design.csv DMESH_DUMP=out/run DMESH_SPLIT=1 python3 -m pyrect 7

    python3 -m pyrect.compare design.csv --binary build-release/rect_mesh --env DMESH_SPLIT=1 DMESH_SPLIT_MASK=mask.txt
        # runs the binary and the port under one environment and compares every dump and log

`scripts/pyrect_engine` wraps the port as an executable, so the triangular repo's benchmark
harness runs it unchanged (`DMESH_BIN=.../scripts/pyrect_engine python3 -m bench.base_fits ...`).

The port started as a copy of the triangular repo's `pymesh` (2026-09-23): everything that does
not touch the geometry is that package's code (the quadrature, the solvers, the gate, the dataset,
the configuration, the programmatic entry point), with two changes in the surface solver that
the binary has too (below).

## Layout

| | |
|---|---|
| `__main__.py`, `run_config.py` | the command line and the environment, validated as the binary validates them |
| `dataset.py` | the design loader (the binary's row rules and messages), the held-out split, the training baseline |
| `mesh.py` | the geometry (`core/mesh/rect.cpp`): dyadic cells, birth parents, heights, the nodal closure and rows, the frontier, candidate columns, admission moves, the EM depth variance |
| `refine.py` | the driver (`core/fit/refine.cpp`): coarse grid, gate rounds (Lindsey null in `gate.py`), admission with live re-profiling, final refit, report and dumps (`report.py`) |
| `estimator/` | `engine.py` (the fit at a fixed topology, the pool variance, the candidate scores), `pool.py` (the centred moments and the surface and variance solvers), `neutral.py` (the mean-neutral location), `design.py` (the nodal live system), `padded_design.py` (bounded-support rows) |
| `api.py` | `fit()`: one fit in a fresh process with a clean environment, dumps returned by path |
| `compare.py` | the check against the binary |
| `stdlib.py`, `units.py`, `errors.py` | the C++ standard-library behaviours the port keeps (`mt19937_64`, the sort), the height units, the exceptions |

## Where the two implementations can differ

The same algorithm in the same order of operations wherever the order is cheap to keep: the
bilinear weights and predictions are the binary's expressions evaluated elementwise, the nodal
rows sum the corners' closures in corner order, the candidate columns are the same influence and
hat expressions. They are not held to bit identity: the vectorised quadrature sums, the BLAS
behind the Cholesky solves and the padded products can round differently. A different admission
would show as a different mesh and fail the comparison outright. The empirical null's iteration
count, a convergence diagnostic in the calibration line, can differ by a step; the comparison
masks it.

The copied surface solver has two changes, made in the binary too: when Newton stalls on the
cached rows (linearised about their base, they switch to exact moments when they drift past
1e-2 logits, and the line search can stall on the switch), it re-bases the cache and continues,
then falls back to exact moments; and its test that the penalty is positive semidefinite is
relative to the penalty's scale, which the nodal prior's mix of large and small precisions
needs.

## Verified

`pyrect/tests/test_port.py` compares the port with the binary on synthetic designs (a given mask,
the engine's own split, a design above the 2048-row chunking, joint levels, the external
baseline), on malformed and edge-case inputs, and checks the port's geometry contracts (a
constrained move leaves the surface unchanged, a candidate's column before its move is its
realised column, the nodal rows reproduce the predictions). Without the binary the comparisons
skip; `DMESH_BINARY=<path>` points them at one.

## Cost

8 to 9 seconds for a benchmark fit (27,000-30,000 pools, about 105 MB) against 0.6 with
the binary; the year battery's 72 fits take about 2 minutes, eight months at a time.
