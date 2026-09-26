# The three-dimensional rectangular mesh

Status 2026-09-23, uncommitted. The mesh over loan age, coupon and log average balance, in the C++
core and the Python port (agreeing to 1e-9), with 3-D star pages and a 3-D grid figure. Two
dimensions are unchanged: every 2-D dump is byte-identical to before, and so are the 2-D pages.

## Running it

    DMESH_AXIS3=log_balance DMESH_DATA=design.csv ... ./build-release/rect_mesh 7    # or python3 -m pyrect 7

`DMESH_AXIS3` names a design column; it is scaled to [0, 1] by its own range, as loan age and
coupon are. The model file gains `axis3`, `axis3_min`, `axis3_max`; the cell table has `z0`, `z1`
and eight corners (`h000` ... `h111`, bit order x, y, z); the vertex, observation and surface tables
gain a `z` column (the candidate table `x3`, since its `z` is the gate score). The start is an
8 x 8 x 4 grid, the balance axis coarser; the corners and the depth-1 vertices (27) start free.

    python3 -m analysis.starplot <run dir> <design.csv> --vertex <id>     # a 3-D run draws the 3-D page
    python3 -m analysis.make_2x2_logodds <design.csv> <dump prefix> <title> <out.png>   # 3 x 4 by balance tercile

## What three dimensions change

Splitting a box along one of three axes still makes new vertices only at edge midpoints, with two
birth parents, so the birth-parent recursion, the hierarchical columns and inert splits carry
over. Two things do not:

1. **Face-hanging vertices.** A vertex can land inside a coarse leaf's face, where two birth axes
   are possible and the parents' average need not match the face's trace. The closure keeps every
   hanging vertex on a leaf edge along its own birth axis, splitting (inertly) any leaf that would
   hold one otherwise; then the 2-D continuity argument applies. The geometry test checks the
   invariant after every move and split, and continuity across every leaf face.
2. **Columns read before a move can be wrong.** The closure can split through an existing vertex
   along another axis than its birth's; it keeps its parents (relaxed rebirth), which is inert for
   every existing coefficient but gives a candidate a column the pre-move reading does not foresee
   (wrong in about half of random test moves). So the gate scores each 3-D candidate on its exact
   column: the move is made on a per-thread copy of the mesh, the column read and the move undone
   (an undo journal). Scoring and admission therefore see the same column.

## Results (rectangular engine, 2020, Ginnie and Fannie with SCR excluded)

3-D mesh + crossed issue-type x balance levels + seller levels, against the 2-D surface with the
same levels (validation/threed_lvs*, cfold_table.csv, oot4.csv under docs/figures/months/):

| test | 3-D better | median |
|---|---|---|
| cohort folds (issue months dealt into 5 folds) | 69/80 | -2.2% |
| random split, seeds 1-3 | 42/48 | -2.0% |
| next month | 13/16 | -1.3% |

The exception is Ginnie pooled in some months, where single- and multi-issuer pools share one
surface. The old cohort split (newest 20% of issue months) held out every young loan, so it
measured age extrapolation, not new cohorts; it is replaced by the cohort folds.

Watch the data's edge: the largest 3-D correction on Fannie April single (vertex 699, +4.0 logit)
sits at the cube's youngest, highest-balance corner, where no pool lives; its held-out pools run
1.37% observed against 2.18% fitted.

![3-D star page](figures/star3d_fx_202004_vertex699.png)
![3-D grid](figures/grid3d_fx_202004_single.png)

## Performance

Fannie pooled October (53k pools), 8 threads: binary 4.1 s -> 3.4 s (dry runs in the parallel
scoring loop on per-thread mesh copies; rows moved, not copied, in splits), 185 MB; the 2-D fit of
the same data takes 1.5 s. The Python port 62 s -> 46 s (dry runs place no rows, and the column
weighs a dry run's new cells in one vectorised pass; flat descent tables; precomputed orthants);
the 2-D port takes 23 s. What remains in the port is the shared pool-moment code. Tried and
dropped: a geometry-only pre-check to skip dry runs (slower, and wrong: no rebirth does not make
the pre-move column exact), lazy row partitions in dry runs (no gain).

### Second pass (after the memory pass)

- Binary: the scoring system's factor is transposed to column order once per round, so the
  per-candidate Cholesky solves skip LAPACKE's row-major transpose (identical output): 3-D 3.2 ->
  3.1 s. Batching the candidates' solves would read the factor once per batch, but changes rounding;
  not done.
- Port, 3-D: each candidate's dry-run column is kept with what it depended on (the pre-existing
  cells it looked at, which must stay unsplit leaves, and the grid points it found empty, which must
  stay empty; vertices are never removed). Most of the frontier sits in unchanged geometry, so most
  dry runs are skipped: 46 -> 38 s, at +37 MB peak (194 -> 231 MB). Entries for candidates that left
  the frontier are dropped every round. `PYRECT_CHECK_COLUMN_CACHE=1` recomputes every kept column
  and requires equality; the 3-D port test runs with it on, and a full benchmark run found none
  differing.
- Where the quadrature's rows go (3-D, Fannie pooled October): partial row-cache passes ~45%, the
  final fit's variance search ~34% (every trial variance refit the surface with exact full passes and
  no row cache), gate rounds after a variance step ~17%. Duplicate rows are 0-1% (pool coordinates
  are loan-weighted averages); warm-starting the mode solves could save at most ~6% (they take 3.0-3.8
  Newton steps against a floor of ~2). The variance search's trial fits now use a row cache and only
  the surface at the root is certified on exact moments: binary 3-D 3.1 -> 3.0 s, 2-D 1.51 -> 1.45 s;
  port 3-D 38.3 -> 35.7 s, 2-D 22.5 -> 21.1 s. Only the final fit moves: fitted rates by at most
  1e-7 relative, held-out deviance by at most 5e-9 relative on eight battery fits, same meshes; the
  port agrees at 1e-9. A variance-aware row cache (reusing rows across a variance step) would not pay:
  the variance search needs every row's moments at each trial variance for its dispersion score, so
  one full pass per trial variance is the floor.
- The shared costs (pool quadrature in the binary; the batched 6x6 eigen-solves and the enumerated
  moments in the port) are at what the current numerics allow.

## Memory

Peak resident size on the same fit (8 threads), every output unchanged:

| | before | after |
|---|---|---|
| binary 3-D | 176 MB | 131 MB |
| binary 2-D | 84 MB | 86 MB (unchanged within noise) |
| port 3-D | 218 MB | 194 MB |
| port 2-D | 156 MB | 146 MB |

What changed: the scoring system reads a candidate's rows of the design in place instead of copying
them (`PaddedDesign::rmatvec_rows`); the scoring drops two gathered copies; the dry-run mesh copies
share the original's per-row data, which a dry run no longer writes (its new cells' rows are weighed
where the column reads them); the level columns are added in one copy of the design. In the port:
a 32-bit design index (as the binary's), the level columns in one stack, and the candidate records
as one float block per round. Tried and dropped: a fixed malloc mmap threshold (10-16 MB less, 60-80%
slower) and per-thread reusable scoring buffers (25 MB more: they keep every thread at its largest
candidate). `DMESH_MEM_PROFILE=1` with `DMESH_CENTERED_PROFILE=1` credits the peak's growth to the
engine's phase timers. About 50 MB of the port is the interpreter and its libraries.

## Open

- The empirical-null (Lindsey) iteration can converge so slowly (about 100 steps) that scores
  equal to ten digits land on either side of its 100-step cap in the binary and the port. A
  shared-gate fragility (2-D too); fixing it changes the pinned triangular reference.
- The Python port's pool-moment code is the remaining port cost, shared with 2-D.
