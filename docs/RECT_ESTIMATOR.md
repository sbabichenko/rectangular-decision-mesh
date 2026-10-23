# The rectangular estimator (2026-09-23)

The rectangular mesh now runs the triangular mesh's current estimator end to end: the
centred-score surface fit with mean-neutral pool effects, the pool-variance steps, the
Lindsey empirical null with linear binning, the lfdr running-mean cutoff at FDR 0.40 (0.10 before 2026-09-29), and the
EM depth variance. `core/` is that estimator with a rectangular geometry underneath; the
August proof of concept is in `poc/core_2026-08-13/`.

    cmake -S core -B build-release -DCMAKE_BUILD_TYPE=Release && cmake --build build-release -j
    DMESH_DATA=design.csv DMESH_DUMP=out/run ./build-release/rect_mesh [seed]

The command line, the environment (`DMESH_DATA`, `DMESH_DUMP`, `DMESH_SPLIT`,
`DMESH_SPLIT_MASK`, `DMESH_BASELINE`, `DMESH_Q`, `DMESH_LEVEL_GROUPS`, `OMP_NUM_THREADS`) and the dumps
the benchmark reads (`run_obs.csv`, `run_model.json`) are the triangular binary's, so the
triangular repo's harness runs it unchanged: `DMESH_BIN=.../rect_mesh python3 -m
bench.base_fits 1 2 3 --month M`.

## What was copied and what is new

Copied unchanged from `triangular-decision-mesh/core` (2026-09-22): the quadrature, the
score system, the surface solver, the pool-variance fit, the padded design, the gate, the
dataset and the configuration. Rewritten for rectangles: `mesh/rect.*` (the geometry),
`estimator/system.*` (the live system), the mesh-facing parts of `estimator/engine.*`,
`fit/refine.cpp` (the frontier and the admissions) and `fit/dumps.cpp` (the vertex and cell
tables). Two changes to the copied solver. First, when Newton stalls on the cached rows it
re-bases the cache and continues, then falls back to exact moments. The rows within 1e-2 logits of their
cached base are linearised and switch to exact moments when they drift past it, and the line
search stalled on that switch in one of the first battery's 72 fits (November, seed 3, pooled:
|score| 8.5e-6 with no descent step). A converging fit never reaches the fallback. Second, the
test that the penalty is positive semidefinite is relative to its scale (-1e-12 times p times its
largest entry, where it was -1e-12 absolute): the nodal prior (below) mixes precisions up to
1/tau_floor^2 = 4e4 with small ones, and an eigenvalue at or near zero rounds to about 1e-11
either side (four fits of one experiment failed the absolute test while the roots had no prior).
Both changes are in the port's solver too; the triangular engine has neither (its diagonal
penalty cannot trip the second).

## The geometry

Anisotropic dyadic bisection on a 2^24 integer grid (`docs/GEOMETRY_DESIGN.md` §1(b)): a cut
halves a leaf at its x or y midline and creates the chord's two endpoints. Every vertex but
the four roots has two birth parents, the endpoints of the segment it bisects, and those are
canonical: a point on a vertical cell boundary at y = (odd) 2^k can only be born bisecting
[y - 2^k, y + 2^k].

**Heights by the birth-parent recursion, no stored constraints.** A vertex's height is its
parents' average plus its surplus, and the surplus is nonzero only for a free coefficient.
A hanging vertex's parents lie on the coarse edge it sits on, so its height is that edge's
linear trace and the surface is continuous without the constraint closures §3 planned. A free
coefficient must be regular (a corner of every leaf touching it); admission makes it so and
later splits cannot undo it.

**No 2:1 balance.** Balance was there to bound constraint chains; with heights resolved by
birth order there are no chains to bound, and balance splits are inert anyway (§2b). Left out.

**The hierarchical basis.** A free coefficient's column is its influence carried to every
descendant at one half per generation, read bilinearly on the leaves: the surplus basis. The
column is fixed at birth, which is why there is no dormancy, no two-node hierarchy and no
retirement pass: a coefficient's support cannot be taken away by later refinement.

**Fitted in nodal coordinates.** The surplus basis is wide: a point lies under a few hats per
level, so the hierarchical rows ran to 28 entries on the benchmark, against at most 7 for the
nodal design B (a free vertex's column stopping at its free descendants). The live system
therefore fits the free vertices' heights with B and puts the depth prior on the surpluses
d = R h as the dense penalty R' diag(1/tau^2) R (p is 60-150, so a dense p x p costs nothing).
It is the same model: the admissions were identical on all 72 battery fits and the held-out
deviances agree to every printed digit, while the Gram, refit and scoring products, which scale
with the width squared, fell from about 1.2 s to 0.1 s of a single-threaded pooled fit. The
engine converts back for the EM depth variance: a child's surplus is its row of R applied to
the heights, its posterior variance r' (B'JB + P)^-1 r.

## The candidates

A round's frontier: every constrained vertex (a welded chord endpoint or a hanging vertex),
and the midpoint of every leaf edge that has no vertex yet. A midpoint on a vertical edge is
a y-cut of the cells on both sides and one on a horizontal edge an x-cut, so both axes of every
cell are offered and the gate owns the axis choice (§7.1).

A candidate's column is the column its coefficient will have after its move, computed without
moving: the influence through its existing descendants on the leaves where it is a corner,
plus, on each leaf that has it inside an edge, the plain bilinear hat on the quadrant cell the
move will split off (the largest dyadic sub-cell with the point as a corner). Admission
realises the move (splits the leaves around the point until it is a corner of each: a new
vertex's birth or a hanging vertex's release, §2), which leaves the surface unchanged, then
re-profiles the candidate on its realised column, as the triangular admission does.

`core/tests/test_rect_geometry.cpp` checks these contracts on random meshes: a constrained move
and a random split leave every prediction unchanged to 1e-12; a candidate's column before its
move equals the realised column to 1e-12; the design rows reproduce the predictions; the surface
is continuous across every leaf edge.

## Settings

- Prior class (the "depth" the EM depth variance is indexed by): the level of the bisected
  segment, 1 for a unit segment. A surplus corrects the interpolation along that segment, so
  its expected size scales with the segment's length whatever the cell's other side.
- Coarse mesh: an 8 x 8 grid; the roots and the depth-1 vertices (the midlines) start free, as
  the triangular engine frees its level and tilt. The rest of the grid waits on the frontier.
- **The four root heights have a prior: sd 1 logit around the baseline.** The triangular roots are
  unpenalised. Here that left the fit extrapolating poorly when the newest cohorts (the youngest
  ages) are held out; see the cohort split below.
- **The empirical null's scale is capped at 3**, not the triangular gate's 6 (`fit/gate.cpp`).
- **A round that admits nothing is a stall** whether nothing was selected or every selected
  candidate failed its re-profile. The triangular driver counts only the first, and a candidate
  selected and rejected in every round kept it going to the 80-round cap (the triangular engine
  on the SMM design, seed 1: 70 idle rounds).
- Cell budget max(600, N/24), the triangular face budget; it has not bound on the benchmark
  (the battery's 72 fits end at 104-226 cells, when the gate stops admitting).
- `mallopt(M_ARENA_MAX, 2)` in `main`: the parallel candidate scoring makes many short-lived
  allocations per thread, and glibc's default arena per thread kept about 7 MB of fragments per
  thread (pooled benchmark at 8 threads: 123 MB with the default, 79 MB with two arenas, same time).

### How they were chosen

Four tests, each run for every alternative: the year battery (random splits, 72 fits, the
criterion); the **cohort split** (the newest 20% of issue months held out, every month of 2020:
extrapolation to new cohorts, which the random split cannot see); the Ginnie design (120,595 rows,
seeds 1-10, the engine's own split); and the SMM design (33,740 rows, seeds 1-5). Against the first
configuration (the depth-1 backbone, flat roots, the null capped at 6):

| alternative | battery, single | battery, pooled | other tests |
|---|---|---|---|
| **root prior, sd 1 logit (adopted)** | **-0.03% (se 0.05%)** | +0.8% | **cohort split -7.1% (se 3.0%), 9 of 12 months better; Ginnie +0.04% (se 0.04%)** |
| root prior, sd 0.5 / sd 2 | | | cohort split -6.9% / -4.0% |
| **null scale capped at 3 (adopted)** | **identical** | **identical** | **cohort split and Ginnie identical; SMM seed 1 1.685 to 1.269** |
| roots, depth 1 and depth 2 free | -0.07% (se 0.09%) | +0.36% | Ginnie -0.57% (se 0.17%), SMM level with the triangular engine, but the cohort split +139% against the triangular engine: rejected |
| roots, depth 1-3 free | | | Ginnie -0.28% (se 0.14%) |
| prior class by segment level and axis (tau per axis) | -0.02% (se 0.04%) | +0.41% | |
| prior class by generation (max parent depth + 1) | -0.05% (se 0.04%) | +0.06% | |
| coarse 4 x 4 grid | +0.08% (se 0.09%) | +11.6% (one +279% case) | |
| coarse 16 x 16 grid | +0.29% (se 0.15%) | +0.37% | Ginnie +0.53% (se 0.14%) |
| only the roots free at the start | +1.87% (se 0.63%) | +35% | |
| candidates need 2 effective pools | +0.94% (se 0.47%) | +0.85% | |
| candidates need 4 effective pools | +0.73% (se 0.33%) | +10% | |
| fine nodes (the two-node counterpart; see Open) | -0.07% (se 0.04%) | +0.78% | |
| depth variance log-linear in depth, one EM over all admissions | -0.04% (se 0.06%) | +0.10% | |
| depth variance by likelihood conditional on selection (see Open) | -0.01% (se 0.05%), 4 fits failed | +1.4% | |
| depth variance from every scored candidate, admitted or not (see Open) | +0.78% (se 0.14%) | +7.8% | |
| slab variance of a null/slab mixture fitted to every scored candidate (see Open) | +0.37% (se 0.11%) | +4.5% | |

The random-split battery could not tell most of these apart; the other tests decided three. The
**depth-2 backbone** looked like a win on Ginnie and SMM and was adopted for an hour, until the
cohort split showed free quarter-scale heights at the edge of the data extrapolating steeply into
the held-out young cohorts: +139% against the triangular engine over the year, worst month +581%.
The **root prior** is the change the cohort split asked for: with flat roots the rectangular fit
extrapolated 6.6% worse than the triangular one (se 2.9%, better in 3 of 12 months), because an
empty corner of the bilinear root surface goes wherever the coarse structure pushes it (-2.68
logits in the old-age, low-coupon corner of the February fit, against -1.82 triangular); with the
prior it is -0.69 and the cohort split is level. The **null cap** binds only when nearly every
candidate carries signal: in the first round of the SMM fit, seed 1, the central fit took the
signal for a null of sd 4 and the gate admitted nothing for three rounds and stopped with the nine
starting coefficients. With the cap the fit is level with the triangular one; it binds on no
other fit here. On the triangular engine the same cap leaves every single-issuer battery fit and
every SMM fit unchanged and improves 3 of the 36 pooled battery fits (by up to 28.7%), none worse:
a candidate change there too, for its owner to decide.

The log-linear variance is informative for what it found rather than for its score: fitted to
all admissions at once, its slope is essentially zero (a prior sd of 1.88 at depth 1 and 1.93 at
depth 8 on the February single-issuer fit). The admitted surpluses do not shrink with depth as
the interpolation-error argument predicts, which is what selection on the gate's threshold would
produce: only deep corrections large enough to pass are ever seen.

## Results

Against the triangular binary of 2026-09-22 (EM depth variance, linear binning), held-out
deviance, the rectangular engine as configured above:

| test | triangular | rectangular | rectangular vs triangular | better in |
|---|---|---|---|---|
| year battery, single-issuer (36 fits) | 1.4314 | 1.4276 | -0.24% (se 0.14%) | 23 of 36 |
| year battery, pooled (36 fits) | 2.3728 | 2.2064 | -4.7% (se 2.1%) | 21 of 36 |
| cohort split, single-issuer (12 months) | 1.978 | 1.932 | -1.8% (se 1.9%) | 5 of 12 |
| cohort split, pooled (12 months) | 6.285 | 6.601 | +7.8% (se 8.9%) | 7 of 12 |
| Ginnie design (10 seeds) | | | +0.38% (se 0.13%) | 2 of 10 |
| SMM design (5 seeds) | | | +3.2% (se 2.8%) | 2 of 5 |
| combined predictor, February 2020, seeds 1 / 2 / 3 | 1.180 / 1.181 / 1.336 | 1.175 / 1.187 / 1.342 | level | |

The year battery (the 2020 disclosure panel, every month, seeds 1-3, the single-issuer and pooled
base fits with joint population levels, `bench.base_fits 1 2 3 --month M` from the triangular
repo with `DMESH_BIN` pointing at each engine) is the criterion. The single-issuer fits are on
par, slightly better on average. The pooled mean is not a fair summary: a pooled split's held-out
deviance is dominated by one or two giant multi-issuer pools (December seed 1: a held-out pool of
95,424 loans at 8.6% carries most of it, and small changes to either engine move its prediction a
long way). The cohort split is level on the single-issuer fits. On the Ginnie design the
rectangular fit is a little behind (0.4%). On SMM it is level except seed 3, where a coefficient
admitted in thin data makes a spike that one held-out pool (179 months, 4.42%) happens to sit on,
predicted at 47% against 0.5% observed.

![battery](figures/battery_rect_vs_tri.png)

**The combined predictor.** The triangular repo's leveled benchmark
(`scripts/run_leveled_benchmark.sh <binary> <out> <seed>`: the base fits, population levels and
the cohort fields) runs on the rectangular base too, once `meshoverlay.mesh` reads the cell table
(added 2026-09-23, `interpolate_cells`: the transferred surface is interpolated at the
multi-issuer pools). Per population, triangular / rectangular:

| seed | all | C | X | M-JM | M-other |
|---|---|---|---|---|---|
| 1 | 1.180 / 1.175 | 1.160 / 1.153 | 1.100 / 1.111 | 1.906 / 1.651 | 1.287 / 1.243 |
| 2 | 1.181 / 1.187 | 1.175 / 1.179 | 1.076 / 1.080 | 1.130 / 1.367 | 1.233 / 1.232 |
| 3 | 1.336 / 1.342 | 1.134 / 1.135 | 1.146 / 1.141 | 6.860 / 7.470 | 1.224 / 1.221 |

(M-SF does not use the surface and is identical.) The triangular run reproduces its pinned 1.180.

![meshes](figures/meshes_feb2020_single.png)

The pinned February 2020 seed-1 fits: single-issuer 1.140 (triangular 1.1415) with 128 cells and
59 free coefficients (triangular 303 faces, 84 free); pooled 1.671 (triangular 1.7591). The gate
split its single-issuer admissions between the axes (28 on loan-age segments, 22 on coupon
segments); the fitted cells are mostly square (median aspect ratio 1, the most elongated 16:1), so
the aspect cap the triangular mesh uses was not missed. Two admissions in thin data make local
bumps (the figure, near 60 months at 5.1% and 3.0%): one has 9.5 effective pools and a held-out
neighbourhood above its fit, the other under one effective pool, admitted exactly at its round's
threshold. The triangular gate takes such candidates on the same terms, and requiring more
effective pools made the battery worse.

**Speed and memory** (this machine, 8 threads unless stated, the best of three runs):

| | triangular binary | rectangular binary | rectangular port |
|---|---|---|---|
| single-issuer fit (27,000 pools) | 1.10 s, 70 MB | 0.58 s, 45 MB | 9.2 s, 106 MB |
| pooled fit (30,000 pools) | 0.88 s, 69 MB | 0.57 s, 48 MB | 8.1 s, 106 MB |
| pooled fit, 1 thread | 2.4 s | 2.2 s | |
| Ginnie design (120,595 rows) | 1.95 s, 169 MB | 1.37 s, 103 MB | |
| year battery (72 fits; binaries 6 months at a time, port 8) | 33 s | 28 s | about 2 min |

What remains of the rectangular fit's time is the pool quadrature shared with the triangular
engine (about 70% of a single-threaded fit); in the port, the Golub-Welsch count quadrature for
large pools (batched `eigh`) is another 17%, shared with the triangular port.

**The port** agrees with the binary on the pinned fits to the last printed digit on every table
of the single-issuer fit and on the mesh, vertex and surface tables of the pooled fit (the rest
within 1e-9), on the SMM fit where the null cap binds, and on all 72 battery fits to every printed
digit of the held-out deviance.

**Checks:** `ctest` (the geometry contracts on 12 random meshes), `pytest pyrect/tests` (17 tests:
the port against the binary on synthetic designs, the engine's own split, a large design, joint
levels, the external baseline, malformed and edge-case inputs, and the port's geometry
contracts), dumps identical at 1, 3 and 8 threads.

## Open

- A fine node, the rectangular counterpart of the triangular two-node hierarchy, was built and
  measured, and not kept. The formulation that works: a coefficient at a free vertex p whose
  neighbourhood a free child has refined, with a fixed injection set at its creation (+1 at p,
  and at each descendant that is then regular the value that holds it at zero, hanging ones
  following their edges), so its column is p's nodal hat on the leaves of that moment, the surface
  stays continuous, later splits stay inert, and in nodal coordinates it enters the closures and
  the surplus transform like any coefficient (the geometry test checked all four). On the year
  battery it gave -0.07% (se 0.04%) single-issuer and +0.8% pooled, with two fine nodes admitted
  on the benchmark fit: the two-node hierarchy repaired a triangular problem (a vertex whose star
  had been taken over lost its coarse role in the nodal basis) that the rectangular hierarchical
  basis does not have. The code was removed; the formulation is here if it is wanted again.
- The selection "bias" of the depth variance. The EM variance is estimated from gate winners
  only, and a simulation with a Gaussian truth put it about three times too high. Three attempts
  to correct it all lost or gained nothing. A likelihood of the admission estimates conditional
  on their selection (b = z sd ~ N(0, tau^2 + sd^2) truncated at |b| >= c sd) gave -0.01% (se
  0.05%); its estimates ran to the edge of the search range at some depths, and four fits
  failed on the absolute semidefinite test the solver then had. A log-linear variance fitted to
  all admissions at once gave -0.04% (se 0.06%), with a flat profile. And the plain marginal
  likelihood of every scored candidate, admitted or not, which has no selection at all, made
  the battery clearly worse (+0.78%, se 0.14%): its estimates were erratic across depths (sd
  0.12 at depth 2, 3.98 at depth 4) because a Gaussian fitted to a population of mostly-null
  candidates with a few strong ones is a poor compromise. That last result reframes the item:
  the gate decides inclusion and EM estimates the variance of the included coefficients, which
  together behave like a spike-and-slab prior, and against a spike-and-slab truth the winners'
  variance is the slab's, not a biased Gaussian variance. The explicit two-component model was
  then tried too: each depth's candidates as (1 - pi) N(0, sd^2) + pi N(0, tau^2 + sd^2), fitted by
  EM to the previous round's scores, the slab variance as the prior. It was worse (+0.37%, se
  0.11%), its slab variance jumping between rounds (sd 10 at depth 4 in round 0, 0.005 at depth 3
  in round 2). A round's candidate estimates are strongly dependent (their columns overlap and
  all are scored against one frozen fit), which any likelihood treating them as independent
  draws misreads. EM on the winners stays; it was never beaten here.
- The giant pools that decide pooled splits, shared with the triangular engine.
- Coefficients admitted in thin data can make local spikes (SMM seed 3: a held-out pool at 179
  months predicted at 47% against 0.5%; the bumps near 60 months in the February figure). The EM
  depth variance at the deep levels is large (sd 2-4 logits), so such a coefficient is barely
  shrunk. Requiring effective pools of a candidate did not help (the battery got worse).
- Three changes made here would apply to the triangular engine unchanged and are for its owner
  to take or leave: the null cap at 3 (measured there above: no single-issuer or SMM fit changes,
  3 pooled battery fits improve), a round with no actual admission counted as a stall (the
  triangular SMM fit, seed 1, idles for 70 rounds without it), and the stalled-Newton fallback in
  the surface solver. The root prior is not one of them: on the triangular engine it is neutral
  on the battery (+0.04%, se 0.05%) and does little on the cohort split (-0.7%, se 0.6%). The
  triangular roots make two planes, which extrapolate like planes; it is the rectangular root
  surface's bilinear twist that swings an empty corner.
