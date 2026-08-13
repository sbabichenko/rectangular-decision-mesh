# Porting FDR control from the triangular mesh

Design working document, 2026-08-13. Source machinery: triangular
`core/lfdr.cpp` (Lindsey lfdr with clamped empirical null),
`core/pipeline.cpp` gate rounds (candidate scoring, B1 exact-gain
invariant, sequential commit with live re-profile), and the aug6/aug12
writeup leads (release rule, e-BH budget, two-groups audit).

## 1. What the triangular gate does (inventory to port)

Per gate round:

1. **Candidate family**: every inactive edge midpoint (and active
   non-free conformity vertex) with support (>= 10 fit points, K_eff
   floor). Geometry proposes candidates for free via the NVB structure.
2. **Candidate score**: displacement (Wald or exact-likelihood profile
   MLE) over a candidate-specific calibrated null sd:
   `gate_var = 1/xTx + GEV * xT2x/xTx^2` (+ parent posterior variance,
   x quasi-likelihood dispersion). The extra-variance term prices
   pool-level noise so admission means "replicated beyond pool noise";
   its continuous form is a min-effective-pools gate.
3. **B1 invariant**: no candidate without a positive exact penalized
   gain against the P1-preserving null (`mu_lin`) may cross the
   boundary, regardless of its z.
4. **Family decision**: Lindsey's method (Poisson GLM histogram fit,
   degree 6) -> lfdr; admit the mean-lfdr <= q prefix. Empirical null
   by central matching with explicit clamps (sigma0 box, pi0 >= 0.5,
   |delta0| <= 0.5 — documented as the least principled part). BH on
   p-values as fallback / variant.
5. **Sequential commit + live re-profile**: batch scores go stale as
   earlier commits move parents; every admission is re-profiled at the
   live state. Geometry is realized at zero surplus FIRST, then the
   statistical decision is made on the actual active column.
6. **Selection records**: `admit_a` (the round's admission threshold in
   delta-hat units) and `admit_sd` per admission, feeding the
   truncation correction in tau estimation.
7. Rounds until stall (3 stalls, with sweeps / IRLS refresh between).

## 2. What ports unchanged

- The **round structure** and family-per-round decision.
- The **B1 invariant** (positive exact gain or no admission).
- **Sequential commit + live re-profile**, verbatim in spirit: commits
  are ordered, each re-scored on the live design.
- The **geometry/statistics separation** — and the rectangular version
  is *stronger*: the inert-split lemma (GEOMETRY_DESIGN §2b) guarantees
  materializing a candidate's composite geometry (release + cut, all
  welded) changes the surface by exactly nothing, so candidates are
  profiled on their actual live hierarchical column at literally zero
  cost, and **rejected candidates' geometry can be garbage-collected**
  (coarsening) — the triangular mesh must keep its conformity cascade
  geometry forever.
- **Candidacy support floors** (fit points, K_eff) — as *candidacy*
  membership rules only; tau needs no membership rule in the EM stack
  (REGRESSION_DESIGN §3).
- `gate_admitted` as the FDP-accounting flag, set only at admissions.

## 3. What changes shape

**Candidate unit.** Triangular candidates are vertices whose geometry
already exists. Rectangular candidates are *coefficients attached to
composite moves*: (cell, axis) cuts proposing their chord-endpoint
coefficients, plus promotable existing welds, plus hanging vertices
with their release splits (§7.3 composite pricing). Both axes are
always offered (§7.1), roughly doubling the per-cell family; the two
chord coefficients of one cut, and the x/y candidates of one cell,
share data — within-family dependence is *structural*, not incidental.
This raises the stakes for the multiple-testing rule's dependence
robustness (see §4).

**Null variance.** In the Gaussian cosine phase the candidate null is
exact: z = beta_hat / (sigma / sqrt(xTx)) is N(0,1) under the null, no
dispersion estimate, no GEV, no pool pricing. The GEV / one-law /
sandwich structure is a *binomial-phase* port; the Gaussian phase
instead validates the gate mechanics themselves against measured FDP.

**Empirical null.** Lindsey + central matching exists because the
binomial pipeline's null scale drifts with fit state. Gaussian phase:
the theoretical null is correct by construction, so Lindsey runs as an
*audit overlay* (it should measure sigma0 ~ 1, pi0 ~ true null share),
not as the primary decision — flipping the triangular arrangement. Its
clamps (sigma0 box, pi0 floor) finally become measurable: on worlds
with known null share, run free-null Lindsey and record when the
clamps would have bound. Primary status returns in the binomial phase.

## 4. The structural upgrade: e-values as the native currency

The aug6 session left "release rule + e-BH budget" as a design lead.
The rectangular gate should adopt it from day one, because the pieces
already exist here:

- The B1 exact penalized gain is (log of) a **Bayes factor**: the
  candidate's marginal likelihood ratio against the P1-preserving null
  under the current depth prior delta ~ N(0, tau_d^2). Under H0,
  E[BF] <= 1: it is an **e-value**, natively.
- **e-BH** (BH on e-values) controls FDR under *arbitrary dependence* —
  exactly the structural dependence the doubled rectangular family has
  (§3). No PRDS argument needed for correlated sibling candidates.
- e-values compose across rounds by multiplication/averaging: a
  candidate scored in round after round accumulates one running
  e-process instead of repeated fresh looks — the anytime-validity
  question the triangular round structure never formally answered.
- The release rule (retiring/demoting an admitted coefficient when its
  evidence decays) becomes spending from the same e-budget rather than
  a separate device.

Gaussian instantiation (cosine phase): for candidate v with column x,
residual r, the Bayes factor against tau-prior N(0, tau_d^2) is closed
form:
  BF = sqrt( sigma^2/(sigma^2 + tau_d^2 xTx) )
       * exp( tau_d^2 (xTr)^2 / (2 sigma^2 (sigma^2 + tau_d^2 xTx)) ).
Plan: run BH-on-p and e-BH side by side on the same candidate stream
and compare realized FDP and power. BH is the triangular-faithful
control arm; e-BH is the proposed native rule.

## 5. Validation worlds (rectangular-exclusive advantage)

Because Q1 dyadic refinement is nested (inert-split lemma), a truth
surface built as a bilinear surface on a coarse rectangular mesh is
EXACTLY representable at every finer topology: every coefficient
beyond the truth mesh has true delta exactly 0. This gives true-null
labels with none of the approximate-null caveats the triangular
true-null battery carries (its README documents the null definition as
an upper-bound approximation).

- **W0, complete null**: f = 0. Every admission is false. Measures
  realized FDR directly; any excess is a mechanics bug, not a
  modeling judgment.
- **W1, coarse-truth**: random bilinear truth on a depth-2 mesh.
  True signal at depth <= 2, exact nulls deeper. Measures FDR and
  power together, plus the misfit-leakage regime (early rounds see
  underfit-induced same-sign signal at null positions — the exact
  phenomenon the triangular pi0 >= 0.5 clamp guards; here it is
  measurable).
- **W2, chirp**: no true nulls (REGRESSION_DESIGN §5); used for power/
  adaptivity only, never FDR claims.

Metrics per world: realized FDP distribution vs nominal q across
replicates, power (share of true coefficients recovered), and for W1
the depth profile of false admissions.

## 5b. Findings from the PoC port (2026-08-13, measured)

Two mechanisms surfaced immediately on W1 and are now design rules:

1. **Every non-free vertex must be a standing candidate.** A first cut
   of the PoC proposed only chord endpoints of new leaf cuts; welds
   parked by earlier partial moves were never re-proposed. Result
   (W1, seed 4): the truth knot at the domain-center weld stayed
   unpromotable, its signal stranded in the residual, and the gate —
   working correctly on contaminated candidates — bought the same
   structure as a staircase of 20 false admissions along the
   stranded knot's line. Making parked welds/hanging vertices
   standing candidates (scored on exact live columns) collapsed this
   to 5; the general rule: any reachable coefficient the gate cannot
   revisit converts its signal into false admissions elsewhere.
   (Triangular analogue: inactive midpoints remain members of future
   candidate families — the pipeline comment says exactly this.)
2. **Live re-profile must see deconfounded residuals.** With several
   strong surpluses unexplained in early rounds, every candidate's
   projection is contaminated (the writeup's "coarse underfit creates
   widespread same-sign signal"), and BH admits true and false
   together in one round. A one-coordinate update between sequential
   commits is too weak: later commits still see the leaked signal and
   their live check passes. Refitting the full penalized solve after
   EVERY accepted commit makes the sequential live re-profile honest;
   on W1 seeds 2-8 this took false admissions from 5/8 to 0/27 with
   power intact. This is the triangular "re-profile at the actual
   commit state" taken to its logical end; the triangular pipeline
   approximates it with sweeps between rounds and should possibly
   port this BACK.

## 6. Port order

1. PoC gate (`poc/gate_poc.py`): Gaussian scoring, B1 invariant,
   BH and e-BH arms, sequential commit with live re-profile, W0/W1
   replicated FDP measurement, W2 power demo. (This file's companion.)
2. Lindsey port as audit overlay (Python; measure the clamps on W0/W1).
3. Truncation records (`admit_a`/`admit_sd`) wired into the EM stack's
   selection-aware variant when E3 (REGRESSION_DESIGN) shows they are
   needed.
4. Binomial-phase ports (GEV/one-law/sandwich null pricing, empirical
   null as primary) — only after the Gaussian gate passes W0/W1.
