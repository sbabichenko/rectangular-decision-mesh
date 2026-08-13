# Proper regression on the known-cosine testbed

Design working document, 2026-08-13. Companion to `GEOMETRY_DESIGN.md`.
Decision: before touching Ginnie data, the rectangular mesh's statistical
layer is developed against known cosine-signal truth with Gaussian noise.
This file says what "properly regress" means there and why.

## 0. The testbed

Truth surface: the triangular repo's chirp (`core/surface.h`), ported
verbatim for cross-mesh continuity. On simulator coordinates
x, y in [-4, 4], amplitude A = 2:

    z      = (x + 4) / 8
    z_warp = (0.2 + 0.8 z) z
    xr     = -4 + 40 z_warp
    coarse = exp(0.80 cos(0.72 xr + 0.44 y + 0.25)) - 1.1665149229
    fine   = exp(0.58 cos(2.65 (-0.36 xr + 0.93 y) - 0.40)) - 1.0859474
    f      = A (0.62 coarse + 0.38 fine)

Horizontal frequency accelerates to the right (the warp), so the optimal
mesh is anisotropic and spatially graded — the surface is engineered to
exercise both-axes candidacy and depth grading. Data: iid uniform
(x_i, y_i), z_i = f(x_i, y_i) + eps_i, eps ~ N(0, sigma^2), sigma known
to the simulator but estimated by the fitter.

## 1. Why Gaussian-first is the right order

The triangular pipeline's tau machinery is a DerSimonian–Laird moment
estimator with five audited mechanical defects (`ESTIMATOR_LESSONS.md`
§3) — approximations forced by the binomial/IRLS setting. In the linear
Gaussian setting nothing needs approximating: posteriors are closed
form, the marginal likelihood is exact, and hyperparameter estimation is
EM-exact. So the cosine phase builds the **exact-inference reference
implementation**. Later, on the road back to binomial data, every
approximation is verified against this reference on overlapping ground
(weighted Gaussian with per-point precisions is the bridge model). The
lessons doc stops being a checklist of patches and becomes a set of
theorems the reference implementation satisfies by construction.

## 2. The model

- Surface: f = sum_v delta_v B_v in the hierarchical (surplus)
  parameterization; B_v is the basis column transported through the
  constraint closure (PoC `design_hier`, verified in demo 2: promotion
  adds a column without moving any other).
- Prior: delta_v ~ N(0, tau_d(v)^2) independent, zero mean by
  construction (defect 4 impossible). Root corners: flat (or sd 10^3).
- Likelihood: z | delta ~ N(X delta, sigma^2 I).
- Joint MAP == posterior mean: solve
  (X'X / sigma^2 + Lambda) delta = X'z / sigma^2,
  Lambda = diag(1 / tau_d(v)^2). Shrinkage enters ONLY through Lambda
  inside the joint solve (lesson 5); no post-hoc coordinate edits, ever.

## 3. Hyperparameters: exact EB by EM, not moments

Estimate {tau_d} and sigma^2 by maximizing the exact marginal likelihood
with EM (equivalently REML for sigma^2):

- E-step: posterior mean m and covariance S from the current solve
  (sparse Cholesky of the penalized information; S never formed densely,
  only the needed diagonal/contrast blocks).
- M-step:
  tau_d^2 = mean over free v at depth d of (m_v^2 + S_vv),
  sigma^2 = [ ||z - X m||^2 + tr(X S X') ] / n.

How this dissolves the five DL defects structurally:

1. No excess-chi^2 denominator exists to get wrong.
2. A zero-information vertex has m_v ~ 0, S_vv ~ tau_d^2, so it
   contributes ~tau_d^2 to its depth's M-step average: *no vote in
   either direction*. The empty-star pathology disappears without an
   exclusion flag or a membership rule.
3. There is one estimator, used everywhere; no in-loop/final-pass
   disagreement to reconcile.
4. Zero-mean is the model, not a convention choice.
5. The E-step uses posterior moments given the current hyperparameters —
   the principled version of "measuring your own output". Plugging
   already-shrunk POINT estimates into a moment formula (the triangular
   defect) is exactly EM with the S_vv term deleted; the cosine phase
   can measure how much that deletion costs (see E1).

Floors: keep a tau floor as an explicit prior bound, applied in the
M-step only, never mutated by any pass (the weld lesson). tau = 0 at a
depth means "fully pooled", i.e. lambda at the floor cap — welding is
not representable in this stack at all, which is the point.

## 4. Honest uncertainty

All reported variances come from the full penalized-information inverse
via sparse Cholesky: coefficient variances S_vv, contrast/prediction
variances a' S a (lesson 4: the dominant optimism term in the
triangular audit was joint-fit correlation, curable only this way).
Known truth makes calibration measurable, not aspirational:
sd(z_err) = sd( (f_hat - f_true at probes) / claimed sd ) with target 1.

## 5. What the cosine world can and cannot validate

CAN: estimation (tau, sigma recovery), shrinkage behavior across
depths, honest-se calibration, adaptivity/approximation quality,
anisotropic axis choice, composite-move accounting.

CANNOT: the gate's null calibration / FDR story. Smooth truth means no
delta is exactly zero — there are no true nulls here. Null and sparse
worlds (the triangular scenario battery: complete null, histogram null)
are a separate later phase; do not read gate error rates off the chirp.

## 6. Experiment program

- **E1 — fixed topology, no selection.** Uniform meshes at several
  depths (and hand-graded anisotropic ones). Verify: EM recovers
  sigma^2; tau_d profile is stable and sensible (for smooth truth,
  tau_d should decay ~4x per depth — bilinear surpluses scale like
  h^2 |d2f|, i.e. the *measured* analogue of the triangular repo's
  4^{-dd} borrowing heuristic); coverage of coefficient and prediction
  intervals ~nominal; EM-without-S_vv (the DL-style plug-in) quantified
  against exact EM on identical data.
- **E2 — oracle frontier.** Refine greedily on TRUE local bilinear
  approximation error (computable from the known surface). Produces the
  error-vs-coefficient-count frontier N -> RMSE_oracle(N): the
  achievable boundary, independent of estimation.
- **E3 — adaptive selection.** Greedy composite moves scored by exact
  marginal-likelihood gain (Bayes factor of adding delta_v under
  current tau_d), both axes always offered, release splits included.
  Report: regret RMSE(N) - RMSE_oracle(N); x/y cut mix vs the oracle's;
  calibration under selection (how much E1's honest coverage degrades
  once topology is data-chosen); tau-by-(depth,axis) measurement for
  GEOMETRY_DESIGN §7.4.
- Holdout: iid split is adequate for synthetic iid data; report
  train/test/noiseless-truth metrics separately (as the PoC does).
  The locally-balanced-split machinery matters only for real pooled
  data and stays out of scope here.

## 7. Non-goals of this phase

No binomial/IRLS, no pool effects, no one-law machinery, no gate/FDR
accounting, no Ginnie data. Each arrives only after the exact-reference
stack passes E1–E3, so that every future approximation has something
exact to be checked against.
