# Estimator lessons from the triangular mesh

Distilled from the triangular-decision-mesh audit sessions (July 30 -
August 12, 2026). These failure modes live in the statistical layer, not
the geometry: expect every one of them to try to reappear in the
rectangular implementation. Treat this as a design-review checklist, not
history. Source details: `experiments/aug6_session/`,
`sec_aug6_session.tex` (the aug7 addendum), and `sec_aug12_shrinkage.tex`
in the triangular repo.

## 1. Floors must be respected by every pass that solves for tau

The lambda=1e8 "wholesale weld": a final re-shrinkage pass lowered the
tau floor to 1e-8 for its own sweep; a later retirement pass re-ran the
in-loop tau moment over already-shrunk surpluses (chi^2 ~ 0) under the
gutted floor, so tau=0 -> max(0, 1e-8) -> lambda=1e8, freezing every
coefficient at those depths regardless of individual information. Rules:

- A tau=0 moment read means "shrink hard to the floor", never "weld".
- Never mutate a shared hyperparameter (the floor) as pass-local state.
  If a pass needs different behavior, pass it explicitly.
- Any pass that re-estimates tau from *already-shrunk* quantities is
  measuring the estimator's own output and can self-reinforce to a
  degenerate fixed point (see 3.5).

## 2. Zero-information voters in hyperparameter moments

Support-starved ("contained") vertices — stars over near-empty space,
K_eff < 0.5 or < 10 fit points — were frozen at their prior by the height
update yet still voted in the tau moments with stale admission-time
sigma^2 and noise surpluses. Each costs a degree of freedom (m-1) while
contributing ~0 to chi^2 and sum(w): tau is deflated where their deltas
are contained (one depth's single real workhorse got welded because two
of three voters were K_eff ~ 0.01) and inflated where they are wild.
Rule: membership in a variance-component moment requires actual
information (fresh sigma, real support); zero-information units stay
fully pooled to the prior and do not vote. Caveat: on seed7 the
exclusion read slightly *worse* held-out via topology feedback —
fresh-seed confirmation before adopting (see 9).

## 3. The DerSimonian-Laird audit (five mechanical defects)

The per-depth tau moment was DL with a floor and nearest-depth
borrowing — a sound skeleton with five defects to avoid from day one:

1. **Denominator**: the excess-chi^2 denominator must be S1 - S2/S1, not
   S1 = sum(w). With weights spanning four decades (K_eff 0.01-100),
   using S1 materially underestimates tau (over-shrinks).
2. **Membership**: see 2 — no zero-information voters.
3. **Truncation-correction consistency**: the selection/truncation
   correction was opt-in in-loop but always-on in the final pass, so the
   two estimators disagreed by construction. One convention, everywhere.
4. **Mean model**: for surpluses defined about the parent interpolation,
   zero-mean is the right model; a fitted per-depth mean absorbs
   depth-wide signal out of tau. Pick the mean model from the definition
   of the residual, and use the same one in-loop and post-hoc.
5. **Self-measurement**: the in-loop moment consumed jointly-fitted
   heights already shrunk by the previous tau — the estimator partly
   measures its own output, the mechanism behind the weld. Recover raw
   (unshrunk) surpluses for hyperparameter estimation.

## 4. Reported uncertainty is optimistic; the fix is structural

Known-truth simulation on the real design geometry: claimed coefficient
standard errors understated realized error 2.2x in a *perfectly
specified* world and 3.2x under issuer-block effects. The dominant term
is joint-fit correlation, not winner's curse: contrast variances read
off the full penalized-information inverse restore sd(z_err) ~ 1.1; a
law-plus-block sandwich Omega handles the misspecified world. Build the
rectangular scorer's se's from the joint factorization from the start —
per-coefficient sigma^2 from a local solve will be optimistic for the
same reason regardless of geometry.

## 5. Shrinkage enters only through the prior inside the joint solve

Post-hoc coordinatewise re-shrinkage is refuted: applying audit-derived
shrinkage marginally to a jointly-fitted surface degraded oracle wMSE
catastrophically (0.033 -> 0.159). Coefficients are fitted jointly and
their errors cancel; marginal edits break the cancellation. Audits may
flag; only Lambda inside the solve may shrink.

## 6. The law fixed point is per-regime and per-model

Pinning a pool-law calibration across regimes (or across model variants
that absorb different shares of pool-effect variance) converts law
misspecification into fake structural conclusions — one "the model loses
these months" finding died this way. Refit the law fixed point whenever
the regime or the model changes.

## 7. Certificates must quote the true footprint of a coefficient

The false-certification anatomy: constraint closure (conformity-welded
descendants) gave a coefficient an effective design column spanning a
footprint far beyond its nominal one-ring; the fit compromised between
two populations and rendered the compromise where the losing population
lived. The triangular mechanism (midpoint welds) is geometry-specific,
but bilinear meshes have their own version — hanging-node constraints on
refinement boundaries create exactly this kind of extended footprint.
Rule: any certificate or audit quotes the coefficient's footprint
*through its constraints*, not its nominal stencil.

## 8. Holdout design: pools stay whole, splits stay local

CSV row parity leaked pool structure across the split. Real-data
holdouts keep whole pools together inside locally balanced blocks (the
triangular repo used WALA/WAC blocks). Port the split design, not just
the data.

## 9. Reproducibility discipline

- Behavioral changes ship as env patches, **default off**, until
  confirmed; the pre-change behavior stays reachable for byte-level
  reproduction of archived runs.
- Frozen benchmark inputs (this repo carries the same
  `data/ginnie_design.csv`, 120,595 rows, pooled rate 0.783241, and
  `data/smm_design_202606.csv` as the triangular repo) so cross-mesh
  comparisons are on identical data.
- **Fresh-seed confirmation before adoption**: a change that helps the
  seed it was diagnosed on has not yet demonstrated anything (the
  containment exclusion helped mechanism-wise and still read worse
  held-out on its diagnosis seed).
- Historical validation numbers from unavailable inputs are retained as
  history, never as a scoreboard.
