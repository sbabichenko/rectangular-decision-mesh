# Split and weld geometry for the rectangular mesh

Design working document, started 2026-08-13. Settles (proposes) the two
foundational geometric questions: what a split is, and what a weld is.
Everything statistical (gates, tau, scoring) sits on top of these two
definitions, so they are worth getting right before any solver code.

## 0. What the triangular mesh does (the reference design)

For contrast, the triangular repo's geometry (`core/face.cpp`,
`core/vertex.h`):

- **Split primitive**: newest-vertex bisection (NVB) on right triangles.
  A face splits along its refinement edge (`edges[0]`, the hypotenuse);
  the new vertex is that edge's midpoint.
- **Conformity by completion**: if a face is asked to split along a
  non-refinement edge, it first activates its own hypotenuse midpoint
  (`Face::split`: the completion branch), recursively. The cascade
  terminates and the mesh is *always conforming* — every active vertex
  is a regular vertex of every face that touches it.
- **Welds are statistical, not geometric**: a completion midpoint is
  active with `free_coefficient = false` and `conformity_origin = true`,
  held at `mu_lin = (h_p0 + h_p1)/2` (zero hierarchical surplus). The
  lfdr gate may later promote it to a free coefficient *without any
  further geometric work*, precisely because conformity already holds.
- **Surplus**: `delta_v = h_v - (h_p0 + h_p1)/2`, always exactly two
  parents (the parent edge's endpoints). Depth-indexed tau prices it.

The one-knob-two-rooms disease (see `ESTIMATOR_LESSONS.md` §7) arises
when a *free* coefficient carries conformity-welded descendants: its
effective design column extends through the welded children's
interpolation into territory beyond its one-ring.

## 1. The split primitive

Three candidates:

**(a) Quadrisection (quadtree).** Cell -> 4 congruent children. Creates
up to 5 new vertices: 4 edge midpoints + 1 center. Pros: canonical,
isotropic, depth is a single integer. Cons: two *kinds* of new vertex
(edge midpoints with 2 natural parents; the center with 4), the
refinement quantum is large (a cell that needs x-resolution is forced to
buy y-resolution too), and no axis adaptivity.

**(b) Anisotropic dyadic bisection.** Cell -> 2 children by cutting at
the midline of a chosen axis. Creates exactly 2 new vertices (the chord
endpoints, midpoints of the two edges perpendicular to the cut). Pros:
every new vertex has exactly 2 parents — the surplus definition
`delta_v = h_v - (h_a + h_b)/2` carries over from the triangular mesh
unchanged, one weld formula everywhere; the refinement quantum is
halved; axis choice is a *statistical* decision the gate machinery can
own (score the x-cut and y-cut as competing candidates), which directly
attacks the boundary-anisotropy disease of the triangular basis; the
tree stays binary, so the triangular `TreeNode` walk
(`split_normal`/`split_intercept` with axis-aligned normals) carries
over as-is. Cons: cells have heterogeneous aspect ratios (needs a cap,
as the triangular mesh caps at 5.0); "depth" becomes a pair
`(lx, ly)` of per-axis levels; hanging-node bookkeeping is the same as
(a) but arises on more edges relative to area refined.

**(c) Data-adaptive cut location** (kd-tree style, cut anywhere). Ruled
out: non-dyadic cuts destroy the shared-vertex hierarchy (no
`coord_map` quantization, no reusable parents), and turn the prior
stencil into a weighted interpolation with data-dependent weights —
the selection machinery would then be choosing its own prior geometry.
Dyadic midline cuts only.

**Proposal: (b), anisotropic dyadic bisection, with quadrisection
available as a composed move** (an x-cut followed by y-cuts of both
children, or vice versa; the composition also creates the center vertex,
which is then an ordinary 2-parent midpoint of the *chord*, not a
4-parent special case). This keeps exactly one vertex-birth rule and
exactly one weld formula in the whole design.

Consequences to accept explicitly:

- **Depth bookkeeping**: a vertex born on a cut of a cell at level
  `(lx, ly)` has scalar depth `d = lx + ly + 1`. Tau stays indexed by
  scalar depth to start (matching the triangular pipeline's shape); if
  anisotropy turns out to matter statistically, tau can later be
  indexed by `(d, axis)` — the lesson doc's fresh-seed discipline
  applies to that upgrade like any other.
- **Aspect-ratio cap**: `|lx - ly| <= A` (A = 2 to start, i.e. 4:1
  physical aspect ratio on a square domain). The cap is a *candidate
  filter*, not a forced split: a cut that would exceed the cap is
  simply not offered to the gate. This differs from the triangular
  `max_aspect_ratio`, which polices a continuous quantity; here it is
  exact and integer.

## 2. The two species of rectangular weld

This is the central structural difference from the triangular mesh.
NVB completion makes conformity affordable: a bounded cascade of extra
midpoints restores it exactly. For rectangles no such move exists — an
adaptively refined quad mesh with different levels in adjacent cells is
*inherently* non-conforming at the shared edge. The standard resolution
(constrained approximation) gives us the second weld species.

**Species 1 — statistical weld (promotable).** Identical in meaning to
the triangular weld. A newly born chord-endpoint vertex whose
surrounding cells all exist at compatible levels ("regular" vertex) is
born active with no coefficient, held at
`h_v = (h_a + h_b)/2` (parents = the endpoints of the edge it
subdivides). Zero surplus, prices nothing, and the gate may promote it
to a free coefficient at any time with no geometric side effects.
`conformity_origin` generalizes to `birth_origin` (chord endpoint vs
balance-induced, §3).

**Species 2 — geometric weld (hanging node).** A vertex on the shared
edge between a fine cell and a *coarser* neighbor. Bilinear restricted
to an edge is linear, so continuity across the shared edge holds iff
the fine side reproduces the coarse side's linear trace: the hanging
vertex must satisfy exactly

    h_v = (h_a + h_b)/2

where `a`, `b` are the coarse edge's endpoints. Same formula as species
1 — but a different *contract*: the constraint is mandatory while the
neighbor stays coarse. A hanging vertex **cannot be promoted** without
breaking surface continuity. Promotion acquires a geometric
precondition:

> **Promotion rule.** The gate may admit a coefficient at vertex v only
> if v is regular. If v is hanging, admitting it requires first
> refining the coarse neighbor(s) whose edge constrains it (a
> *release split*), which converts v to regular and typically creates
> new species-1 and species-2 welds of its own.

This is the rectangular analogue of the NVB completion cascade, but
with the cost moved: NVB pays geometry *at split time* to keep every
vertex promotable; rectangles pay nothing at split time and pay
geometry *at promotion time*, only for vertices the gate actually
wants. The release split must be priced into the candidate's score
(it changes the design; per the triangular repo's rule, design-changing
events re-price everything they touch).

Note the happy algebraic accident: both species share one formula, so
`delta_v = h_v - (h_a + h_b)/2` with the appropriate parent pair is
*the* surplus definition mesh-wide, and a hanging vertex is precisely a
vertex whose surplus is structurally pinned to zero. The tau moments
must treat species 2 like the triangular repo's contained vertices:
structurally-zero surpluses are not evidence about tau
(`ESTIMATOR_LESSONS.md` §2 — they never vote).

## 3. Balance policy: 2:1, enforced, with welded byproducts

Unrestricted level differences let hanging constraints chain (a master
of one constraint is itself the slave of another), so a coefficient's
effective column closure can grow without bound — the footprint disease
compounded. Standard remedy, adopted here:

**2:1 edge balance.** Adjacent cells (sharing an edge segment) may
differ by at most one level per axis. A split that would violate
balance triggers the minimal set of neighbor splits to restore it
(bounded cascade, as in quadtrees). Balance-induced splits create
vertices born as species-1/species-2 welds (`birth_origin = balance`),
never free coefficients — the exact analogue of NVB completion
vertices, and like them they are "structural, not selected slab draws"
for every selection-aware correction.

With 2:1 balance, a hanging vertex's two masters are corners of the
coarse cell. A master can still itself be hanging (on a *different*
edge, against a diagonal-coarser region), so constraints can chain to
depth 2 in corner configurations. Decision: **store constraints in
closed form** — each hanging vertex holds its fully resolved master
list `{(master_j, w_j)}` with all masters regular, recomputed on any
topology change to its neighborhood. Never resolve lazily at
evaluation time. This makes the footprint of every coefficient a
concrete stored object, which is what §5 needs.

## 4. Basis, evaluation, and the initial mesh

- **Basis**: Q1 bilinear hat per free coefficient over its (up to 4)
  surrounding cells, *transported through the constraint closure*: a
  free coefficient that serves as master to hanging vertices extends
  its column into the fine cells those hanging vertices touch, with
  weight `w_j` times the fine cell's bilinear weight. Continuity is
  exact by construction (linear edge traces + midpoint constraints).
- **Point location**: binary tree descent on axis-aligned cuts —
  identical control flow to the triangular `DecisionMesh::predict`
  walk, with the barycentric step replaced by two normalized local
  coordinates `(u, v)` and the 4-corner bilinear form. The per-point
  cache (`point_face`/`point_bary`/`point_basis`) carries over with a
  4-wide basis vector; cells own disjoint point subsets exactly as
  faces do now.
- **Initial mesh**: one root cell on the unit square, four corner
  vertices, no diagonal. (The triangular mesh's outer split into two
  faces exists only because triangles must tile the square; rectangles
  do not need it. One fewer arbitrary orientation choice — the
  diagonal's direction was itself a mild anisotropy the rectangular
  design gets rid of for free.)

## 5. Certification footprint (the §7 lesson, made structural)

Because constraint closures are stored (not implicit), define once,
in the geometry layer:

    footprint(v) = cells where v's basis is nonzero, THROUGH:
      (i)  its own bilinear support,
      (ii) species-1 welded descendants interpolating from it,
      (iii) hanging vertices (species 2) for which v is a resolved
            master.

Every certificate, audit, and gate variance quotes `footprint(v)` —
never the nominal 4-cell support. The triangular repo discovered this
after a false certification; here it is a day-one invariant with a
natural home, because §3 already forces the closure to exist as data.

## 6. What the gate sees (summary of geometry -> statistics contract)

| vertex state        | surplus                  | votes in tau? | promotable?            |
|---------------------|--------------------------|---------------|------------------------|
| free coefficient    | fitted, priced by tau(d) | yes (if informative) | already free    |
| species-1 weld      | structurally 0           | no            | yes, no side effects   |
| species-2 (hanging) | structurally 0           | no            | only via release split |
| dormant / retired   | frozen / 0               | no            | per triangular rules   |

Everything below the line in the triangular pipeline (admission gates,
tau estimation, reshrink, holdouts) consumes only: vertex states,
surpluses, depths, and stored footprints. That is the interface this
geometry must deliver; the estimator lessons doc constrains how the
statistical side may use it.

## 7. Open decisions (not yet settled)

1. **Axis-choice policy for bisection candidates**: offer both axes as
   separate gate candidates always, or pre-filter by a cheap
   directional-curvature diagnostic? (Both-always is cleaner and lets
   the gate own the decision; it doubles candidate count.)
2. **Aspect cap A**: start at 2 (4:1)? The Ginnie design's natural
   WALA/WAC anisotropy should inform this — measure before choosing.
3. **Release-split pricing**: is a release split scored as part of the
   candidate's gain (one composite candidate), or must the neighbor
   split be independently justified first (two gate events)? Composite
   risks smuggling unjustified refinement in on a strong candidate's
   coattails; sequential risks never seeing strong hanging candidates
   at all. Leaning composite-with-full-repricing, but this deserves a
   known-truth simulation before commitment.
4. **Tau by (depth, axis)** — deferred until anisotropy is measured
   (§1 consequences).
