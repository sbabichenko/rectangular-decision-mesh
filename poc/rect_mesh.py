"""Proof-of-concept rectangular decision mesh (GEOMETRY_DESIGN.md).

Implements the geometry layer only, on the unit square with integer dyadic
coordinates: anisotropic dyadic bisection (x- and y-cuts), the two weld
species (statistical weld / hanging node), closed-form constraint
resolution, 2:1 edge balance with welded byproducts, composite release
splits, and bilinear evaluation. Statistics are stubbed with plain least
squares so the adaptive demo can run; no gates, no tau, no pipeline.

Vertex states (monotone over a run, no coarsening in the PoC):
  hanging -> weld -> free
"""

import numpy as np

MAXL = 12
S = 1 << MAXL  # integer span of the unit square


class Vertex:
    __slots__ = ("ix", "iy", "state", "height", "birth_parents", "depth", "birth_origin")

    def __init__(self, ix, iy, depth, birth_parents=None, birth_origin="root"):
        self.ix, self.iy = ix, iy
        self.state = "weld" if birth_parents else "free"
        self.height = 0.0
        self.birth_parents = birth_parents or []  # [(Vertex, 0.5), (Vertex, 0.5)]
        self.depth = depth
        self.birth_origin = birth_origin

    @property
    def x(self):
        return self.ix / S

    @property
    def y(self):
        return self.iy / S


class Cell:
    __slots__ = ("x0", "y0", "x1", "y1", "lx", "ly", "children", "axis")

    def __init__(self, x0, y0, x1, y1, lx, ly):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.lx, self.ly = lx, ly
        self.children = None
        self.axis = None

    @property
    def leaf(self):
        return self.children is None

    def corner_keys(self):
        return [(self.x0, self.y0), (self.x1, self.y0),
                (self.x0, self.y1), (self.x1, self.y1)]

    def contains(self, ix, iy):
        return self.x0 <= ix <= self.x1 and self.y0 <= iy <= self.y1


class RectMesh:
    def __init__(self, balance=True):
        self.balance = balance
        self.verts = {}
        for ix in (0, S):
            for iy in (0, S):
                self.verts[(ix, iy)] = Vertex(ix, iy, depth=0)
        self.root = Cell(0, 0, S, S, 0, 0)
        self.leaves = [self.root]
        # hanging map: vertex key -> (master_key_a, master_key_b)
        self.hanging = {}

    # ---------------------------------------------------------- topology

    def _get_or_make(self, ix, iy, parents, depth, origin):
        v = self.verts.get((ix, iy))
        if v is None:
            v = Vertex(ix, iy, depth, birth_parents=parents, birth_origin=origin)
            self.verts[(ix, iy)] = v
        return v

    def split(self, cell, axis, origin="selected"):
        """Dyadic bisection. New chord-endpoint vertices are born welded
        (species decided by the hanging recomputation afterwards)."""
        assert cell.leaf
        d = cell.lx + cell.ly + 1
        if axis == "x":
            xm = (cell.x0 + cell.x1) // 2
            pa = [(self.verts[(cell.x0, cell.y0)], 0.5), (self.verts[(cell.x1, cell.y0)], 0.5)]
            pb = [(self.verts[(cell.x0, cell.y1)], 0.5), (self.verts[(cell.x1, cell.y1)], 0.5)]
            self._get_or_make(xm, cell.y0, pa, d, origin)
            self._get_or_make(xm, cell.y1, pb, d, origin)
            a = Cell(cell.x0, cell.y0, xm, cell.y1, cell.lx + 1, cell.ly)
            b = Cell(xm, cell.y0, cell.x1, cell.y1, cell.lx + 1, cell.ly)
        else:
            ym = (cell.y0 + cell.y1) // 2
            pa = [(self.verts[(cell.x0, cell.y0)], 0.5), (self.verts[(cell.x0, cell.y1)], 0.5)]
            pb = [(self.verts[(cell.x1, cell.y0)], 0.5), (self.verts[(cell.x1, cell.y1)], 0.5)]
            self._get_or_make(cell.x0, ym, pa, d, origin)
            self._get_or_make(cell.x1, ym, pb, d, origin)
            a = Cell(cell.x0, cell.y0, cell.x1, ym, cell.lx, cell.ly + 1)
            b = Cell(cell.x0, ym, cell.x1, cell.y1, cell.lx, cell.ly + 1)
        cell.children, cell.axis = (a, b), axis
        self.leaves.remove(cell)
        self.leaves += [a, b]
        if self.balance:
            self._rebalance()
        self._recompute_hanging()
        return a, b

    def _rebalance(self):
        """2:1 edge balance: leaves sharing an edge segment may differ by at
        most one level in the axis transverse to that edge. Balance splits
        are welded byproducts (origin='balance')."""
        changed = True
        while changed:
            changed = False
            for c in list(self.leaves):
                for n in list(self.leaves):
                    if n is c:
                        continue
                    if c.x1 == n.x0 or c.x0 == n.x1:  # vertical shared edge
                        if not (c.y0 < n.y1 and n.y0 < c.y1):
                            continue
                        if n.ly < c.ly - 1:
                            self._plain_split(n, "y")
                            changed = True
                            break
                    if c.y1 == n.y0 or c.y0 == n.y1:  # horizontal shared edge
                        if not (c.x0 < n.x1 and n.x0 < c.x1):
                            continue
                        if n.lx < c.lx - 1:
                            self._plain_split(n, "x")
                            changed = True
                            break
                if changed:
                    break

    def _plain_split(self, cell, axis):
        """split() without recursion into rebalance (called from it)."""
        saved, self.balance = self.balance, False
        try:
            self.split(cell, axis, origin="balance")
        finally:
            self.balance = saved

    def _recompute_hanging(self):
        """A vertex strictly inside a leaf's edge is hanging, constrained to
        that edge's endpoints (masters). Overrides weld state; never
        overrides free (promotion requires regularity first)."""
        self.hanging = {}
        for leaf in self.leaves:
            edges = [((leaf.x0, leaf.y0), (leaf.x1, leaf.y0)),
                     ((leaf.x0, leaf.y1), (leaf.x1, leaf.y1)),
                     ((leaf.x0, leaf.y0), (leaf.x0, leaf.y1)),
                     ((leaf.x1, leaf.y0), (leaf.x1, leaf.y1))]
            for (ax, ay), (bx, by) in edges:
                for (ix, iy), v in self.verts.items():
                    if (ix, iy) == (ax, ay) or (ix, iy) == (bx, by):
                        continue
                    on = (ay == by == iy and ax < ix < bx) or \
                         (ax == bx == ix and ay < iy < by)
                    if on:
                        assert v.state != "free", "free vertex became hanging"
                        v.state = "hanging"
                        self.hanging[(ix, iy)] = ((ax, ay), (bx, by))
        for key, v in self.verts.items():
            if v.state == "hanging" and key not in self.hanging:
                v.state = "weld"  # released by a neighbor split

    # ------------------------------------------------------- constraints

    def masters(self, v):
        """Current constraint pair for a non-free vertex. Hanging weights are
        the linear interpolation parameter along the coarse edge — exactly
        1/2 at the midpoint (the only steady state under 2:1 balance), but
        general positions occur transiently mid-rebalance."""
        if v.state == "hanging":
            a, b = self.hanging[(v.ix, v.iy)]
            (ax, ay), (bx, by) = a, b
            t = ((v.ix - ax) / (bx - ax)) if ax != bx else ((v.iy - ay) / (by - ay))
            return [(self.verts[a], 1.0 - t), (self.verts[b], t)]
        return v.birth_parents

    def closure(self, v):
        """Fully resolved master list {(free vertex, weight)} (GEOMETRY_DESIGN §3)."""
        if v.state == "free":
            return {(v.ix, v.iy): 1.0}
        out = {}
        for m, w in self.masters(v):
            for k, w2 in self.closure(m).items():
                out[k] = out.get(k, 0.0) + w * w2
        return out

    def resolve_heights(self):
        for v in sorted(self.verts.values(), key=lambda v: v.depth):
            if v.state != "free":
                v.height = sum(w * m.height for m, w in self.masters(v))

    def resolve_heights_hier(self, delta):
        """Hierarchical (surplus) parameterization: every vertex sits at its
        parent interpolation plus its own surplus; welds and hanging vertices
        have surplus structurally zero (delta only holds free keys)."""
        for v in sorted(self.verts.values(), key=lambda v: v.depth):
            base = sum(w * m.height for m, w in self.masters(v)) \
                if (v.birth_parents or v.state == "hanging") else 0.0
            v.height = base + delta.get((v.ix, v.iy), 0.0)

    def design_hier(self, xs, ys):
        """Design columns in the hierarchical basis: column of delta_k is the
        surface response to a unit surplus at k, transported through all
        descendants. Promotion adds a column here without moving any other
        (GEOMETRY_DESIGN §2b)."""
        keys = self.free_keys()
        saved = {k: self.verts[k].height for k in self.verts}
        cols = []
        for k in keys:
            self.resolve_heights_hier({k: 1.0})
            cols.append([self.eval(x, y) for x, y in zip(xs, ys)])
        for kk, h in saved.items():
            self.verts[kk].height = h
        return keys, np.array(cols).T

    # ------------------------------------------------------- promotion

    def promote(self, key):
        """Composite move: release splits until regular, then free (§2, §7.3)."""
        n_release = 0
        while self.verts[key].state == "hanging":
            (ax, ay), (bx, by) = self.hanging[key]
            leaf = next(l for l in self.leaves
                        if l.contains(*key) and l.contains(ax, ay) and l.contains(bx, by))
            self.split(leaf, "y" if ay != by else "x", origin="release")
            n_release += 1
        self.verts[key].state = "free"
        return n_release

    # ------------------------------------------------------- evaluation

    def find_leaf(self, x, y):
        ix = min(max(x, 0.0), 1.0) * S
        iy = min(max(y, 0.0), 1.0) * S
        c = self.root
        while not c.leaf:
            a, b = c.children
            c = a if (ix <= a.x1 if c.axis == "x" else iy <= a.y1) else b
        return c

    def eval(self, x, y):
        c = self.find_leaf(x, y)
        u = (x * S - c.x0) / (c.x1 - c.x0)
        w = (y * S - c.y0) / (c.y1 - c.y0)
        h00 = self.verts[(c.x0, c.y0)].height
        h10 = self.verts[(c.x1, c.y0)].height
        h01 = self.verts[(c.x0, c.y1)].height
        h11 = self.verts[(c.x1, c.y1)].height
        return (h00 * (1 - u) * (1 - w) + h10 * u * (1 - w)
                + h01 * (1 - u) * w + h11 * u * w)

    def eval_grid(self, n=96):
        g = (np.arange(n) + 0.5) / n
        self.resolve_heights()
        return np.array([[self.eval(x, y) for x in g] for y in g])

    # ------------------------------------------------------- least squares

    def free_keys(self):
        return [k for k, v in self.verts.items() if v.state == "free"]

    def design(self, xs, ys):
        """Design column per free coefficient via unit-height probes; exact
        through the constraint closure by construction."""
        keys = self.free_keys()
        saved = {k: self.verts[k].height for k in self.verts}
        cols = []
        for k in keys:
            for kk in self.verts:
                self.verts[kk].height = 0.0
            self.verts[k].height = 1.0
            self.resolve_heights()
            cols.append([self.eval(x, y) for x, y in zip(xs, ys)])
        for kk, h in saved.items():
            self.verts[kk].height = h
        return keys, np.array(cols).T

    def fit(self, xs, ys, zs, ridge=1e-8):
        keys, X = self.design(xs, ys)
        A = X.T @ X + ridge * np.eye(len(keys))
        b = X.T @ np.asarray(zs)
        h = np.linalg.solve(A, b)
        for k, hv in zip(keys, h):
            self.verts[k].height = hv
        self.resolve_heights()
        return h
