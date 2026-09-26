"""core/mesh/rect.{h,cpp}: the rectangular decision mesh.

Anisotropic dyadic bisection of the unit square (or cube) on a 2^24 integer grid, multilinear
interpolation on the leaf cells, and the hierarchical (surplus) basis the estimator fits
(docs/RECT_ESTIMATOR.md, docs/RECT_3D.md). Every vertex but the roots is born bisecting a dyadic
segment and has two birth parents, the segment's endpoints; its height is their average plus its
surplus, which is nonzero only for a free coefficient. In three dimensions the closure keeps every
hanging vertex on a leaf edge along its own birth axis, and the gate reads a candidate's column
from a dry-run move (see the binary). The row work (placement, weights, predictions, candidate
columns) is numpy; the arithmetic is the binary's, in the binary's order, so the two agree to rounding.
"""
from __future__ import annotations

import numpy as np

K_LEVELS = 24
K_SIDE = 1 << K_LEVELS
K_MINIMUM_FIT_POINTS = 10
K_TAU_FLOOR_LOGIT_SD = 0.005
K_MAX_DIM = 3
_CHECK_COLUMN_CACHE = __import__('os').environ.get('PYRECT_CHECK_COLUMN_CACHE') == '1'


def depth_of_segment(length):
    """The prior class of a vertex bisecting a segment of `length` grid units (1 for a unit segment)."""
    k = int(length).bit_length() - 1
    if (1 << k) != length:
        raise ValueError('segment length is not a power of two')
    return K_LEVELS - k + 1


def _merge_scaled(x, a, y, b):
    """a*x + b*y for two {id: value} maps (the binary's merge: common ids a*x+b*y, dropped when zero)."""
    out = {}
    for i, v in x.items():
        out[i] = a * v
    for i, v in y.items():
        if i in x:
            value = a * x[i] + b * v
            if value != 0.0:
                out[i] = value
            else:
                del out[i]
        else:
            out[i] = b * v
    return dict(sorted(out.items()))


class Vertex:
    __slots__ = ('_id', 'ix', 'iy', 'iz', 'x', 'y', 'z', 'p0', 'p1', 'axis', 'depth', 'free_coefficient', 'gate_admitted',
                 'backbone', 'surplus', 'height', 'children', 'em_second', 'sigma_sq', 'sigma_pooled', 'lambda_v',
                 'current_fit_points', 'current_xTx', 'current_keff',
                 'cand_round', 'cand_beta', 'cand_sd', 'cand_z', 'cand_xTx', 'cand_keff', 'cand_c', 'cand_gain',
                 'cand_signed_root', 'cand_info_null',
                 'admit_round', 'admit_beta', 'admit_sd', 'admit_z', 'admit_c', 'admit_xTx', 'admit_keff',
                 'admit_gain', 'admit_signed_root', 'admit_info_null', 'admit_move', 'admit_splits', 'admit_chord_lo', 'admit_chord_hi')

    def __init__(self, vid, ix, iy, p0, p1, axis, iz=0):
        self._id = vid; self.ix = ix; self.iy = iy; self.iz = iz
        self.x = ix / K_SIDE; self.y = iy / K_SIDE; self.z = iz / K_SIDE
        self.p0 = p0; self.p1 = p1; self.axis = axis; self.depth = 0
        self.free_coefficient = False; self.gate_admitted = False; self.backbone = False
        self.surplus = 0.0; self.height = 0.0; self.children = []
        self.em_second = -1.0; self.sigma_sq = 1e300; self.sigma_pooled = 1e300; self.lambda_v = 0.0
        self.current_fit_points = 0; self.current_xTx = 0.0; self.current_keff = 0.0
        self.cand_round = -1
        self.cand_beta = self.cand_sd = self.cand_z = self.cand_xTx = self.cand_keff = self.cand_c = 0.0
        self.cand_gain = self.cand_signed_root = self.cand_info_null = 0.0
        self.admit_round = -1
        self.admit_beta = self.admit_sd = self.admit_z = self.admit_c = self.admit_xTx = self.admit_keff = 0.0
        self.admit_gain = self.admit_signed_root = self.admit_info_null = 0.0
        self.admit_move = -1; self.admit_splits = 0   # 0 new midpoint, 1 weld promoted, 2 hanging released; cells split
        self.admit_chord_lo = self.admit_chord_hi = -1   # the far ends of the cut through it, when admitted

    def has_prior(self):
        return self.p0 >= 0

    def at(self):
        return (self.ix, self.iy, self.iz)


class Cell:
    """A box: lo and hi per axis (the unused axes 0), and its corners, corner k having bit d set
    when it sits at hi[d] (two dimensions: (x0,y0), (x1,y0), (x0,y1), (x1,y1))."""
    __slots__ = ('lo', 'hi', 'a', 'b', 'axis', 'corner', 'rows')

    def __init__(self, lo, hi, corner, rows):
        self.lo = lo; self.hi = hi
        self.a = -1; self.b = -1; self.axis = -1
        self.corner = corner
        self.rows = rows       # a leaf's rows, ascending (np.intp)

    x0 = property(lambda self: self.lo[0]); x1 = property(lambda self: self.hi[0])
    y0 = property(lambda self: self.lo[1]); y1 = property(lambda self: self.hi[1])

    def leaf(self):
        return self.a < 0


class Candidate:
    """A position the gate may promote: an existing constrained vertex, or the midpoint of a leaf
    edge with no vertex yet (its would-be parents, segment direction and prior class given)."""
    __slots__ = ('ix', 'iy', 'iz', 'vertex', 'p0', 'p1', 'axis', 'depth')

    def __init__(self, ix, iy, vertex=-1, p0=-1, p1=-1, axis=-1, depth=0, iz=0):
        self.ix = ix; self.iy = iy; self.iz = iz; self.vertex = vertex; self.p0 = p0; self.p1 = p1; self.axis = axis; self.depth = depth

    def at(self):
        return (self.ix, self.iy, self.iz)

    def copy(self):
        return Candidate(self.ix, self.iy, self.vertex, self.p0, self.p1, self.axis, self.depth, self.iz)


class _Journal:
    """The undo log of a dry-run move: what split and _make_vertex change in existing structures."""
    __slots__ = ('vertices', 'cells', 'leaf_count', 'topology', 'pending', 'cells_before', 'corners_before',
                 'children_before', 'read_cells', 'read_points')

    def __init__(self, mesh):
        self.vertices = len(mesh.vertices); self.cells = len(mesh.cells)
        self.leaf_count = mesh._leaf_count; self.topology = mesh.topology_version; self.pending = list(mesh._pending)
        self.cells_before = {}; self.corners_before = {}; self.children_before = {}
        # what the dry run's outcome depended on: the pre-existing cells it looked at (they must stay
        # unsplit leaves) and the grid points it found empty (they must stay empty; a vertex that
        # existed stays, since vertices are never removed)
        self.read_cells = set(); self.read_points = set()


class RectMesh:
    def __init__(self, observations, training_mask, dim=2):
        if dim not in (2, 3):
            raise ValueError('the mesh has two or three dimensions')
        self.dim = dim
        self.obs_store = observations
        self.n_points = len(observations)
        self.px = np.clip(np.array([o.x for o in observations], float), 0.0, 1.0)
        self.py = np.clip(np.array([o.y for o in observations], float), 0.0, 1.0)
        self.pz = np.clip(np.array([o.z for o in observations], float), 0.0, 1.0) if dim == 3 else np.empty(0)
        mask = np.asarray(training_mask, dtype=np.uint8)
        self.training_mask = mask if len(mask) == self.n_points else np.ones(self.n_points, np.uint8)
        self.vertices = []
        self.cells = []
        self.topology_version = 0
        self.coefficient_version = 0
        self.refresh_epoch = 0
        self.tau_sq = {}
        self.tau_default_sq = 1.0
        self.tau_floor_sq = K_TAU_FLOOR_LOGIT_SD * K_TAU_FLOOR_LOGIT_SD
        self._index = {}
        self._corner_cells = []
        self._closure = None
        self._closure_version = None
        self._pending = []
        self._journal = None
        K = self.corner_count()
        # the corners in corner order (bit d: at hi[d]), in two dimensions the historical order
        corner = []
        for k in range(K):
            q = [0, 0, 0]
            for d in range(dim):
                q[d] = K_SIDE if (k >> d) & 1 else 0
            corner.append(self._make_vertex(tuple(q), -1, -1, -1))
        corner = tuple(corner)
        hi = [0, 0, 0]
        for d in range(dim):
            hi[d] = K_SIDE
        self.cells.append(Cell((0, 0, 0), tuple(hi), corner, np.arange(self.n_points, dtype=np.intp)))
        # the descent tables leaf_at walks: per cell its children, cut axis and midline (-1: a leaf)
        self._ca = [-1]; self._cb = [-1]; self._cax = [-1]; self._cm = [0]
        # the orthant sign patterns, axis 0 the most significant
        self._orthant_signs = [tuple((1 if (o >> (dim - 1 - d)) & 1 else -1) if d < dim else 1 for d in range(3))
                               for o in range(1 << dim)]
        for c in corner:
            self._corner_cells[c].append(0)
        self._leaf_count = 1
        self.row_cell = np.zeros(self.n_points, dtype=np.intp)
        self.row_weight = np.zeros((self.n_points, K))
        self._place_rows(0)
        self._pending = []
        self._obs_arrays = None

    def corner_count(self):
        return 1 << self.dim

    def coords(self, d):
        return self.px if d == 0 else self.py if d == 1 else self.pz

    # ---- the observations
    def obs_arrays(self):
        """offset, exposure, events of every observation as arrays (cached until the engine drops it)."""
        if getattr(self, "_obs_arrays", None) is None:
            obs = self.obs_store
            self._obs_arrays = (np.array([o.offset for o in obs]), np.array([float(o.exposure) for o in obs]),
                                np.array([float(o.events) for o in obs]))
        return self._obs_arrays

    # ---- the depth prior
    def recompute_tau_sq(self):
        """Per depth, EM: the mean posterior second moment of the depth's gate admissions (see the binary)."""
        by_depth = {}
        target = set()
        for v in self.vertices:
            if not v.has_prior():
                continue
            target.add(v.depth)
            if not v.free_coefficient or not v.gate_admitted:
                continue
            if v.sigma_sq >= 1e300 or v.sigma_sq <= 0:
                continue
            by_depth.setdefault(v.depth, []).append(v)
        previous = dict(self.tau_sq)
        self.tau_sq = {}
        for d in sorted(by_depth):
            verts = by_depth[d]
            if len(verts) < 3:
                continue
            total = 0.0; count = 0
            for v in verts:
                if v.em_second >= 0.0:
                    total += v.em_second; count += 1
            if count >= 3:
                self.tau_sq[d] = max(total / count, self.tau_floor_sq)
            elif d in previous:
                self.tau_sq[d] = previous[d]
        known = sorted(self.tau_sq)
        for d in sorted(target):
            if d in self.tau_sq:
                continue
            if not known:
                self.tau_sq[d] = max(self.tau_default_sq, self.tau_floor_sq)
                continue
            nearest = known[0]
            for d2 in known:
                if abs(d2 - d) < abs(nearest - d):
                    nearest = d2
            self.tau_sq[d] = self.tau_sq[nearest]
        self.tau_sq = dict(sorted(self.tau_sq.items()))

    def tau_for_depth(self, depth):
        if depth in self.tau_sq:
            return max(self.tau_sq[depth], self.tau_floor_sq)
        if not self.tau_sq:
            return max(self.tau_default_sq, self.tau_floor_sq)
        keys = sorted(self.tau_sq)
        hi = next((k for k in keys if k >= depth), None)
        if hi is None:
            return max(self.tau_sq[keys[-1]], self.tau_floor_sq)
        if hi == keys[0]:
            return max(self.tau_sq[hi], self.tau_floor_sq)
        lo = keys[keys.index(hi) - 1]
        return max(self.tau_sq[lo] if depth - lo <= hi - depth else self.tau_sq[hi], self.tau_floor_sq)

    # ---- geometry
    def leaf_count(self):
        return self._leaf_count

    def leaves(self):
        return [c for c, cell in enumerate(self.cells) if cell.leaf()]

    def vertex_at(self, ix, iy, iz=0):
        return self._index.get((ix, iy, iz), -1)

    def _make_vertex(self, p, p0, p1, axis):
        existing = self._lookup(p)
        if existing >= 0:
            v = self.vertices[existing]
            # in three dimensions a split can reach an existing vertex along another axis than its
            # birth's: the vertex is then a corner of both halves and keeps its parents
            if (v.p0 != p0 or v.p1 != p1) and self.dim == 2:
                raise RuntimeError('a vertex was reborn with different parents')
            return existing
        v = Vertex(len(self.vertices), p[0], p[1], p0, p1, axis, p[2])
        if p0 >= 0:
            a, b = self.vertices[p0], self.vertices[p1]
            length = abs(b.at()[axis] - a.at()[axis])
            v.depth = depth_of_segment(length)
            v.height = 0.5 * (a.height + b.height)
            j = self._journal
            if j is not None:
                for q in (p0, p1):
                    if q < j.vertices and q not in j.children_before:
                        j.children_before[q] = len(self.vertices[q].children)
        self.vertices.append(v)
        if p0 >= 0:
            self.vertices[p0].children.append(v._id)
            self.vertices[p1].children.append(v._id)
        self._index[p] = v._id
        self._corner_cells.append([])
        if self.dim == 3:
            self._pending.append(v._id)
        return v._id

    def _is_corner(self, c, p):
        for d in range(self.dim):
            if p[d] != c.lo[d] and p[d] != c.hi[d]:
                return False
        return True

    def _interior_axes(self, c, p):
        return [d for d in range(self.dim) if c.lo[d] < p[d] < c.hi[d]]

    def _batch_weights(self, rows, lo, hi):
        """_compute_weights for rows of different cells at once (lo, hi: each row's cell box)."""
        f = [np.clip((self.coords(d)[rows] * K_SIDE - lo[:, d].astype(float)) / (hi[:, d] - lo[:, d]).astype(float), 0.0, 1.0) for d in range(3)]
        out = np.empty((len(rows), 8))
        for k in range(8):
            weight = f[0] if k & 1 else 1 - f[0]
            for d in (1, 2):
                weight = weight * (f[d] if (k >> d) & 1 else 1 - f[d])
            out[:, k] = weight
        return out

    def _compute_weights(self, c, rows, out):
        f = [np.clip((self.coords(d)[rows] * K_SIDE - float(c.lo[d])) / float(c.hi[d] - c.lo[d]), 0.0, 1.0) for d in range(3)]
        for k in range(8):
            weight = f[0] if k & 1 else 1 - f[0]
            for d in (1, 2):
                weight = weight * (f[d] if (k >> d) & 1 else 1 - f[d])
            out[:, k] = weight

    def _place_rows(self, cell):
        c = self.cells[cell]
        rows = c.rows
        if len(rows) == 0:
            return
        self.row_cell[rows] = cell
        if self.dim == 2:
            w = float(c.hi[0] - c.lo[0]); h = float(c.hi[1] - c.lo[1])
            u = np.clip((self.px[rows] * K_SIDE - float(c.lo[0])) / w, 0.0, 1.0)
            t = np.clip((self.py[rows] * K_SIDE - float(c.lo[1])) / h, 0.0, 1.0)
            self.row_weight[rows, 0] = (1 - u) * (1 - t)
            self.row_weight[rows, 1] = u * (1 - t)
            self.row_weight[rows, 2] = (1 - u) * t
            self.row_weight[rows, 3] = u * t
            return
        w = np.empty((len(rows), 8))
        self._compute_weights(c, rows, w)
        self.row_weight[rows] = w

    def split(self, cell, axis):
        parent = self.cells[cell]
        if not parent.leaf():
            raise RuntimeError('split of a cell that is not a leaf')
        if axis < 0 or axis >= self.dim:
            raise RuntimeError('split along an axis the mesh does not have')
        if parent.hi[axis] - parent.lo[axis] < 2:
            raise RuntimeError('cell too narrow to split' if axis == 0 else 'cell too thin to split')
        j = self._journal
        if j is not None and cell < j.cells and cell not in j.cells_before:
            j.cells_before[cell] = (parent.a, parent.b, parent.axis, parent.rows)
            j.read_cells.add(cell)
        m = (parent.lo[axis] + parent.hi[axis]) // 2
        K = self.corner_count(); bit = 1 << axis
        # the midpoints of the parent's edges along the axis, lower corner k first
        mid = {}
        for k in range(K):
            if k & bit:
                continue
            q = [0, 0, 0]
            for d in range(self.dim):
                q[d] = parent.hi[d] if (k >> d) & 1 else parent.lo[d]
            q[axis] = m
            mid[k] = self._make_vertex(tuple(q), parent.corner[k], parent.corner[k | bit], axis)
        a_hi = list(parent.hi); a_hi[axis] = m
        b_lo = list(parent.lo); b_lo[axis] = m
        a_corner = tuple(mid[k & ~bit] if k & bit else parent.corner[k] for k in range(K))
        b_corner = tuple(parent.corner[k] if k & bit else mid[k] for k in range(K))
        upper = self.coords(axis)[parent.rows] >= m / K_SIDE
        a = Cell(parent.lo, tuple(a_hi), a_corner, parent.rows[~upper])
        b = Cell(tuple(b_lo), parent.hi, b_corner, parent.rows[upper])
        if j is not None:
            for k in range(K):
                for q in (parent.corner[k], mid.get(k & ~bit, -1)):
                    if 0 <= q < j.vertices and q not in j.corners_before:
                        j.corners_before[q] = list(self._corner_cells[q])
        ia = len(self.cells); ib = ia + 1
        self.cells.append(a); self.cells.append(b)
        parent.a = ia; parent.b = ib; parent.axis = axis
        self._ca[cell] = ia; self._cb[cell] = ib; self._cax[cell] = axis; self._cm[cell] = m
        self._ca += (-1, -1); self._cb += (-1, -1); self._cax += (-1, -1); self._cm += (0, 0)
        parent.rows = np.empty(0, dtype=np.intp)
        for v in parent.corner:
            lst = self._corner_cells[v]
            lst[:] = [c for c in lst if c != cell]
        for child in (ia, ib):
            for v in self.cells[child].corner:
                self._corner_cells[v].append(child)
        if j is None:   # a dry run leaves the rows' placement alone; its column weighs what it reads
            self._place_rows(ia)
            self._place_rows(ib)
        self._leaf_count += 1
        self.topology_version += 1

    def chord_ends(self, v):
        """The cut through a vertex: the leaf edges through it across its segment. Their far ends,
        below/left and above/right (-1 where no leaf edge runs that way, or for a root). Two
        dimensions only: in three the cut is a plane (both -1)."""
        out = [-1, -1]
        u = self.vertices[v]
        if u.axis < 0 or self.dim != 2:
            return out
        for c in self._corner_cells[v]:
            cell = self.cells[c]
            lo, hi = (cell.lo[1], cell.hi[1]) if u.axis == 0 else (cell.lo[0], cell.hi[0])
            at = u.iy if u.axis == 0 else u.ix
            side = 1 if at == lo else 0
            far = hi if side else lo
            out[side] = self.vertex_at(u.ix, far) if u.axis == 0 else self.vertex_at(far, u.iy)
        return out

    def leaf_at(self, p, s):
        ca, cb, cax, cm = self._ca, self._cb, self._cax, self._cm
        c = 0
        while ca[c] >= 0:
            d = cax[c]; m = cm[c]; q = p[d]
            c = cb[c] if (q > m or (q == m and s[d] > 0)) else ca[c]
        j = self._journal
        if j is not None and c < j.cells:
            j.read_cells.add(c)
        return c

    def _lookup(self, p):
        """The vertex at a grid point (-1: none); a dry run records the answer it depended on."""
        v = self._index.get(p, -1)
        j = self._journal
        if j is not None and (v < 0 or v >= j.vertices):
            j.read_points.add(p)
        return v

    def _read_cell(self, c):
        j = self._journal
        if j is not None and c < j.cells:
            j.read_cells.add(c)

    def _orthants(self, p):
        """The orthant signs around the point that stay on the grid, axis 0 the most significant."""
        dim = self.dim
        low = [p[d] == 0 for d in range(dim)]; high = [p[d] == K_SIDE for d in range(dim)]
        if not any(low) and not any(high):
            return self._orthant_signs
        return [s for s in self._orthant_signs
                if not any((low[d] and s[d] < 0) or (high[d] and s[d] > 0) for d in range(dim))]

    def make_corner(self, p):
        for s in self._orthants(p):
            while True:
                c = self.leaf_at(p, s)
                L = self.cells[c]
                if self._is_corner(L, p):
                    break
                self.split(c, self._interior_axes(L, p)[0])
        v = self._lookup(p)
        if v < 0:
            raise RuntimeError('make_corner left no vertex')
        if self.dim == 3:
            self.close_pending()
        return v

    def close_pending(self):
        """Three dimensions: split leaves until every vertex created since the last closure is a
        corner of the leaves around it or hangs on a leaf edge along its own birth axis."""
        while self._pending:
            v = self._pending.pop()
            p = self.vertices[v].at(); birth = self.vertices[v].axis
            for s in self._orthants(p):
                while True:
                    c = self.leaf_at(p, s)
                    axes = self._interior_axes(self.cells[c], p)
                    if not axes or (len(axes) == 1 and axes[0] == birth):
                        break
                    along = next((a for a in axes if a != birth), axes[0])
                    self.split(c, along)

    def hanging_invariant_holds(self):
        for v in self.vertices:
            p = v.at()
            for s in self._orthants(p):
                axes = self._interior_axes(self.cells[self.leaf_at(p, s)], p)
                if axes and not (len(axes) == 1 and axes[0] == v.axis):
                    return False
        return True

    def refine_uniform(self, levels):
        levels = [levels] * 3 if isinstance(levels, int) else list(levels)
        top = max(levels[:self.dim])
        for level in range(top):
            for axis in range(self.dim):
                for c in self.leaves():
                    cell = self.cells[c]
                    if cell.hi[axis] - cell.lo[axis] > (K_SIDE >> levels[axis]):
                        self.split(c, axis)
        if self.dim == 3:
            self.close_pending()

    # ---- heights and prediction
    def recompute_heights(self):
        vs = self.vertices
        for v in vs:
            v.height = v.surplus + (0.5 * (vs[v.p0].height + vs[v.p1].height) if v.p0 >= 0 else 0.0)
        self._heights_version = getattr(self, '_heights_version', 0) + 1

    def predict_all_array(self):
        """predict_index for every row: the corner products summed left to right, as the binary does.
        Kept until the topology or a height changes (every height is set by recompute_heights or by a
        split, which moves the topology); returned read-only."""
        key = (self.topology_version, getattr(self, '_heights_version', 0), len(self.vertices))
        cached = getattr(self, '_prediction_cache', None)
        if cached is not None and cached[0] == key:
            return cached[1]
        heights = np.fromiter((v.height for v in self.vertices), float, len(self.vertices))
        corners = np.array([c.corner for c in self.cells], dtype=np.intp)[self.row_cell]
        w = self.row_weight
        out = w[:, 0] * heights[corners[:, 0]]
        for k in range(1, self.corner_count()):
            out = out + w[:, k] * heights[corners[:, k]]
        out.flags.writeable = False
        self._prediction_cache = (key, out)
        return out

    def predict_all(self):
        return self.predict_all_array().tolist()

    def predict(self, x, y, z=0.0):
        p = (x, y, z)
        c = 0
        cells = self.cells
        while not cells[c].leaf():
            n = cells[c]
            m = ((n.lo[n.axis] + n.hi[n.axis]) // 2) / K_SIDE
            c = n.b if p[n.axis] >= m else n.a
        L = cells[c]
        f = [min(max((p[d] * K_SIDE - L.lo[d]) / float(L.hi[d] - L.lo[d]), 0.0), 1.0) for d in range(self.dim)]
        total = 0.0
        for k in range(self.corner_count()):
            weight = f[0] if k & 1 else 1 - f[0]
            for d in range(1, self.dim):
                weight = weight * (f[d] if (k >> d) & 1 else 1 - f[d])
            h = self.vertices[L.corner[k]].height
            total = weight * h if k == 0 else total + weight * h
        return total

    # ---- the nodal design
    def nodal_closure(self, v):
        """A vertex's height in the free coefficients' heights: {id: weight}, ids ascending."""
        if self._closure_version != self.coefficient_version or len(self._closure) != len(self.vertices):
            self._closure = [None] * len(self.vertices)
            self._closure_version = self.coefficient_version
        cl = self._closure
        if cl[v] is not None:
            return cl[v]
        stack = [v]
        while stack:
            u = stack[-1]
            w = self.vertices[u]
            if cl[u] is not None:
                stack.pop(); continue
            if not w.free_coefficient:
                if w.p0 < 0:
                    raise RuntimeError('a root is not free')
                if cl[w.p0] is None or cl[w.p1] is None:
                    if cl[w.p0] is None: stack.append(w.p0)
                    if cl[w.p1] is None: stack.append(w.p1)
                    continue
                cl[u] = _merge_scaled(cl[w.p0], 0.5, cl[w.p1], 0.5)
            else:
                cl[u] = {u: 1.0}
            stack.pop()
        return cl[v]

    def nodal_rows(self, rows):
        """The nodal design rows of the given observations: (ids, values), each (rows x width), the
        free vertex ids ascending within a row and padded with (0, 0.0). A row's value for an id is
        the binary's merge, sum_k w_k closure_k(id) over its cell's corners in corner order; weights
        and closures are nonnegative, so nothing cancels and the zero entries are the absent ones."""
        rows = np.asarray(rows, dtype=np.intp)
        cells_of = self.row_cell[rows]
        order = np.argsort(cells_of, kind='stable')
        sorted_cells = cells_of[order]
        bounds = np.flatnonzero(np.r_[True, sorted_cells[1:] != sorted_cells[:-1], True])
        blocks = []
        width = 1
        K = self.corner_count()
        for b in range(len(bounds) - 1):
            idx = order[bounds[b]:bounds[b + 1]]
            corner = self.cells[sorted_cells[bounds[b]]].corner
            closures = [self.nodal_closure(k) for k in corner]
            ids = sorted(set().union(*closures))
            w = self.row_weight[rows[idx]]
            V = np.zeros((len(idx), len(ids)))
            for j, vid in enumerate(ids):
                col = np.zeros(len(idx))
                for k in range(K):
                    c = closures[k].get(vid)
                    if c is not None:
                        col = col + w[:, k] * c
                V[:, j] = col
            blocks.append((idx, np.asarray(ids, dtype=np.intp), V))
            width = max(width, len(ids))
        index = np.zeros((len(rows), width), dtype=np.intp)
        value = np.zeros((len(rows), width))
        for idx, ids, V in blocks:
            nonzero = V != 0.0
            # the nonzero entries of every row first, in id order (a stable sort on "is zero")
            pos = np.argsort(~nonzero, axis=1, kind='stable')
            count = nonzero.sum(axis=1)
            take_ids = np.where(np.arange(len(ids))[None, :] < count[:, None], ids[pos], 0)
            take_val = np.where(np.arange(len(ids))[None, :] < count[:, None], np.take_along_axis(V, pos, axis=1), 0.0)
            index[idx, :len(ids)] = take_ids
            value[idx, :len(ids)] = take_val
        return index, value

    # ---- the gate's candidates
    def frontier(self):
        out = [Candidate(v.ix, v.iy, v._id, v.p0, v.p1, v.axis, v.depth, v.iz)
               for v in self.vertices if v.has_prior() and not v.free_coefficient]
        seen = set()
        K = self.corner_count()
        for L in self.cells:
            if not L.leaf():
                continue
            # every edge's midpoint, axis by axis, lower corner first; the first leaf to offer a
            # point names its segment
            for axis in range(self.dim):
                bit = 1 << axis
                length = L.hi[axis] - L.lo[axis]
                for k in range(K):
                    if k & bit:
                        continue
                    q = [0, 0, 0]
                    for d in range(self.dim):
                        q[d] = L.hi[d] if (k >> d) & 1 else L.lo[d]
                    q[axis] = (L.lo[axis] + L.hi[axis]) // 2
                    q = tuple(q)
                    if length < 2 or q in self._index or q in seen:
                        continue
                    seen.add(q)
                    out.append(Candidate(q[0], q[1], -1, L.corner[k], L.corner[k | bit], axis, depth_of_segment(length), q[2]))
        out.sort(key=lambda c: (c.ix, c.iy, c.iz))   # stable, as the binary's sort sees no equal keys
        return out

    def refresh(self, candidate):
        c = candidate.copy()
        v = self._lookup(c.at())
        if v >= 0:
            w = self.vertices[v]
            c.vertex, c.p0, c.p1, c.axis, c.depth = v, w.p0, w.p1, w.axis, w.depth
        return c

    def candidate_column(self, candidate):
        """The column the candidate's coefficient would have after its move: (rows, values), rows
        ascending. Read before the move; exact in two dimensions (exact_candidate_column in three)."""
        P = candidate.at()
        v = self._lookup(P)
        K = self.corner_count(); dim = self.dim
        shrunk = []
        for s in self._orthants(P):
            c = self.leaf_at(P, s)
            L = self.cells[c]
            if self._is_corner(L, P):
                continue
            lo = list(L.lo); hi = list(L.hi)
            for d in range(dim):
                while not (P[d] == lo[d] or P[d] == hi[d]):
                    m = (lo[d] + hi[d]) // 2
                    if P[d] > m or (P[d] == m and s[d] > 0): lo[d] = m
                    else: hi[d] = m
            shrunk.append((c, lo, hi))
        rows_out, values_out = [], []
        if v >= 0:
            descendants = [v]
            seen = {v}
            q = 0
            while q < len(descendants):
                for child in self.vertices[descendants[q]].children:
                    if child not in seen:
                        seen.add(child); descendants.append(child)
                q += 1
            descendants.sort()
            influence = {v: 1.0}
            for d in descendants:
                if d == v:
                    continue
                w = self.vertices[d]
                value = 0.0
                if w.p0 in influence: value += 0.5 * influence[w.p0]
                if w.p1 in influence: value += 0.5 * influence[w.p1]
                influence[d] = value
            hat_cells = set()
            for d in descendants:
                hat_cells.update(self._corner_cells[d])
            for c in hat_cells:
                self._read_cell(c)
            for sh in shrunk:
                if sh[0] in hat_cells:
                    raise RuntimeError('a split leaf also carries the hat')
            # every hat cell's rows at once: the corner influences repeated per row, the weights the
            # placed ones or (a dry run's own cells) computed in one pass; each row's arithmetic is
            # the per-cell one
            cells_sorted = sorted(hat_cells)
            cell_rows = [self.cells[c].rows for c in cells_sorted]
            counts = [len(r) for r in cell_rows]
            if sum(counts):
                rows = np.concatenate(cell_rows)
                A = np.repeat(np.array([[influence.get(k, 0.0) for k in self.cells[c].corner] for c in cells_sorted]), counts, axis=0)
                j = self._journal
                fresh = np.repeat(np.array([j is not None and c >= j.cells for c in cells_sorted]), counts)
                w = self.row_weight[rows]
                if fresh.any():
                    lo = np.repeat(np.array([self.cells[c].lo for c in cells_sorted]), counts, axis=0)[fresh]
                    hi = np.repeat(np.array([self.cells[c].hi for c in cells_sorted]), counts, axis=0)[fresh]
                    w[fresh] = self._batch_weights(rows[fresh], lo, hi)
                value = w[:, 0] * A[:, 0]
                for k in range(1, K):
                    value = value + w[:, k] * A[:, k]
                keep = value != 0.0
                rows_out.append(rows[keep]); values_out.append(value[keep])
        shrunk.sort(key=lambda sh: sh[0])
        q = 0
        while q < len(shrunk):
            e = q
            while e < len(shrunk) and shrunk[e][0] == shrunk[q][0]:
                e += 1
            rows = self.cells[shrunk[q][0]].rows
            g = [self.coords(d)[rows] * K_SIDE for d in range(dim)]
            value = np.zeros(len(rows)); taken = np.zeros(len(rows), bool)
            for _, lo, hi in shrunk[q:e]:
                inside = ~taken
                for d in range(dim):
                    inside = inside & (g[d] >= lo[d]) & (g[d] <= hi[d])
                hat = 1.0 - np.abs(g[0][inside] - float(P[0])) / float(hi[0] - lo[0])
                for d in range(1, dim):
                    hat = hat * (1.0 - np.abs(g[d][inside] - float(P[d])) / float(hi[d] - lo[d]))
                value[inside] = hat
                taken |= inside
            keep = taken & (value > 0.0)
            rows_out.append(rows[keep]); values_out.append(value[keep])
            q = e
        if not rows_out:
            return np.empty(0, np.intp), np.empty(0)
        rows = np.concatenate(rows_out); values = np.concatenate(values_out)
        order = np.argsort(rows, kind='stable')
        return rows[order], values[order]

    def realise(self, candidate):
        P = candidate.at()
        if self.dim == 3 and candidate.vertex < 0 and self._lookup(P) < 0:
            # a point on edges along two axes is offered once: split first the leaf whose edge the
            # gate scored, so the vertex has the candidate's parents
            for c in self._corner_cells[candidate.p0]:
                self._read_cell(c)
                if candidate.p1 in self.cells[c].corner:
                    self.split(c, candidate.axis)
                    break
        v = self.make_corner(P)
        if candidate.vertex < 0 and (self.vertices[v].p0 != candidate.p0 or self.vertices[v].p1 != candidate.p1):
            raise RuntimeError("a new vertex was born with parents other than its candidate's")
        return v

    def exact_candidate_column(self, candidate):
        """The candidate's column exactly as its move will leave it: candidate_column in two
        dimensions; in three the move is made, the column read and the move undone.

        Three dimensions keep each dry run's column with what it depended on: the pre-existing cells
        it looked at and the vertex lookups it made. While every such cell is still an unsplit leaf
        and every lookup gives the same answer, a new dry run would repeat the same steps on the same
        data (the rows of those cells, the descendants and their corner cells all live in them), so
        the kept column is returned. Most of the frontier sits in unchanged geometry from round to
        round. PYRECT_CHECK_COLUMN_CACHE=1 recomputes every kept column and requires equality."""
        if self.dim == 2:
            return self.candidate_column(candidate)
        key = (candidate.at(), candidate.vertex, candidate.p0, candidate.p1, candidate.axis)
        cache = self.__dict__.setdefault('_column_cache', {})
        kept = cache.get(key)
        if kept is not None:
            cells, points, column = kept
            ca, index = self._ca, self._index
            if all(ca[c] < 0 for c in cells) and not any(p in index for p in points):
                if _CHECK_COLUMN_CACHE:
                    fresh = self._dry_run_column(candidate)[2]
                    if not (np.array_equal(fresh[0], column[0]) and np.array_equal(fresh[1], column[1])):
                        raise RuntimeError('a kept candidate column differs from a fresh dry run')
                return column[0].astype(np.intp), column[1]
        cells, points, column = self._dry_run_column(candidate)
        # kept compactly: 32-bit rows (a view back to intp is exact)
        cache[key] = (cells, points, (column[0].astype(np.int32), column[1]))
        return column

    def _dry_run_column(self, candidate):
        journal = _Journal(self)
        self._journal = journal
        try:
            self.realise(candidate)
            column = self.candidate_column(self.refresh(candidate))
        finally:
            self._journal = None
            self._rollback(journal)
        return tuple(journal.read_cells), tuple(journal.read_points), column

    def _rollback(self, j):
        for v in self.vertices[j.vertices:]:
            del self._index[v.at()]
        del self.vertices[j.vertices:]
        del self._corner_cells[j.vertices:]
        for v, size in j.children_before.items():
            del self.vertices[v].children[size:]
        for v, lst in j.corners_before.items():
            self._corner_cells[v] = lst
        del self.cells[j.cells:]
        del self._ca[j.cells:]; del self._cb[j.cells:]; del self._cax[j.cells:]; del self._cm[j.cells:]
        for c, (a, b, axis, rows) in j.cells_before.items():
            cell = self.cells[c]
            cell.a = a; cell.b = b; cell.axis = axis; cell.rows = rows
            self._ca[c] = a; self._cb[c] = b; self._cax[c] = axis; self._cm[c] = 0
        # a dry run placed no rows (row_cell and row_weight were never touched)
        self._leaf_count = j.leaf_count; self.topology_version = j.topology; self._pending = j.pending

