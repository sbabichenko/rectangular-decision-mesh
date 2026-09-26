"""core/estimator/system.cpp: the live system of the rectangular mesh, in nodal coordinates.

The free coefficients' heights, the padded training design B of their nodal columns (a constrained
vertex carries its parents' average, a free one stops the propagation), and the depth prior on the
surpluses d = R h as the dense penalty R' diag(1/tau^2) R. The same model as the surplus basis
X = B R^-1 with a diagonal prior, with rows a few entries wide instead of tens.
"""
import numpy as np

from pyrect.estimator.padded_design import PaddedDesign

K_ROOT_PRIOR_VARIANCE = 1.0   # logits^2


class LiveSystem:
    __slots__ = ('free', 'children', 'root_count', 'transform', 'X', 'initial', 'precision', 'penalty')


def _order_key(mesh):
    def key(i):
        v = mesh.vertices[i]
        return (v.depth if v.has_prior() else 0, i)
    return key


def _refresh(mesh, system):
    p = len(system.free)
    system.initial = np.array([mesh.vertices[i].height for i in system.free], float)
    precision = np.zeros(p)
    # the four corner heights: sd 1 logit around the baseline (core/estimator/system.cpp)
    precision[:system.root_count] = 1.0 / K_ROOT_PRIOR_VARIANCE
    for c, vid in enumerate(system.children):
        precision[system.root_count + c] = 1.0 / mesh.tau_for_depth(mesh.vertices[vid].depth)
    system.precision = precision
    penalty = np.zeros((p, p))
    for r in range(p):
        w = precision[r]
        if w == 0.0:
            continue
        cols, vals = system.transform[r]
        for a in range(len(cols)):
            for b in range(a, len(cols)):
                penalty[cols[a], cols[b]] += w * vals[a] * vals[b]
    iu = np.triu_indices(p, 1)
    penalty[(iu[1], iu[0])] = penalty[iu]      # the upper triangle is summed; mirror it exactly
    system.penalty = penalty


def live_system(mesh, rows):
    """The live system on the training rows `rows` (ascending), its structure reused while the
    topology, the free set and the rows are unchanged."""
    rows = np.asarray(rows, dtype=np.intp)
    key = (mesh.topology_version, mesh.coefficient_version, len(mesh.vertices), mesh.n_points, rows.tobytes())
    cached = getattr(mesh, '_live_cache', None)
    if cached is not None and cached[0] == key:
        system = cached[1]
        _refresh(mesh, system)
        return system
    system = LiveSystem()
    free = sorted((v._id for v in mesh.vertices if v.free_coefficient), key=_order_key(mesh))
    column = np.full(len(mesh.vertices), -1, dtype=np.intp)
    column[free] = np.arange(len(free))
    system.free = free
    system.children = [i for i in free if mesh.vertices[i].has_prior()]
    system.root_count = len(free) - len(system.children)
    if any(mesh.vertices[i].has_prior() for i in free[:system.root_count]):
        raise RuntimeError('free coefficients are not ordered roots first')
    transform = []
    for r, vid in enumerate(free):
        v = mesh.vertices[vid]
        acc = {r: 1.0}
        if v.has_prior():
            for parent in (v.p0, v.p1):
                for pid, w in mesh.nodal_closure(parent).items():
                    c = int(column[pid])
                    acc[c] = acc.get(c, 0.0) - 0.5 * w
        items = sorted((c, value) for c, value in acc.items() if value != 0.0)
        transform.append((np.array([c for c, _ in items], np.intp), np.array([value for _, value in items])))
    system.transform = transform
    ids, values = mesh.nodal_rows(rows)
    index = np.where(values != 0.0, column[ids], 0)
    # the columns ascending within a row, as the binary sorts them (the free order is not the id order)
    order = np.argsort(np.where(values != 0.0, index, np.iinfo(np.intp).max), axis=1, kind='stable')
    index = np.take_along_axis(index, order, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    system.X = PaddedDesign(index, values, len(free))
    _refresh(mesh, system)
    mesh._live_cache = (key, system)
    return system


def surpluses_from(system, heights):
    """R h: the surpluses of the free coefficients from their heights."""
    out = np.zeros(len(system.transform))
    for r, (cols, vals) in enumerate(system.transform):
        total = 0.0
        for c, v in zip(cols.tolist(), vals.tolist()):
            total += v * heights[c]
        out[r] = total
    return out
