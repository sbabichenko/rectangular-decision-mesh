#include "rect_mesh.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <stdexcept>

namespace rectmesh {

RectMesh::RectMesh() {
    for (int ix : {0, kS})
        for (int iy : {0, kS}) {
            auto v = std::make_unique<Vertex>();
            v->ix = ix;
            v->iy = iy;
            v->state = VState::Free;
            v->depth = 0;
            v->origin = Origin::Root;
            verts_[key_of(ix, iy)] = v.get();
            vert_store_.push_back(std::move(v));
        }
    auto c = std::make_unique<Cell>();
    c->x0 = 0; c->y0 = 0; c->x1 = kS; c->y1 = kS; c->lx = 0; c->ly = 0;
    root_ = c.get();
    cell_store_.push_back(std::move(c));
    leaves_.push_back(root_);
}

Vertex* RectMesh::get_or_make(int ix, int iy,
                              std::vector<std::pair<Vertex*, double>> parents,
                              int depth, Origin origin) {
    auto it = verts_.find(key_of(ix, iy));
    if (it != verts_.end()) return it->second;
    auto v = std::make_unique<Vertex>();
    v->ix = ix;
    v->iy = iy;
    v->state = VState::Weld;
    v->birth_parents = std::move(parents);
    v->depth = depth;
    v->origin = origin;
    Vertex* raw = v.get();
    verts_[key_of(ix, iy)] = raw;
    vert_store_.push_back(std::move(v));
    return raw;
}

void RectMesh::plain_split(Cell* cell, char axis, Origin origin) {
    assert(cell->leaf());
    const int d = cell->lx + cell->ly + 1;
    auto va = [&](int ix, int iy) { return verts_.at(key_of(ix, iy)); };
    Cell *a, *b;
    auto ca = std::make_unique<Cell>();
    auto cb = std::make_unique<Cell>();
    a = ca.get();
    b = cb.get();
    if (axis == 'x') {
        const int xm = (cell->x0 + cell->x1) / 2;
        get_or_make(xm, cell->y0,
                    {{va(cell->x0, cell->y0), 0.5}, {va(cell->x1, cell->y0), 0.5}},
                    d, origin);
        get_or_make(xm, cell->y1,
                    {{va(cell->x0, cell->y1), 0.5}, {va(cell->x1, cell->y1), 0.5}},
                    d, origin);
        *a = {cell->x0, cell->y0, xm, cell->y1, cell->lx + 1, cell->ly};
        *b = {xm, cell->y0, cell->x1, cell->y1, cell->lx + 1, cell->ly};
    } else {
        const int ym = (cell->y0 + cell->y1) / 2;
        get_or_make(cell->x0, ym,
                    {{va(cell->x0, cell->y0), 0.5}, {va(cell->x0, cell->y1), 0.5}},
                    d, origin);
        get_or_make(cell->x1, ym,
                    {{va(cell->x1, cell->y0), 0.5}, {va(cell->x1, cell->y1), 0.5}},
                    d, origin);
        *a = {cell->x0, cell->y0, cell->x1, ym, cell->lx, cell->ly + 1};
        *b = {cell->x0, ym, cell->x1, cell->y1, cell->lx, cell->ly + 1};
    }
    cell->child_a = a;
    cell->child_b = b;
    cell->axis = axis;
    cell_store_.push_back(std::move(ca));
    cell_store_.push_back(std::move(cb));
    for (size_t i = 0; i < leaves_.size(); ++i)
        if (leaves_[i] == cell) {
            leaves_[i] = leaves_.back();
            leaves_.pop_back();
            break;
        }
    leaves_.push_back(a);
    leaves_.push_back(b);
    ++topo_version_;
}

void RectMesh::split(Cell* cell, char axis, Origin origin) {
    plain_split(cell, axis, origin);
    if (balance_enabled_) rebalance();
    recompute_hanging();
}

void RectMesh::rebalance() {
    // 2:1 edge balance: leaves sharing an edge segment differ by at most
    // one level in the axis transverse to that edge. Balance splits are
    // welded byproducts.
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < leaves_.size() && !changed; ++i) {
            Cell* c = leaves_[i];
            for (size_t j = 0; j < leaves_.size() && !changed; ++j) {
                Cell* n = leaves_[j];
                if (n == c) continue;
                const bool vshare = (c->x1 == n->x0 || c->x0 == n->x1) &&
                                    (c->y0 < n->y1 && n->y0 < c->y1);
                if (vshare && n->ly < c->ly - 1) {
                    plain_split(n, 'y', Origin::Balance);
                    changed = true;
                    break;
                }
                const bool hshare = (c->y1 == n->y0 || c->y0 == n->y1) &&
                                    (c->x0 < n->x1 && n->x0 < c->x1);
                if (hshare && n->lx < c->lx - 1) {
                    plain_split(n, 'x', Origin::Balance);
                    changed = true;
                    break;
                }
            }
        }
    }
}

void RectMesh::recompute_hanging() {
    hanging_.clear();
    for (Cell* leaf : leaves_) {
        const int ex[4][4] = {
            {leaf->x0, leaf->y0, leaf->x1, leaf->y0},
            {leaf->x0, leaf->y1, leaf->x1, leaf->y1},
            {leaf->x0, leaf->y0, leaf->x0, leaf->y1},
            {leaf->x1, leaf->y0, leaf->x1, leaf->y1}};
        for (const auto& e : ex) {
            const int ax = e[0], ay = e[1], bx = e[2], by = e[3];
            for (auto& kv : verts_) {
                Vertex* v = kv.second;
                if ((v->ix == ax && v->iy == ay) ||
                    (v->ix == bx && v->iy == by))
                    continue;
                const bool on =
                    (ay == by && v->iy == ay && ax < v->ix && v->ix < bx) ||
                    (ax == bx && v->ix == ax && ay < v->iy && v->iy < by);
                if (on) {
                    assert(v->state != VState::Free &&
                           "free vertex became hanging");
                    v->state = VState::Hanging;
                    hanging_[kv.first] = {key_of(ax, ay), key_of(bx, by)};
                }
            }
        }
    }
    for (auto& kv : verts_) {
        Vertex* v = kv.second;
        if (v->state == VState::Hanging && !hanging_.count(kv.first))
            v->state = VState::Weld;  // released by a neighbor split
    }
    ++topo_version_;
}

std::vector<std::pair<Vertex*, double>> RectMesh::masters(
    const Vertex* v) const {
    if (v->state == VState::Hanging) {
        const auto& ab = hanging_.at(key_of(v->ix, v->iy));
        Vertex* a = verts_.at(ab.first);
        Vertex* b = verts_.at(ab.second);
        double t;
        if (a->ix != b->ix)
            t = static_cast<double>(v->ix - a->ix) / (b->ix - a->ix);
        else
            t = static_cast<double>(v->iy - a->iy) / (b->iy - a->iy);
        return {{a, 1.0 - t}, {b, t}};
    }
    return v->birth_parents;
}

int RectMesh::promote(VKey key) {
    int n_release = 0;
    while (verts_.at(key)->state == VState::Hanging) {
        const auto& ab = hanging_.at(key);
        const int ax = key_ix(ab.first), ay = key_iy(ab.first);
        const int bx = key_ix(ab.second), by = key_iy(ab.second);
        Cell* target = nullptr;
        for (Cell* l : leaves_)
            if (l->contains(key_ix(key), key_iy(key)) && l->contains(ax, ay) &&
                l->contains(bx, by)) {
                target = l;
                break;
            }
        if (!target) throw std::runtime_error("promote: no constraining leaf");
        split(target, ay != by ? 'y' : 'x', Origin::Release);
        ++n_release;
    }
    verts_.at(key)->state = VState::Free;
    return n_release;
}

std::vector<VKey> RectMesh::free_keys() const {
    std::vector<VKey> out;
    for (auto& kv : verts_)
        if (kv.second->state == VState::Free) out.push_back(kv.first);
    return out;
}

void RectMesh::resolve_heights_hier(
    const std::unordered_map<VKey, double>& delta) {
    // depth order: masters always have strictly lower depth
    std::vector<Vertex*> order;
    order.reserve(verts_.size());
    for (auto& kv : verts_) order.push_back(kv.second);
    std::sort(order.begin(), order.end(),
              [](const Vertex* a, const Vertex* b) { return a->depth < b->depth; });
    for (Vertex* v : order) {
        double base = 0.0;
        if (!v->birth_parents.empty() || v->state == VState::Hanging)
            for (auto& mw : masters(v)) base += mw.second * mw.first->height;
        auto it = delta.find(key_of(v->ix, v->iy));
        v->height = base + (it == delta.end() ? 0.0 : it->second);
    }
}

const Cell* RectMesh::find_leaf(double x, double y) const {
    double ix = x * kS, iy = y * kS;
    if (ix < 0) ix = 0;
    if (ix > kS) ix = kS;
    if (iy < 0) iy = 0;
    if (iy > kS) iy = kS;
    const Cell* c = root_;
    while (!c->leaf())
        c = ((c->axis == 'x') ? (ix <= c->child_a->x1) : (iy <= c->child_a->y1))
                ? c->child_a
                : c->child_b;
    return c;
}

double RectMesh::eval(double x, double y) const {
    const Cell* c = find_leaf(x, y);
    const double u = (x * kS - c->x0) / (c->x1 - c->x0);
    const double w = (y * kS - c->y0) / (c->y1 - c->y0);
    const double h00 = verts_.at(key_of(c->x0, c->y0))->height;
    const double h10 = verts_.at(key_of(c->x1, c->y0))->height;
    const double h01 = verts_.at(key_of(c->x0, c->y1))->height;
    const double h11 = verts_.at(key_of(c->x1, c->y1))->height;
    return h00 * (1 - u) * (1 - w) + h10 * u * (1 - w) + h01 * (1 - u) * w +
           h11 * u * w;
}

void RectMesh::closure_of(Vertex* v, std::unordered_map<VKey, double>* acc,
                          double w) {
    if (v->state == VState::Free) (*acc)[key_of(v->ix, v->iy)] += w;
    if (!v->birth_parents.empty() || v->state == VState::Hanging)
        for (auto& mw : masters(v)) closure_of(mw.first, acc, w * mw.second);
}

const std::vector<std::pair<VKey, double>>& RectMesh::closure(Vertex* v) {
    if (closure_version_ != topo_version_) {
        closure_memo_.clear();
        closure_version_ = topo_version_;
    }
    auto it = closure_memo_.find(v);
    if (it != closure_memo_.end()) return it->second;
    std::unordered_map<VKey, double> acc;
    closure_of(v, &acc, 1.0);
    std::vector<std::pair<VKey, double>> out(acc.begin(), acc.end());
    return closure_memo_[v] = std::move(out);
}

void RectMesh::design_row(double x, double y,
                          const std::unordered_map<VKey, int>& key_index,
                          std::vector<std::pair<int, double>>* out) {
    out->clear();
    const Cell* c = find_leaf(x, y);
    const double u = (x * kS - c->x0) / (c->x1 - c->x0);
    const double w = (y * kS - c->y0) / (c->y1 - c->y0);
    const std::pair<VKey, double> corners[4] = {
        {key_of(c->x0, c->y0), (1 - u) * (1 - w)},
        {key_of(c->x1, c->y0), u * (1 - w)},
        {key_of(c->x0, c->y1), (1 - u) * w},
        {key_of(c->x1, c->y1), u * w}};
    std::unordered_map<int, double> acc;
    for (const auto& cw : corners) {
        if (cw.second == 0.0) continue;
        for (const auto& fw : closure(verts_.at(cw.first)))
            acc[key_index.at(fw.first)] += cw.second * fw.second;
    }
    out->assign(acc.begin(), acc.end());
}

void RectMesh::probe_column(VKey v0, const std::vector<double>& xs,
                            const std::vector<double>& ys,
                            std::vector<double>* out) {
    // influence I(v; v0) = [v==v0] + sum_m w * I(m; v0), memoized locally
    std::unordered_map<const Vertex*, double> infl;
    // depth order guarantees masters computed first
    std::vector<Vertex*> order;
    order.reserve(verts_.size());
    for (auto& kv : verts_) order.push_back(kv.second);
    std::sort(order.begin(), order.end(),
              [](const Vertex* a, const Vertex* b) { return a->depth < b->depth; });
    for (Vertex* v : order) {
        double s = (key_of(v->ix, v->iy) == v0) ? 1.0 : 0.0;
        if (!v->birth_parents.empty() || v->state == VState::Hanging)
            for (auto& mw : masters(v)) s += mw.second * infl[mw.first];
        infl[v] = s;
    }
    out->resize(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        const Cell* c = find_leaf(xs[i], ys[i]);
        const double u = (xs[i] * kS - c->x0) / (c->x1 - c->x0);
        const double w = (ys[i] * kS - c->y0) / (c->y1 - c->y0);
        (*out)[i] = infl[verts_.at(key_of(c->x0, c->y0))] * (1 - u) * (1 - w) +
                    infl[verts_.at(key_of(c->x1, c->y0))] * u * (1 - w) +
                    infl[verts_.at(key_of(c->x0, c->y1))] * (1 - u) * w +
                    infl[verts_.at(key_of(c->x1, c->y1))] * u * w;
    }
}

}  // namespace rectmesh
