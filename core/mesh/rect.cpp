#include "mesh/rect.h"
#include "fit/units.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_set>



namespace rect {

namespace {
constexpr double kTauFloorLogitSd = 0.005;

// a * x + b * y merged by id, both sorted by id; zero entries dropped
void merge_scaled(const std::vector<std::pair<int, double>>& x, double a,
                  const std::vector<std::pair<int, double>>& y, double b,
                  std::vector<std::pair<int, double>>& out) {
    out.clear();
    out.reserve(x.size() + y.size());
    std::size_t i = 0, j = 0;
    while (i < x.size() || j < y.size()) {
        if (j == y.size() || (i < x.size() && x[i].first < y[j].first)) { out.emplace_back(x[i].first, a * x[i].second); ++i; }
        else if (i == x.size() || y[j].first < x[i].first) { out.emplace_back(y[j].first, b * y[j].second); ++j; }
        else { const double v = a * x[i].second + b * y[j].second; if (v != 0.0) out.emplace_back(x[i].first, v); ++i; ++j; }
    }
}

int log2_exact(int64_t v) {
    int k = 0;
    while ((int64_t(1) << k) < v) ++k;
    if ((int64_t(1) << k) != v) throw std::logic_error("segment length is not a power of two");
    return k;
}
}  // namespace

int depth_of_segment(int64_t length) { return kLevels - log2_exact(length) + 1; }


Mesh::Mesh(const std::vector<Observation>& observations, const std::vector<unsigned char>& mask, int dimensions)
    : obs_store(&observations), n_points(static_cast<int>(observations.size())), dim(dimensions), training_mask(mask) {
    if (dim != 2 && dim != 3) throw std::invalid_argument("the mesh has two or three dimensions");
    tau_floor_sq = kVarianceScale * kTauFloorLogitSd * kTauFloorLogitSd;
    px.resize(n_points); py.resize(n_points);
    if (dim == 3) pz.resize(n_points);
    for (int i = 0; i < n_points; ++i) {
        px[i] = std::clamp(observations[i].x, 0.0, 1.0);
        py[i] = std::clamp(observations[i].y, 0.0, 1.0);
        if (dim == 3) pz[i] = std::clamp(observations[i].z, 0.0, 1.0);
    }
    if (training_mask.size() != static_cast<std::size_t>(n_points)) training_mask.assign(n_points, 1);
    Cell root;
    for (int d = 0; d < dim; ++d) root.hi[d] = kSide;
    // the corners in corner order (bit d: at hi[d]), which in two dimensions is the historical
    // (0,0), (1,0), (0,1), (1,1)
    for (int k = 0; k < corner_count(); ++k) {
        Point q{};
        for (int d = 0; d < dim; ++d) q[d] = (k >> d) & 1 ? kSide : 0;
        root.corner[k] = make_vertex(q, -1, -1, -1);
    }
    root.rows.resize(n_points);
    for (int i = 0; i < n_points; ++i) root.rows[i] = i;
    cells.push_back(std::move(root));
    for (int k = 0; k < corner_count(); ++k) corner_cells_[cells[0].corner[k]].push_back(0);
    leaf_count_ = 1;
    row_cell_.assign(n_points, 0);
    row_weight_.assign(n_points, {});
    place_rows(0);
    pending_.clear();
}

Mesh::Mesh(const Mesh& o, int)
    : obs_store(o.obs_store), n_points(o.n_points), dim(o.dim), training_mask(o.training_mask),
      vertices(o.vertices), cells(o.cells), topology_version(o.topology_version), coefficient_version(o.coefficient_version),
      refresh_epoch(o.refresh_epoch), tau_sq(o.tau_sq), tau_default_sq(o.tau_default_sq), tau_floor_sq(o.tau_floor_sq),
      leaf_count_(o.leaf_count_), vertex_index_(o.vertex_index_), corner_cells_(o.corner_cells_),
      coords_{o.coords_[0], o.coords_[1], o.coords_[2]}, weights_(o.weights_), pending_(o.pending_) {}

std::unique_ptr<Mesh> Mesh::clone() const { return std::unique_ptr<Mesh>(new Mesh(*this, 0)); }

std::vector<int> Mesh::leaves() const {
    std::vector<int> out;
    out.reserve(leaf_count_);
    for (int c = 0; c < static_cast<int>(cells.size()); ++c) if (cells[c].leaf()) out.push_back(c);
    return out;
}

int Mesh::vertex_at(const Point& p) const {
    auto it = vertex_index_.find(p);
    return it == vertex_index_.end() ? -1 : it->second;
}

int Mesh::make_vertex(const Point& p, int p0, int p1, int axis) {
    if (const int existing = vertex_at(p); existing >= 0) {
        const Vertex& v = vertices[existing];
        // In three dimensions a split can reach an existing vertex along another axis than its
        // birth's; the vertex is then a corner of both halves and keeps its parents.
        if ((v.p0 != p0 || v.p1 != p1) && dim == 2) throw std::logic_error("a vertex was reborn with different parents");
        return existing;
    }
    Vertex v;
    v._id = static_cast<int>(vertices.size());
    v.ix = p[0]; v.iy = p[1]; v.iz = p[2];
    v.x = static_cast<double>(p[0]) / kSide; v.y = static_cast<double>(p[1]) / kSide; v.z = static_cast<double>(p[2]) / kSide;
    v.p0 = p0; v.p1 = p1; v.axis = axis;
    if (p0 >= 0) {
        const Vertex& a = vertices[p0];
        const Vertex& b = vertices[p1];
        const int64_t length = std::llabs(b.at()[axis] - a.at()[axis]);
        v.depth = depth_of_segment(length);
        v.height = 0.5 * (a.height + b.height);   // born constrained: the surface there, so a split is invisible
    }
    if (journal_ && p0 >= 0)
        for (int q : {p0, p1})
            if (q < static_cast<int>(journal_->vertices) && journal_->children_seen.insert(q).second)
                journal_->children_before.emplace_back(q, vertices[q].children.size());
    vertices.push_back(v);
    if (p0 >= 0) { vertices[p0].children.push_back(v._id); vertices[p1].children.push_back(v._id); }
    vertex_index_[p] = v._id;
    corner_cells_.emplace_back();
    if (dim == 3) pending_.push_back(v._id);
    return v._id;
}

bool Mesh::is_corner(const Cell& c, const Point& p) const {
    for (int d = 0; d < dim; ++d) if (p[d] != c.lo[d] && p[d] != c.hi[d]) return false;
    return true;
}

int Mesh::interior_axes(const Cell& c, const Point& p, std::array<int, kMaxDim>& axes) const {
    int count = 0;
    for (int d = 0; d < dim; ++d) if (c.lo[d] < p[d] && p[d] < c.hi[d]) axes[count++] = d;
    return count;
}

void Mesh::place_rows(int cell) {
    const Cell& c = cells[cell];
    for (int i : c.rows) {
        row_cell_[i] = cell;
        weights_in(c, i, row_weight_[i]);
    }
}

void Mesh::weights_in(const Cell& c, int i, std::array<double, 8>& w) const {
    // per axis the fraction across the cell, and each corner's weight the product of its axes'
    // factors in axis order (two dimensions: (1-u)(1-t), u(1-t), (1-u)t, ut)
    std::array<double, kMaxDim> f{};
    for (int d = 0; d < dim; ++d)
        f[d] = std::clamp((coord(i, d) * kSide - static_cast<double>(c.lo[d])) / static_cast<double>(c.hi[d] - c.lo[d]), 0.0, 1.0);
    for (int k = 0; k < corner_count(); ++k) {
        double weight = (k & 1) ? f[0] : 1 - f[0];
        for (int d = 1; d < dim; ++d) weight = weight * ((k >> d) & 1 ? f[d] : 1 - f[d]);
        w[k] = weight;
    }
}

void Mesh::split(int cell, int axis) {
    if (!cells[cell].leaf()) throw std::logic_error("split of a cell that is not a leaf");
    if (axis < 0 || axis >= dim) throw std::logic_error("split along an axis the mesh does not have");
    if (cells[cell].hi[axis] - cells[cell].lo[axis] < 2) throw std::logic_error(axis == 0 ? "cell too narrow to split" : "cell too thin to split");
    // the parent's rows are moved out, not copied (they are read once, then the parent keeps none);
    // the rest of the parent is a small copy, as cells grows below
    std::vector<int> parent_rows = std::move(cells[cell].rows);
    const Cell parent = cells[cell];
    const bool journaled = journal_ && cell < static_cast<int>(journal_->cells) && journal_->cells_seen.insert(cell).second;
    const int64_t m = (parent.lo[axis] + parent.hi[axis]) / 2;
    const int K = corner_count(), bit = 1 << axis;
    // the midpoints of the parent's edges along the axis, lower corner k first (two dimensions,
    // along x: bottom then top; along y: left then right)
    std::array<int, 8> mid{};
    for (int k = 0; k < K; ++k) {
        if (k & bit) continue;
        Point q{};
        for (int d = 0; d < dim; ++d) q[d] = (k >> d) & 1 ? parent.hi[d] : parent.lo[d];
        q[axis] = m;
        mid[k] = make_vertex(q, parent.corner[k], parent.corner[k | bit], axis);
    }
    Cell a, b;
    a.lo = b.lo = parent.lo; a.hi = b.hi = parent.hi;
    a.hi[axis] = m; b.lo[axis] = m;
    for (int k = 0; k < K; ++k) {
        a.corner[k] = (k & bit) ? mid[k & ~bit] : parent.corner[k];
        b.corner[k] = (k & bit) ? parent.corner[k] : mid[k];
    }
    const double cut = static_cast<double>(m) / kSide;
    for (int i : parent_rows) (coord(i, axis) >= cut ? b.rows : a.rows).push_back(i);
    if (journaled) { journal_->cells_before.emplace_back(cell, parent); journal_->cells_before.back().second.rows = std::move(parent_rows); }
    const int ia = static_cast<int>(cells.size());
    const int ib = ia + 1;
    cells.push_back(std::move(a));
    cells.push_back(std::move(b));
    Cell& p = cells[cell];
    p.a = ia; p.b = ib; p.axis = axis;
    std::vector<int>().swap(p.rows);
    if (journal_)
        for (int k = 0; k < K; ++k)
            for (int q : {parent.corner[k], mid[k & ~bit]})
                if (q < static_cast<int>(journal_->vertices) && journal_->corners_seen.insert(q).second)
                    journal_->corners_before.emplace_back(q, corner_cells_[q]);
    for (int k = 0; k < K; ++k) {
        auto& list = corner_cells_[parent.corner[k]];
        list.erase(std::remove(list.begin(), list.end(), cell), list.end());
    }
    for (int child : {ia, ib})
        for (int k = 0; k < K; ++k) corner_cells_[cells[child].corner[k]].push_back(child);
    if (!journal_) {   // a dry run leaves the rows' placement alone; its column weighs the rows it reads
        place_rows(ia);
        place_rows(ib);
    }
    ++leaf_count_;
    ++topology_version;
}

int Mesh::leaf_at(const Point& p, const std::array<int, kMaxDim>& s) const {
    int c = 0;
    while (!cells[c].leaf()) {
        const Cell& n = cells[c];
        const int d = n.axis;
        const int64_t m = (n.lo[d] + n.hi[d]) / 2;
        c = (p[d] > m || (p[d] == m && s[d] > 0)) ? n.b : n.a;
    }
    return c;
}

std::array<int, kMaxDim> Mesh::orthant(int o) const {
    std::array<int, kMaxDim> s{1, 1, 1};
    for (int d = 0; d < dim; ++d) s[d] = (o >> (dim - 1 - d)) & 1 ? 1 : -1;
    return s;
}

bool Mesh::outside(const Point& p, const std::array<int, kMaxDim>& s) const {
    for (int d = 0; d < dim; ++d)
        if ((p[d] == 0 && s[d] < 0) || (p[d] == kSide && s[d] > 0)) return true;
    return false;
}

int Mesh::make_corner(const Point& p) {
    for (int o = 0; o < corner_count(); ++o) {
        const auto s = orthant(o);
        if (outside(p, s)) continue;
        while (true) {
            const int c = leaf_at(p, s);
            const Cell& L = cells[c];
            if (is_corner(L, p)) break;
            std::array<int, kMaxDim> axes{};
            interior_axes(L, p, axes);
            split(c, axes[0]);
        }
    }
    const int v = vertex_at(p);
    if (v < 0) throw std::logic_error("make_corner left no vertex");
    if (dim == 3) close_pending();
    return v;
}

void Mesh::close_pending() {
    // Every leaf holding a pending vertex on its boundary but not as a corner must hold it on an
    // edge along the vertex's birth axis: split it otherwise, along an interior axis other than
    // the birth axis where there is one. The splits are inert and their new vertices join the queue.
    while (!pending_.empty()) {
        const int v = pending_.back();
        pending_.pop_back();
        const Point p = vertices[v].at();
        const int birth = vertices[v].axis;
        for (int o = 0; o < corner_count(); ++o) {
            const auto s = orthant(o);
            if (outside(p, s)) continue;
            while (true) {
                const int c = leaf_at(p, s);
                std::array<int, kMaxDim> axes{};
                const int n = interior_axes(cells[c], p, axes);
                if (n == 0 || (n == 1 && axes[0] == birth)) break;
                int along = axes[0];
                for (int q = 0; q < n; ++q) if (axes[q] != birth) { along = axes[q]; break; }
                split(c, along);
            }
        }
    }
}

bool Mesh::hanging_invariant_holds() const {
    for (const Vertex& v : vertices) {
        const Point p = v.at();
        for (int o = 0; o < corner_count(); ++o) {
            const auto s = orthant(o);
            if (outside(p, s)) continue;
            std::array<int, kMaxDim> axes{};
            const int n = interior_axes(cells[leaf_at(p, s)], p, axes);
            if (n == 0 || (n == 1 && axes[0] == v.axis)) continue;
            return false;
        }
    }
    return true;
}

void Mesh::refine_uniform(const std::array<int, kMaxDim>& levels) {
    int top = 0;
    for (int d = 0; d < dim; ++d) top = std::max(top, levels[d]);
    for (int level = 0; level < top; ++level)
        for (int axis = 0; axis < dim; ++axis)
            for (int c : leaves())
                if (cells[c].hi[axis] - cells[c].lo[axis] > (kSide >> levels[axis])) split(c, axis);
    if (dim == 3) close_pending();
}

void Mesh::recompute_heights() {
    for (Vertex& v : vertices)
        v.height = v.surplus + (v.p0 >= 0 ? 0.5 * (vertices[v.p0].height + vertices[v.p1].height) : 0.0);
}

double Mesh::predict_index(int i) const {
    const Cell& c = cells[row_cell_[i]];
    const auto& w = row_weight_[i];
    double total = w[0] * vertices[c.corner[0]].height;
    for (int k = 1; k < corner_count(); ++k) total = total + w[k] * vertices[c.corner[k]].height;
    return total;
}

double Mesh::predict(const std::array<double, kMaxDim>& p) const {
    int c = 0;
    while (!cells[c].leaf()) {
        const Cell& n = cells[c];
        const double m = static_cast<double>((n.lo[n.axis] + n.hi[n.axis]) / 2) / kSide;
        c = (p[n.axis] >= m) ? n.b : n.a;
    }
    const Cell& L = cells[c];
    std::array<double, kMaxDim> f{};
    for (int d = 0; d < dim; ++d)
        f[d] = std::clamp((p[d] * kSide - L.lo[d]) / static_cast<double>(L.hi[d] - L.lo[d]), 0.0, 1.0);
    double total = 0.0;
    for (int k = 0; k < corner_count(); ++k) {
        double weight = (k & 1) ? f[0] : 1 - f[0];
        for (int d = 1; d < dim; ++d) weight = weight * ((k >> d) & 1 ? f[d] : 1 - f[d]);
        total = k == 0 ? weight * vertices[L.corner[0]].height : total + weight * vertices[L.corner[k]].height;
    }
    return total;
}

const std::vector<std::pair<int, double>>& Mesh::free_ancestors(int v) {
    if (ancestors_version_ != coefficient_version || ancestors_.size() != vertices.size()) {
        ancestors_.assign(vertices.size(), {});
        ancestors_done_.assign(vertices.size(), 0);
        ancestors_version_ = coefficient_version;
    }
    if (ancestors_done_[v]) return ancestors_[v];
    // parents first (ids are a topological order), iteratively to bound the stack
    std::vector<int> stack{v};
    while (!stack.empty()) {
        const int u = stack.back();
        const Vertex& w = vertices[u];
        if (ancestors_done_[u]) { stack.pop_back(); continue; }
        if (w.p0 >= 0 && (!ancestors_done_[w.p0] || !ancestors_done_[w.p1])) {
            if (!ancestors_done_[w.p0]) stack.push_back(w.p0);
            if (!ancestors_done_[w.p1]) stack.push_back(w.p1);
            continue;
        }
        std::vector<std::pair<int, double>> own;
        if (w.p0 >= 0) merge_scaled(ancestors_[w.p0], 0.5, ancestors_[w.p1], 0.5, own);
        if (w.free_coefficient) {
            // the vertex's own id is larger than every ancestor's
            own.emplace_back(u, 1.0);
        }
        ancestors_[u] = std::move(own);
        ancestors_done_[u] = 1;
        stack.pop_back();
    }
    return ancestors_[v];
}

void Mesh::design_row(int i, std::vector<std::pair<int, double>>& out) {
    const Cell& c = cells[row_cell_[i]];
    const auto w = row_weight_[i];
    std::vector<std::pair<int, double>> acc, next;
    for (int k = 0; k < corner_count(); ++k) {
        if (w[k] == 0.0) continue;
        const auto& anc = free_ancestors(c.corner[k]);
        merge_scaled(acc, 1.0, anc, w[k], next);
        acc.swap(next);
    }
    out.swap(acc);
}

const std::vector<std::pair<int, double>>& Mesh::nodal_closure(int v) {
    if (closure_version_ != coefficient_version || closure_.size() != vertices.size()) {
        closure_.assign(vertices.size(), {});
        closure_done_.assign(vertices.size(), 0);
        closure_version_ = coefficient_version;
    }
    if (closure_done_[v]) return closure_[v];
    std::vector<int> stack{v};
    while (!stack.empty()) {
        const int u = stack.back();
        const Vertex& w = vertices[u];
        if (closure_done_[u]) { stack.pop_back(); continue; }
        if (!w.free_coefficient) {
            if (w.p0 < 0) throw std::logic_error("a root is not free");
            if (!closure_done_[w.p0] || !closure_done_[w.p1]) {
                if (!closure_done_[w.p0]) stack.push_back(w.p0);
                if (!closure_done_[w.p1]) stack.push_back(w.p1);
                continue;
            }
            merge_scaled(closure_[w.p0], 0.5, closure_[w.p1], 0.5, closure_[u]);
        } else {
            closure_[u] = {{u, 1.0}};
        }
        closure_done_[u] = 1;
        stack.pop_back();
    }
    return closure_[v];
}

void Mesh::nodal_row(int i, std::vector<std::pair<int, double>>& out) {
    const Cell& c = cells[row_cell_[i]];
    const auto w = row_weight_[i];
    std::vector<std::pair<int, double>> acc, next;
    for (int k = 0; k < corner_count(); ++k) {
        if (w[k] == 0.0) continue;
        merge_scaled(acc, 1.0, nodal_closure(c.corner[k]), w[k], next);
        acc.swap(next);
    }
    out.swap(acc);
}

std::vector<Candidate> Mesh::frontier() const {
    std::vector<Candidate> out;
    for (const Vertex& v : vertices) {
        if (!v.has_prior() || v.free_coefficient) continue;
        Candidate c;
        c.ix = v.ix; c.iy = v.iy; c.iz = v.iz; c.vertex = v._id; c.p0 = v.p0; c.p1 = v.p1; c.axis = v.axis; c.depth = v.depth;
        out.push_back(c);
    }
    std::unordered_set<Point, PointHash> seen;
    const int K = corner_count();
    for (int ci = 0; ci < static_cast<int>(cells.size()); ++ci) {
        const Cell& L = cells[ci];
        if (!L.leaf()) continue;
        // every edge's midpoint, axis by axis, lower corner first (two dimensions: the bottom and
        // top edges, then the left and right); the first leaf to offer a point names its segment
        for (int axis = 0; axis < dim; ++axis) {
            const int bit = 1 << axis;
            const int64_t length = L.hi[axis] - L.lo[axis];
            for (int k = 0; k < K; ++k) {
                if (k & bit) continue;
                Point q{};
                for (int d = 0; d < dim; ++d) q[d] = (k >> d) & 1 ? L.hi[d] : L.lo[d];
                q[axis] = (L.lo[axis] + L.hi[axis]) / 2;
                if (length < 2 || vertex_at(q) >= 0 || !seen.insert(q).second) continue;
                Candidate c;
                c.ix = q[0]; c.iy = q[1]; c.iz = q[2];
                c.p0 = L.corner[k]; c.p1 = L.corner[k | bit]; c.axis = axis; c.depth = depth_of_segment(length);
                out.push_back(c);
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) {
        if (a.ix != b.ix) return a.ix < b.ix;
        if (a.iy != b.iy) return a.iy < b.iy;
        return a.iz < b.iz;
    });
    return out;
}

Candidate Mesh::refresh(const Candidate& candidate) const {
    Candidate c = candidate;
    const int v = vertex_at(Point{c.ix, c.iy, c.iz});
    if (v >= 0) {
        const Vertex& w = vertices[v];
        c.vertex = v; c.p0 = w.p0; c.p1 = w.p1; c.axis = w.axis; c.depth = w.depth;
    }
    return c;
}

SparseColumn Mesh::candidate_column(const Candidate& candidate) const {
    const Point P{candidate.ix, candidate.iy, candidate.iz};
    const int v = vertex_at(P);
    const int K = corner_count();
    // the orthants whose leaf does not have the point as a corner: the move splits them, and the
    // coefficient's hat there is the plain multilinear hat on the orthant's shrunk cell
    struct Shrunk { int leaf; Point lo, hi; };
    std::vector<Shrunk> shrunk;
    for (int o = 0; o < K; ++o) {
        const auto s = orthant(o);
        if (outside(P, s)) continue;
        const int c = leaf_at(P, s);
        const Cell& L = cells[c];
        if (is_corner(L, P)) continue;
        Point lo = L.lo, hi = L.hi;
        for (int d = 0; d < dim; ++d)
            while (!(P[d] == lo[d] || P[d] == hi[d])) {
                const int64_t m = (lo[d] + hi[d]) / 2;
                if (P[d] > m || (P[d] == m && s[d] > 0)) lo[d] = m; else hi[d] = m;
            }
        shrunk.push_back({c, lo, hi});
    }
    SparseColumn column;
    // the hierarchical hat through the vertex's descendants on the current leaves
    if (v >= 0) {
        std::vector<int> descendants{v};
        std::unordered_set<int> seen{v};
        for (std::size_t q = 0; q < descendants.size(); ++q)
            for (int child : vertices[descendants[q]].children)
                if (seen.insert(child).second) descendants.push_back(child);
        std::sort(descendants.begin(), descendants.end());
        std::unordered_map<int, double> influence;
        influence[v] = 1.0;
        for (int d : descendants) {
            if (d == v) continue;
            const Vertex& w = vertices[d];
            double value = 0.0;
            if (auto it = influence.find(w.p0); it != influence.end()) value += 0.5 * it->second;
            if (auto it = influence.find(w.p1); it != influence.end()) value += 0.5 * it->second;
            influence[d] = value;
        }
        std::set<int> hat_cells;
        for (int d : descendants)
            for (int c : corner_cells_[d]) hat_cells.insert(c);
        for (const Shrunk& s : shrunk)
            if (hat_cells.count(s.leaf)) throw std::logic_error("a split leaf also carries the hat");
        for (int c : hat_cells) {
            const Cell& Kc = cells[c];
            double at[8];
            for (int k = 0; k < K; ++k) {
                auto it = influence.find(Kc.corner[k]);
                at[k] = it == influence.end() ? 0.0 : it->second;
            }
            // a cell a dry run made has its rows unplaced: weigh them here, as place_rows would
            const bool fresh = journal_ && c >= static_cast<int>(journal_->cells);
            std::array<double, 8> computed{};
            for (int i : Kc.rows) {
                if (fresh) weights_in(Kc, i, computed);
                const auto& w = fresh ? computed : (*weights_)[i];
                double value = w[0] * at[0];
                for (int k = 1; k < K; ++k) value = value + w[k] * at[k];
                if (value != 0.0) column.emplace_back(i, value);
            }
        }
    }
    // the shrunk orthant cells: the plain multilinear hat, one at the point and zero at the other corners
    std::sort(shrunk.begin(), shrunk.end(), [](const Shrunk& a, const Shrunk& b) { return a.leaf < b.leaf; });
    for (std::size_t q = 0; q < shrunk.size();) {
        std::size_t e = q;
        while (e < shrunk.size() && shrunk[e].leaf == shrunk[q].leaf) ++e;
        for (int i : cells[shrunk[q].leaf].rows) {
            std::array<double, kMaxDim> g{};
            for (int d = 0; d < dim; ++d) g[d] = coord(i, d) * kSide;
            for (std::size_t s = q; s < e; ++s) {
                const Shrunk& Q = shrunk[s];
                bool inside = true;
                for (int d = 0; d < dim; ++d) if (g[d] < Q.lo[d] || g[d] > Q.hi[d]) { inside = false; break; }
                if (!inside) continue;
                double value = 1.0 - std::fabs(g[0] - static_cast<double>(P[0])) / static_cast<double>(Q.hi[0] - Q.lo[0]);
                for (int d = 1; d < dim; ++d)
                    value = value * (1.0 - std::fabs(g[d] - static_cast<double>(P[d])) / static_cast<double>(Q.hi[d] - Q.lo[d]));
                if (value > 0.0) column.emplace_back(i, value);
                break;
            }
        }
        q = e;
    }
    std::sort(column.begin(), column.end());
    return column;
}

std::array<int, 2> Mesh::chord_ends(int v) const {
    std::array<int, 2> out{-1, -1};
    const Vertex& u = vertices[v];
    if (u.axis < 0 || dim != 2) return out;
    for (int c : corner_cells_[v]) {
        const Cell& cell = cells[c];
        // a segment along x is cut at x = u.ix, from y0 to y1 of the cell; along y, at y = u.iy
        const int across = u.axis == 0 ? 1 : 0;
        const int64_t lo = cell.lo[across], hi = cell.hi[across];
        const int64_t at = across == 1 ? u.iy : u.ix;
        const int side = at == lo ? 1 : 0;
        const int64_t far = side ? hi : lo;
        out[side] = u.axis == 0 ? vertex_at(u.ix, far) : vertex_at(far, u.iy);
    }
    return out;
}

int Mesh::realise(const Candidate& candidate) {
    const Point P{candidate.ix, candidate.iy, candidate.iz};
    if (dim == 3 && candidate.vertex < 0 && vertex_at(P) < 0) {
        // a point on edges along two axes (of leaves on either side of a face) is offered once;
        // split first the leaf whose edge the gate scored, so the vertex has the candidate's parents
        for (int c : corner_cells_[candidate.p0]) {
            const Cell& L = cells[c];
            bool has_p1 = false;
            for (int k = 0; k < corner_count(); ++k) has_p1 |= L.corner[k] == candidate.p1;
            if (!has_p1) continue;
            split(c, candidate.axis);
            break;
        }
    }
    const int v = make_corner(P);
    if (candidate.vertex < 0 && (vertices[v].p0 != candidate.p0 || vertices[v].p1 != candidate.p1))
        throw std::logic_error("a new vertex was born with parents other than its candidate's");
    return v;
}

SparseColumn Mesh::exact_candidate_column(const Candidate& candidate) {
    if (dim == 2) return candidate_column(candidate);
    Journal journal;
    journal.vertices = vertices.size(); journal.cells = cells.size();
    journal.leaf_count = leaf_count_; journal.topology = topology_version; journal.pending = pending_;
    journal_ = &journal;
    SparseColumn column;
    try {
        realise(candidate);
        column = candidate_column(refresh(candidate));
    } catch (...) { journal_ = nullptr; rollback(journal); throw; }
    journal_ = nullptr;
    rollback(journal);
    return column;
}

void Mesh::rollback(Journal& journal) {
    for (std::size_t v = journal.vertices; v < vertices.size(); ++v) vertex_index_.erase(vertices[v].at());
    vertices.resize(journal.vertices);
    corner_cells_.resize(journal.vertices);
    for (auto& [v, size] : journal.children_before) vertices[v].children.resize(size);
    for (auto& [v, list] : journal.corners_before) corner_cells_[v] = std::move(list);
    cells.resize(journal.cells);
    for (auto& [c, cell] : journal.cells_before) cells[c] = std::move(cell);
    // a dry run placed no rows, so the rows' cells and weights are as they were
    leaf_count_ = journal.leaf_count; topology_version = journal.topology; pending_ = journal.pending;
}

void Mesh::recompute_tau_sq() {
    // Per depth, EM: the mean posterior second moment (surplus squared plus posterior variance) of
    // the depth's gate admissions, stored by the engine after every fit. With fewer than three
    // moments a depth keeps its previous value; depths without one borrow the nearest estimate,
    // and before the first discoveries every depth has the weak default.
    std::map<int, std::vector<const Vertex*>> by_depth;
    std::set<int> target_depths;
    for (const Vertex& v : vertices) {
        if (!v.has_prior()) continue;
        target_depths.insert(v.depth);
        if (!v.free_coefficient || !v.gate_admitted) continue;
        if (v.sigma_sq >= 1e300 || v.sigma_sq <= 0) continue;
        by_depth[v.depth].push_back(&v);
    }
    const std::map<int, double> previous = tau_sq;
    tau_sq.clear();
    for (auto& [d, verts] : by_depth) {
        if (verts.size() < 3) continue;
        double sum = 0.0; int count = 0;
        for (const Vertex* v : verts) if (v->em_second >= 0.0) { sum += v->em_second; ++count; }
        if (count >= 3) tau_sq[d] = std::max(sum / count, tau_floor_sq);
        else if (auto it = previous.find(d); it != previous.end()) tau_sq[d] = it->second;
    }
    std::vector<int> known;
    for (auto& [d, _] : tau_sq) known.push_back(d);
    for (int d : target_depths) {
        if (tau_sq.count(d)) continue;
        if (known.empty()) { tau_sq[d] = std::max(tau_default_sq, tau_floor_sq); continue; }
        int nearest = known[0];
        for (int d2 : known) if (std::abs(d2 - d) < std::abs(nearest - d)) nearest = d2;
        tau_sq[d] = tau_sq[nearest];
    }
}

double Mesh::tau_for_depth(int depth) const {
    auto it = tau_sq.find(depth);
    if (it != tau_sq.end()) return std::max(it->second, tau_floor_sq);
    if (tau_sq.empty()) return std::max(tau_default_sq, tau_floor_sq);
    auto hi = tau_sq.lower_bound(depth);
    if (hi == tau_sq.begin()) return std::max(hi->second, tau_floor_sq);
    if (hi == tau_sq.end()) return std::max(std::prev(hi)->second, tau_floor_sq);
    auto lo = std::prev(hi);
    return std::max((depth - lo->first <= hi->first - depth) ? lo->second : hi->second, tau_floor_sq);
}

}  // namespace rect
