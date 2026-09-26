#include "estimator/system.h"
#include "estimator/profile.h"

#include "mesh/rect.h"

#include <algorithm>
#include <map>
#include <stdexcept>

namespace estimator {

namespace {

constexpr double kRootPriorVariance = 1.0;   // logits^2

std::vector<int64_t> live_key(const rect::Mesh& mesh, bool heldout_split, const std::vector<int64_t>& rows) {
    std::vector<int64_t> key;
    key.push_back(static_cast<int64_t>(mesh.topology_version));
    key.push_back(static_cast<int64_t>(mesh.coefficient_version));
    key.push_back(static_cast<int64_t>(mesh.vertices.size()));
    key.push_back(heldout_split ? 1 : 0);
    key.push_back(static_cast<int64_t>(mesh.n_points));
    key.push_back(static_cast<int64_t>(rows.size()));
    key.insert(key.end(), rows.begin(), rows.end());
    return key;
}

// initial heights, the precisions at the current tau, and the penalty R' diag(precision) R
void refresh_live(const rect::Mesh& mesh, LiveSystem& system) {
    const std::size_t p = system.free.size();
    system.initial.resize(p);
    for (std::size_t j = 0; j < p; ++j) system.initial[j] = mesh.vertices[system.free[j]].height;
    system.precision.assign(p, 0.0);
    // The four corner heights: sd 1 logit around the baseline. The triangular engine leaves its roots
    // unpenalised; here that let an empty corner extrapolate to +-1.5 logits, and when the newest
    // cohorts (the youngest ages) are held out, the rectangular fit extrapolated 7% worse than with
    // this prior (the cohort split over 2020: -7.1%, se 3.0%, better in 9 of 12 months), which puts
    // it level with the triangular engine. Neutral on the random-split battery (-0.03%, se 0.05%).
    for (std::size_t r = 0; r < system.root_count; ++r) system.precision[r] = 1.0 / kRootPriorVariance;
    for (std::size_t c = 0; c < system.children.size(); ++c)
        system.precision[system.root_count + c] = 1.0 / mesh.tau_for_depth(mesh.vertices[system.children[c]].depth);
    system.penalty.assign(p * p, 0.0);
    for (std::size_t r = 0; r < p; ++r) {
        const double w = system.precision[r];
        if (w == 0.0) continue;
        const auto& e = system.transform[r].entries;
        for (std::size_t a = 0; a < e.size(); ++a)
            for (std::size_t b = a; b < e.size(); ++b)
                system.penalty[static_cast<std::size_t>(e[a].first) * p + static_cast<std::size_t>(e[b].first)] += w * e[a].second * e[b].second;
    }
    for (std::size_t i = 0; i < p; ++i)          // the upper triangle is summed; mirror it exactly
        for (std::size_t j = i + 1; j < p; ++j) system.penalty[j * p + i] = system.penalty[i * p + j];
}

}  // namespace

LiveSystem live_system(rect::Mesh& mesh, bool heldout_split, const std::vector<int64_t>& rows, LiveCache* cache) {
    ScopeTimer timer("live_system");
    std::vector<int64_t> key;
    if (cache != nullptr) {
        key = live_key(mesh, heldout_split, rows);
        if (cache->have && cache->key == key) {
            LiveSystem system = cache->system;
            refresh_live(mesh, system);
            return system;
        }
    }
    LiveSystem system;
    for (const rect::Vertex& v : mesh.vertices)
        if (v.free_coefficient) system.free.push_back(v._id);
    std::stable_sort(system.free.begin(), system.free.end(), [&](int a, int b) {
        const rect::Vertex& u = mesh.vertices[a];
        const rect::Vertex& w = mesh.vertices[b];
        const int du = u.has_prior() ? u.depth : 0, dw = w.has_prior() ? w.depth : 0;
        return du != dw ? du < dw : a < b;
    });
    std::vector<int> column(mesh.vertices.size(), -1);
    for (std::size_t j = 0; j < system.free.size(); ++j) {
        const int id = system.free[j];
        column[id] = static_cast<int>(j);
        if (mesh.vertices[id].has_prior()) system.children.push_back(id);
        else ++system.root_count;
    }
    for (std::size_t j = 0; j < system.root_count; ++j)
        if (mesh.vertices[system.free[j]].has_prior()) throw std::logic_error("free coefficients are not ordered roots first");
    const std::size_t p = system.free.size();
    // the transform: a root's surplus is its height; a child's is its height less its parents'
    // average, the parents read through their nodal closures
    system.transform.resize(p);
    for (std::size_t r = 0; r < p; ++r) {
        const rect::Vertex& v = mesh.vertices[system.free[r]];
        std::map<int64_t, double> acc;
        acc[static_cast<int64_t>(r)] += 1.0;
        if (v.has_prior())
            for (int parent : {v.p0, v.p1})
                for (const auto& [id, w] : mesh.nodal_closure(parent)) acc[column[id]] -= 0.5 * w;
        for (const auto& [c, value] : acc) if (value != 0.0) system.transform[r].entries.emplace_back(c, value);
    }
    std::vector<std::vector<std::pair<int64_t, double>>> B_rows(rows.size());
    std::vector<std::pair<int, double>> design;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        mesh.nodal_row(static_cast<int>(rows[r]), design);
        auto& out = B_rows[r];
        out.reserve(design.size());
        for (const auto& [id, value] : design) {
            if (column[id] < 0) throw std::logic_error("a design row reads a vertex that is not free");
            out.emplace_back(column[id], value);
        }
        std::sort(out.begin(), out.end());
    }
    system.X = std::make_shared<const PaddedDesign>(PaddedDesign::from_sparse_rows(B_rows, p));
    refresh_live(mesh, system);
    if (cache != nullptr) { cache->have = true; cache->key = std::move(key); cache->system = system; }
    return system;
}

std::vector<double> surpluses_from(const LiveSystem& system, const std::vector<double>& heights) {
    std::vector<double> out(system.transform.size(), 0.0);
    for (std::size_t r = 0; r < out.size(); ++r)
        for (const auto& [c, v] : system.transform[r].entries) out[r] += v * heights[static_cast<std::size_t>(c)];
    return out;
}

}  // namespace estimator
