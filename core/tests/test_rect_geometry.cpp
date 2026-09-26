// The rectangular geometry's contracts, on random meshes: constrained splits leave the surface
// unchanged; a candidate's column before its move is the column its coefficient has after it;
// the design rows reproduce the predictions; the surface is continuous across hanging edges.
#include "mesh/rect.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;
int premove_differs = 0, moves3 = 0;
void check(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
}

std::vector<Observation> random_points(int n, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<Observation> obs(n);
    for (auto& o : obs) { o.x = u(rng); o.y = u(rng); o.exposure = 10; }
    // a few points on dyadic lines and stacked on one point, the cases a grid makes special
    for (int i = 0; i < n / 20; ++i) { obs[i].x = 0.5; obs[n - 1 - i].y = 0.25; }
    for (int i = 0; i < 8; ++i) { obs[n / 2 + i].x = 0.375; obs[n / 2 + i].y = 0.625; }
    return obs;
}

std::vector<double> predictions(const rect::Mesh& mesh) {
    std::vector<double> out(mesh.n_points);
    for (int i = 0; i < mesh.n_points; ++i) out[i] = mesh.predict_index(i);
    return out;
}

double max_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double worst = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(a[i] - b[i]));
    return worst;
}

void run_trial(unsigned seed) {
    std::mt19937 rng(seed);
    const auto obs = random_points(3000, rng);
    rect::Mesh mesh(obs, std::vector<unsigned char>(obs.size(), 1));
    mesh.refine_uniform(2);
    std::normal_distribution<double> g(0.0, 1.0);
    for (auto& v : mesh.vertices) if (!v.has_prior() || v.depth <= 1) { v.free_coefficient = true; v.surplus = g(rng); }
    ++mesh.coefficient_version;
    mesh.recompute_heights();

    for (int step = 0; step < 60; ++step) {
        // a random candidate from the frontier: its prospective column, then its move
        const auto frontier = mesh.frontier();
        std::uniform_int_distribution<std::size_t> pick(0, frontier.size() - 1);
        const rect::Candidate candidate = frontier[pick(rng)];
        const rect::SparseColumn before = mesh.candidate_column(candidate);
        const auto p0 = predictions(mesh);
        const int id = mesh.realise(candidate);
        mesh.recompute_heights();
        check(max_diff(p0, predictions(mesh)) < 1e-12, "a constrained move changed the surface (seed " + std::to_string(seed) + ")");
        // the realised column: the change in predictions under a unit surplus
        mesh.vertices[id].free_coefficient = true;
        ++mesh.coefficient_version;
        mesh.vertices[id].surplus = 1.0;
        mesh.recompute_heights();
        const auto p1 = predictions(mesh);
        std::vector<double> dense(obs.size(), 0.0);
        for (const auto& [i, value] : before) dense[i] = value;
        double worst = 0.0;
        for (std::size_t i = 0; i < obs.size(); ++i) worst = std::max(worst, std::fabs((p1[i] - p0[i]) - dense[i]));
        check(worst < 1e-12, "a candidate's column differs from its realised column (seed " + std::to_string(seed) + ", step " + std::to_string(step) + ")");
        // the same column read from the vertex's existing record after the move
        const rect::SparseColumn after = mesh.candidate_column(mesh.refresh(candidate));
        std::vector<double> dense_after(obs.size(), 0.0);
        for (const auto& [i, value] : after) dense_after[i] = value;
        check(max_diff(dense, dense_after) < 1e-12, "the column changed across the move");
        mesh.vertices[id].surplus = g(rng) * 0.3;
        mesh.recompute_heights();
        // random constrained refinement elsewhere
        const auto leaves = mesh.leaves();
        std::uniform_int_distribution<std::size_t> leaf(0, leaves.size() - 1);
        const int c = leaves[leaf(rng)];
        const auto p2 = predictions(mesh);
        const int axis = static_cast<int>(rng() % 2);
        const rect::Cell before_cell = mesh.cells[c];
        const std::size_t nv = mesh.vertices.size();
        if ((axis == 0 ? mesh.cells[c].hi[0] - mesh.cells[c].lo[0] : mesh.cells[c].hi[1] - mesh.cells[c].lo[1]) >= 2) mesh.split(c, axis);
        if (!(max_diff(p2, predictions(mesh)) < 1e-12) && failures < 3) {
            std::fprintf(stderr, "split cell [%g,%g]x[%g,%g] axis %d, new vertices %zu\n", double(before_cell.lo[0]) / rect::kSide, double(before_cell.hi[0]) / rect::kSide,
                         double(before_cell.lo[1]) / rect::kSide, double(before_cell.hi[1]) / rect::kSide, axis, mesh.vertices.size() - nv);
            const auto p3 = predictions(mesh);
            for (int i = 0; i < mesh.n_points; ++i) if (std::fabs(p3[i] - p2[i]) > 1e-9) { std::fprintf(stderr, "  row %d at (%g, %g): %g -> %g\n", i, obs[i].x, obs[i].y, p2[i], p3[i]); break; }
            for (int k = 0; k < 4; ++k) { const auto& v = mesh.vertices[before_cell.corner[k]]; std::fprintf(stderr, "  corner %d (%g,%g) h %g free %d\n", v._id, v.x, v.y, v.height, v.free_coefficient); }
            for (std::size_t k = 0; k < mesh.vertices.size(); ++k) { const auto& v = mesh.vertices[k]; if (k >= nv || (std::fabs(v.x - 0) >= 0 && ((axis == 0 && (v.y == before_cell.lo[1] / double(rect::kSide) || v.y == before_cell.hi[1] / double(rect::kSide)) && v.x > before_cell.lo[0] / double(rect::kSide) && v.x < before_cell.hi[0] / double(rect::kSide))))) std::fprintf(stderr, "  edge vertex %d (%g,%g) h %g surplus %g free %d parents %d %d\n", v._id, v.x, v.y, v.height, v.surplus, v.free_coefficient, v.p0, v.p1); }
        }
        check(max_diff(p2, predictions(mesh)) < 1e-12, "a split changed the surface");
    }
    // the design rows reproduce the predictions
    std::vector<std::pair<int, double>> row;
    double worst = 0.0;
    for (int i = 0; i < mesh.n_points; ++i) {
        mesh.design_row(i, row);
        double value = 0.0;
        for (const auto& [id, x] : row) value += x * mesh.vertices[id].surplus;
        worst = std::max(worst, std::fabs(value - mesh.predict_index(i)));
    }
    check(worst < 1e-10, "design rows do not reproduce the predictions");
    // continuity: the surface just either side of every leaf edge
    worst = 0.0;
    const double eps = 1e-9;
    for (const auto& c : mesh.cells) {
        if (!c.leaf()) continue;
        const double x0 = double(c.lo[0]) / rect::kSide, x1 = double(c.hi[0]) / rect::kSide;
        const double y0 = double(c.lo[1]) / rect::kSide, y1 = double(c.hi[1]) / rect::kSide;
        for (double t : {0.1, 0.37, 0.5, 0.83}) {
            const double y = y0 + t * (y1 - y0), x = x0 + t * (x1 - x0);
            if (x0 > 0) worst = std::max(worst, std::fabs(mesh.predict(x0 - eps, y) - mesh.predict(x0 + eps, y)));
            if (y0 > 0) worst = std::max(worst, std::fabs(mesh.predict(x, y0 - eps) - mesh.predict(x, y0 + eps)));
        }
    }
    check(worst < 1e-6, "the surface jumps across a leaf edge: " + std::to_string(worst));
    // predictions at the rows agree with point location
    worst = 0.0;
    for (int i = 0; i < mesh.n_points; ++i) worst = std::max(worst, std::fabs(mesh.predict(obs[i].x, obs[i].y) - mesh.predict_index(i)));
    check(worst < 1e-10, "row predictions disagree with point location");
}

// Three dimensions: the same contracts, plus the closure's invariant (every vertex a corner of the
// leaves around it or hanging on an edge along its birth axis) and continuity across leaf faces.
void run_trial3(unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<Observation> obs(4000);
    for (auto& o : obs) { o.x = u(rng); o.y = u(rng); o.z = u(rng); o.exposure = 10; }
    for (int i = 0; i < 200; ++i) { obs[i].x = 0.5; obs[i + 200].y = 0.25; obs[i + 400].z = 0.5; }
    for (int i = 0; i < 8; ++i) { obs[2000 + i].x = 0.375; obs[2000 + i].y = 0.625; obs[2000 + i].z = 0.75; }
    rect::Mesh mesh(obs, std::vector<unsigned char>(obs.size(), 1), 3);
    mesh.refine_uniform(std::array<int, rect::kMaxDim>{2, 2, 1});
    check(mesh.hanging_invariant_holds(), "3-D: the pre-refined mesh breaks the hanging invariant");
    std::normal_distribution<double> g(0.0, 1.0);
    for (auto& v : mesh.vertices) if (!v.has_prior() || v.depth <= 1) { v.free_coefficient = true; v.surplus = g(rng); }
    ++mesh.coefficient_version;
    mesh.recompute_heights();
    const std::string tag = " (3-D seed " + std::to_string(seed);
    for (int step = 0; step < 50; ++step) {
        const auto frontier = mesh.frontier();
        std::uniform_int_distribution<std::size_t> pick(0, frontier.size() - 1);
        const rect::Candidate candidate = frontier[pick(rng)];
        const std::size_t nv = mesh.vertices.size(), nc = mesh.cells.size(), topo = mesh.topology_version;
        const auto p0 = predictions(mesh);
        const rect::SparseColumn premove0 = mesh.candidate_column(candidate);
        const rect::SparseColumn before = mesh.exact_candidate_column(candidate);
        check(mesh.candidate_column(candidate) == premove0, "the full dry run left a trace" + tag + ")");
        ++moves3; if (premove0 != before) ++premove_differs;   // how often the pre-move reading would have been wrong
        // the dry run leaves nothing behind
        check(mesh.vertices.size() == nv && mesh.cells.size() == nc && mesh.topology_version == topo && max_diff(p0, predictions(mesh)) == 0.0,
              "the dry-run move was not undone" + tag + ")");
        const int id = mesh.realise(candidate);
        mesh.recompute_heights();
        check(max_diff(p0, predictions(mesh)) < 1e-12, "a constrained move changed the surface" + tag + ")");
        check(mesh.hanging_invariant_holds(), "a move broke the hanging invariant" + tag + ", step " + std::to_string(step) + ")");
        mesh.vertices[id].free_coefficient = true;
        ++mesh.coefficient_version;
        mesh.vertices[id].surplus = 1.0;
        mesh.recompute_heights();
        const auto p1 = predictions(mesh);
        std::vector<double> dense(obs.size(), 0.0);
        for (const auto& [i, value] : before) dense[i] = value;
        double worst = 0.0;
        for (std::size_t i = 0; i < obs.size(); ++i) worst = std::max(worst, std::fabs((p1[i] - p0[i]) - dense[i]));
        check(worst < 1e-12, "a candidate's column differs from its realised column" + tag + ", step " + std::to_string(step) + "): " + std::to_string(worst));
        const rect::SparseColumn after = mesh.candidate_column(mesh.refresh(candidate));
        std::vector<double> dense_after(obs.size(), 0.0);
        for (const auto& [i, value] : after) dense_after[i] = value;
        check(max_diff(dense, dense_after) < 1e-12, "the column changed across the move" + tag + ")");
        mesh.vertices[id].surplus = g(rng) * 0.3;
        mesh.recompute_heights();
        // random constrained refinement elsewhere, closed
        const auto leaves = mesh.leaves();
        std::uniform_int_distribution<std::size_t> leaf(0, leaves.size() - 1);
        const int c = leaves[leaf(rng)];
        const int axis = static_cast<int>(rng() % 3);
        const auto p2 = predictions(mesh);
        if (mesh.cells[c].hi[axis] - mesh.cells[c].lo[axis] >= 2) { mesh.split(c, axis); mesh.close_pending(); }
        check(max_diff(p2, predictions(mesh)) < 1e-12, "a split changed the surface" + tag + ")");
        check(mesh.hanging_invariant_holds(), "a split broke the hanging invariant" + tag + ")");
    }
    std::vector<std::pair<int, double>> row;
    double worst = 0.0;
    for (int i = 0; i < mesh.n_points; ++i) {
        mesh.design_row(i, row);
        double value = 0.0;
        for (const auto& [vid, x] : row) value += x * mesh.vertices[vid].surplus;
        worst = std::max(worst, std::fabs(value - mesh.predict_index(i)));
    }
    check(worst < 1e-10, "3-D design rows do not reproduce the predictions");
    // continuity: the surface just either side of every leaf face, at interior points of the face
    worst = 0.0;
    const double eps = 1e-9;
    for (const auto& c : mesh.cells) {
        if (!c.leaf()) continue;
        std::array<double, 3> lo{}, hi{};
        for (int d = 0; d < 3; ++d) { lo[d] = double(c.lo[d]) / rect::kSide; hi[d] = double(c.hi[d]) / rect::kSide; }
        for (int d = 0; d < 3; ++d) {
            if (lo[d] <= 0) continue;
            const int a = (d + 1) % 3, b = (d + 2) % 3;
            for (double s : {0.1, 0.37, 0.5, 0.83})
                for (double t : {0.13, 0.5, 0.71}) {
                    std::array<double, 3> q{};
                    q[a] = lo[a] + s * (hi[a] - lo[a]); q[b] = lo[b] + t * (hi[b] - lo[b]);
                    auto below = q, above = q;
                    below[d] = lo[d] - eps; above[d] = lo[d] + eps;
                    worst = std::max(worst, std::fabs(mesh.predict(below) - mesh.predict(above)));
                }
        }
    }
    check(worst < 1e-6, "the 3-D surface jumps across a leaf face: " + std::to_string(worst) + tag + ")");
    worst = 0.0;
    for (int i = 0; i < mesh.n_points; ++i) worst = std::max(worst, std::fabs(mesh.predict({obs[i].x, obs[i].y, obs[i].z}) - mesh.predict_index(i)));
    check(worst < 1e-10, "3-D row predictions disagree with point location");
}

}  // namespace

int main() {
    for (unsigned seed = 1; seed <= 12; ++seed) run_trial(seed);
    for (unsigned seed = 1; seed <= 12; ++seed) run_trial3(seed);
    if (failures) { std::fprintf(stderr, "%d failures\n", failures); return 1; }
    std::printf("rect geometry: all checks passed (3-D moves: the pre-move reading was wrong in %d of %d)\n", premove_differs, moves3);
    return 0;
}
