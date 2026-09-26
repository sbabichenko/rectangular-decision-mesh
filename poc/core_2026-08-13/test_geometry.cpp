// Geometry invariants, ported from the Python PoC's verified demos:
//  1. Inert-split lemma: welded splits are invisible in function space.
//  2. Closure-based design rows equal probe columns exactly.
//  3. Composite release split adds exactly one hierarchical column.
#include <cmath>
#include <cstdio>
#include <random>

#include "rect_mesh.h"

using namespace rectmesh;

static int failures = 0;
#define CHECK(cond, msg)                                        \
    do {                                                        \
        if (!(cond)) {                                          \
            std::printf("FAIL: %s\n", msg);                     \
            ++failures;                                         \
        }                                                       \
    } while (0)

static double grid_max_diff(RectMesh& m,
                            const std::vector<double>& ref) {
    double mx = 0.0;
    int idx = 0;
    for (int iy = 0; iy < 96; ++iy)
        for (int ix = 0; ix < 96; ++ix) {
            const double v = m.eval((ix + 0.5) / 96.0, (iy + 0.5) / 96.0);
            mx = std::max(mx, std::fabs(v - ref[idx++]));
        }
    return mx;
}

static std::vector<double> grid_eval(RectMesh& m) {
    std::vector<double> out;
    out.reserve(96 * 96);
    for (int iy = 0; iy < 96; ++iy)
        for (int ix = 0; ix < 96; ++ix)
            out.push_back(m.eval((ix + 0.5) / 96.0, (iy + 0.5) / 96.0));
    return out;
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    // ---- 1. inert-split lemma
    {
        RectMesh m;
        for (int i = 0; i < 10; ++i) {
            auto leaves = m.leaf_cells();
            Cell* leaf = leaves[rng() % leaves.size()];
            m.split(leaf, (rng() % 2) ? 'x' : 'y', Origin::Selected);
            for (auto& kv : m.vertices())
                if (kv.second->state == VState::Weld && unif(rng) < 0.6)
                    m.promote(kv.first);
        }
        std::unordered_map<VKey, double> delta;
        for (VKey k : m.free_keys()) delta[k] = gauss(rng);
        m.resolve_heights_hier(delta);
        auto ref = grid_eval(m);
        for (int i = 0; i < 6; ++i) {
            auto leaves = m.leaf_cells();
            m.split(leaves[rng() % leaves.size()], (rng() % 2) ? 'x' : 'y',
                    Origin::Selected);
        }
        m.resolve_heights_hier(delta);
        const double d = grid_max_diff(m, ref);
        std::printf("lemma: max surface change after welded splits = %.3e\n", d);
        CHECK(d < 1e-12, "inert-split lemma");
    }

    // ---- 2. closure design rows == probe columns
    {
        RectMesh m;
        for (int i = 0; i < 8; ++i) {
            auto leaves = m.leaf_cells();
            m.split(leaves[rng() % leaves.size()], (rng() % 2) ? 'x' : 'y',
                    Origin::Selected);
            for (auto& kv : m.vertices())
                if (kv.second->state == VState::Weld && unif(rng) < 0.5)
                    m.promote(kv.first);
        }
        std::vector<double> xs, ys;
        for (int i = 0; i < 500; ++i) {
            xs.push_back(unif(rng));
            ys.push_back(unif(rng));
        }
        auto fk = m.free_keys();
        std::unordered_map<VKey, int> kidx;
        for (size_t i = 0; i < fk.size(); ++i) kidx[fk[i]] = (int)i;
        double mx = 0.0;
        std::vector<std::vector<double>> cols(fk.size());
        for (size_t j = 0; j < fk.size(); ++j)
            m.probe_column(fk[j], xs, ys, &cols[j]);
        std::vector<std::pair<int, double>> row;
        for (size_t i = 0; i < xs.size(); ++i) {
            m.design_row(xs[i], ys[i], kidx, &row);
            std::vector<double> dense(fk.size(), 0.0);
            for (auto& e : row) dense[e.first] = e.second;
            for (size_t j = 0; j < fk.size(); ++j)
                mx = std::max(mx, std::fabs(dense[j] - cols[j][i]));
        }
        std::printf("closure vs probe: max |diff| = %.3e (%zu free, %zu pts)\n",
                    mx, fk.size(), xs.size());
        CHECK(mx < 1e-12, "closure design equals probe columns");
    }

    // ---- 3. composite release split = +1 column
    {
        RectMesh m;
        m.split(m.leaf_cells()[0], 'x', Origin::Selected);
        Cell* left = nullptr;
        for (Cell* l : m.leaf_cells())
            if (l->x0 == 0) left = l;
        m.split(left, 'x', Origin::Selected);
        Cell* lr = nullptr;
        for (Cell* l : m.leaf_cells())
            if (l->x0 == kS / 4) lr = l;
        m.split(lr, 'y', Origin::Selected);
        const VKey key = key_of(kS / 4, kS / 2);
        CHECK(m.vertex(key)->state == VState::Hanging, "hanging setup");
        std::unordered_map<VKey, double> delta;
        for (VKey k : m.free_keys()) delta[k] = gauss(rng);
        m.resolve_heights_hier(delta);
        auto ref = grid_eval(m);
        auto fk0 = m.free_keys();
        std::vector<double> xs, ys;
        for (int i = 0; i < 400; ++i) {
            xs.push_back(unif(rng));
            ys.push_back(unif(rng));
        }
        std::vector<std::vector<double>> pre(fk0.size());
        for (size_t j = 0; j < fk0.size(); ++j)
            m.probe_column(fk0[j], xs, ys, &pre[j]);
        const int nrel = m.promote(key);
        CHECK(nrel == 1, "one release split");
        m.resolve_heights_hier(delta);
        const double inert = grid_max_diff(m, ref);
        std::printf("release: surface change from geometry = %.3e\n", inert);
        CHECK(inert < 1e-12, "release geometry inert");
        double mx = 0.0;
        std::vector<double> col;
        for (size_t j = 0; j < fk0.size(); ++j) {
            m.probe_column(fk0[j], xs, ys, &col);
            for (size_t i = 0; i < xs.size(); ++i)
                mx = std::max(mx, std::fabs(col[i] - pre[j][i]));
        }
        std::printf("release: max pre-existing column change = %.3e\n", mx);
        CHECK(mx < 1e-12, "+1 column: ancestors unchanged");
    }

    if (failures == 0) std::printf("ALL GEOMETRY TESTS PASSED\n");
    return failures == 0 ? 0 : 1;
}
