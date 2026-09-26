#pragma once
// Rectangular decision mesh geometry (GEOMETRY_DESIGN.md): anisotropic
// dyadic bisection on integer coordinates, the two weld species, 2:1
// balance with welded byproducts, composite release splits, bilinear
// evaluation, and hierarchical (surplus) design columns through stored
// constraint closures. C++ port of poc/rect_mesh.py.

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rectmesh {

constexpr int kMaxLevel = 12;
constexpr int kS = 1 << kMaxLevel;

using VKey = int64_t;
inline VKey key_of(int ix, int iy) {
    return (static_cast<int64_t>(ix) << 32) | static_cast<uint32_t>(iy);
}
inline int key_ix(VKey k) { return static_cast<int>(k >> 32); }
inline int key_iy(VKey k) { return static_cast<int>(static_cast<uint32_t>(k)); }

enum class VState { Free, Weld, Hanging };
enum class Origin { Root, Seed, Selected, Release, Balance };

struct Vertex {
    int ix, iy;
    VState state;
    double height = 0.0;
    // exactly two birth parents for every non-root vertex
    std::vector<std::pair<Vertex*, double>> birth_parents;
    int depth;
    Origin origin;
    double x() const { return static_cast<double>(ix) / kS; }
    double y() const { return static_cast<double>(iy) / kS; }
};

struct Cell {
    int x0, y0, x1, y1;
    int lx, ly;
    Cell* child_a = nullptr;
    Cell* child_b = nullptr;
    char axis = 0;  // 'x' or 'y' once split
    bool leaf() const { return child_a == nullptr; }
    bool contains(int ix, int iy) const {
        return x0 <= ix && ix <= x1 && y0 <= iy && iy <= y1;
    }
};

class RectMesh {
  public:
    RectMesh();

    // ---- topology
    void split(Cell* cell, char axis, Origin origin);
    int promote(VKey key);  // composite move; returns #release splits
    const std::vector<Cell*>& leaf_cells() const { return leaves_; }
    Cell* root() { return root_; }
    Vertex* vertex(VKey key) {
        auto it = verts_.find(key);
        return it == verts_.end() ? nullptr : it->second;
    }
    const std::unordered_map<VKey, Vertex*>& vertices() const { return verts_; }
    std::vector<VKey> free_keys() const;
    // current constraint pair (hanging masters or birth parents)
    std::vector<std::pair<Vertex*, double>> masters(const Vertex* v) const;

    // ---- heights and evaluation
    // hierarchical parameterization: every vertex sits at its parent
    // interpolation plus its own surplus (delta only holds probed keys)
    void resolve_heights_hier(const std::unordered_map<VKey, double>& delta);
    const Cell* find_leaf(double x, double y) const;
    double eval(double x, double y) const;

    // ---- design through stored constraint closures (GEOMETRY_DESIGN §3)
    // closure(v) = sparse weights of every FREE vertex's surplus on v's
    // height; memoized per topology version.
    const std::vector<std::pair<VKey, double>>& closure(Vertex* v);
    // sparse hierarchical design row for a point: (free-key index, weight)
    // pairs; key_index maps free keys to column indices.
    void design_row(double x, double y,
                    const std::unordered_map<VKey, int>& key_index,
                    std::vector<std::pair<int, double>>* out);
    // exact column of a unit surplus injected at v0 (free or not),
    // evaluated at the given points.
    void probe_column(VKey v0, const std::vector<double>& xs,
                      const std::vector<double>& ys, std::vector<double>* out);

    uint64_t topology_version() const { return topo_version_; }

  private:
    Vertex* get_or_make(int ix, int iy,
                        std::vector<std::pair<Vertex*, double>> parents,
                        int depth, Origin origin);
    void plain_split(Cell* cell, char axis, Origin origin);
    void rebalance();
    void recompute_hanging();

    std::vector<std::unique_ptr<Vertex>> vert_store_;
    std::vector<std::unique_ptr<Cell>> cell_store_;
    std::unordered_map<VKey, Vertex*> verts_;
    Cell* root_ = nullptr;
    std::vector<Cell*> leaves_;
    std::unordered_map<VKey, std::pair<VKey, VKey>> hanging_;
    bool balance_enabled_ = true;
    uint64_t topo_version_ = 0;
    // closure memo, invalidated on topology change
    uint64_t closure_version_ = ~0ull;
    std::unordered_map<const Vertex*, std::vector<std::pair<VKey, double>>>
        closure_memo_;
    void closure_of(Vertex* v, std::unordered_map<VKey, double>* acc,
                    double w);
};

}  // namespace rectmesh
