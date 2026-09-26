// The rectangular decision mesh: anisotropic dyadic bisection of the unit square (or cube) on an
// integer grid, multilinear interpolation on the leaf cells, and the hierarchical (surplus) basis
// the estimator fits (docs/GEOMETRY_DESIGN.md, docs/RECT_ESTIMATOR.md, docs/RECT_3D.md).
//
// Every vertex but the four roots is born bisecting a dyadic segment and has exactly two birth
// parents, the segment's endpoints. Its height is the parents' average plus its surplus, and the
// surplus is nonzero only for a free coefficient. Heights resolved this way are continuous across
// hanging vertices without stored constraints: a hanging vertex's parents lie on the coarse edge
// it sits on, so its height is that edge's linear trace. A free coefficient's design column is
// its hierarchical hat: its influence carried to every descendant at one half per generation and
// read bilinearly on the leaf cells. It never changes after birth, so there is no dormancy and
// no retirement pass.
//
// In three dimensions a vertex could also hang inside a coarse leaf's face, where two birth axes
// are possible and the parents' average need not match the face's trace. The closure keeps every
// hanging vertex on a leaf edge along its own birth axis (splitting, inertly, any leaf that would
// hold one otherwise), so the two-dimensional argument applies unchanged. In two dimensions the
// closure never fires.
#pragma once
#include "fit/observation.h"

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rect {

constexpr int kLevels = 24;                      // the integer grid is 2^24 per side
constexpr int64_t kSide = int64_t(1) << kLevels;
constexpr int kMaxDim = 3;
using Point = std::array<int64_t, kMaxDim>;       // grid coordinates; the unused axes are 0

using SparseColumn = std::vector<std::pair<int, double>>;   // (row, value), rows ascending

struct Vertex {
    static constexpr int kMinimumFitPoints = 10;
    int _id = -1;
    int64_t ix = 0, iy = 0, iz = 0;
    double x = 0.0, y = 0.0, z = 0.0;
    int p0 = -1, p1 = -1;          // birth parents; -1 for the roots
    int axis = -1;                 // the bisected segment's direction: 0 along x, 1 along y, 2 along z; -1 for a root
    int depth = 0;                 // the prior class: the bisected segment's level (1 for a unit segment)
    bool free_coefficient = false;
    bool gate_admitted = false;
    bool backbone = false;
    double surplus = 0.0;          // the coefficient; zero unless free
    double height = 0.0;           // parents' average plus surplus (logits)
    std::vector<int> children;     // vertices born with this one as a parent
    // shrinkage and support at the last fit (free coefficients)
    double em_second = -1.0;       // surplus^2 + posterior variance; -1: none yet
    double sigma_sq = 1e300;       // 1 / column information
    double sigma_pooled = 1e300, lambda_v = 0.0;
    int current_fit_points = 0;
    double current_xTx = 0.0, current_keff = 0.0;
    // the gate's record: written at every scoring (cand_*), frozen at admission (admit_*)
    int cand_round = -1;
    double cand_beta = 0.0, cand_sd = 0.0, cand_z = 0.0, cand_xTx = 0.0, cand_keff = 0.0, cand_c = 0.0;
    double cand_gain = 0.0, cand_signed_root = 0.0, cand_info_null = 0.0;
    int admit_round = -1;
    double admit_beta = 0.0, admit_sd = 0.0, admit_z = 0.0, admit_c = 0.0, admit_xTx = 0.0, admit_keff = 0.0;
    double admit_gain = 0.0, admit_signed_root = 0.0, admit_info_null = 0.0;
    // the admission's move: 0 a new edge midpoint, 1 a welded vertex promoted, 2 a hanging vertex
    // released; and how many cells it split (inert splits, before its coefficient was freed)
    int admit_move = -1, admit_splits = 0;
    int admit_chord_lo = -1, admit_chord_hi = -1;   // the far ends of the cut through it, when admitted
    bool has_prior() const { return p0 >= 0; }
    Point at() const { return {ix, iy, iz}; }
};

struct Cell {
    Point lo{}, hi{};              // the box, per axis
    int a = -1, b = -1;            // children (lower/left, upper/right); -1 for a leaf
    int axis = -1;                 // the cut: 0 at the x midline, 1 at the y midline, 2 at the z midline
    // corner k has bit d set when it sits at hi[d]: in two dimensions (x0,y0), (x1,y0), (x0,y1), (x1,y1)
    std::array<int, 8> corner{};
    std::vector<int> rows;         // a leaf's rows, ascending
    bool leaf() const { return a < 0; }
};

// A position the gate may promote: an existing constrained vertex, or the midpoint of a leaf
// edge that has no vertex yet (its would-be parents, segment direction and prior class given).
struct Candidate {
    int64_t ix = 0, iy = 0, iz = 0;
    int vertex = -1;
    int p0 = -1, p1 = -1, axis = -1, depth = 0;
};

class Mesh {
public:
    Mesh(const std::vector<Observation>& observations, const std::vector<unsigned char>& training_mask, int dim = 2);
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    // A private copy for one thread's dry-run moves (three dimensions): the geometry, reading the
    // original's per-row data (coordinates, placement, weights), which a dry run never writes.
    std::unique_ptr<Mesh> clone() const;

    const std::vector<Observation>* obs_store = nullptr;
    int n_points = 0;
    int dim = 2;                            // 2: (loan age, coupon); 3: and a third axis
    int corner_count() const { return 1 << dim; }
    std::vector<double> px, py, pz;         // pz is empty in two dimensions (a clone keeps none: see coord)
    double coord(int i, int d) const { return (*coords_[d])[i]; }
    std::vector<unsigned char> training_mask;
    std::vector<Vertex> vertices;
    std::vector<Cell> cells;
    std::size_t topology_version = 0;       // bumped at every split
    std::size_t coefficient_version = 0;    // bumped when the free set changes
    int refresh_epoch = 0;

    // the depth prior: tau^2 per class, EM over the gate's admissions
    std::map<int, double> tau_sq;
    double tau_default_sq = 1.0;
    double tau_floor_sq = 2.5e-5;
    void recompute_tau_sq();
    double tau_for_depth(int depth) const;

    // ---- geometry
    std::size_t leaf_count() const { return leaf_count_; }
    std::vector<int> leaves() const;
    int vertex_at(int64_t ix, int64_t iy) const { return vertex_at(Point{ix, iy, 0}); }
    int vertex_at(const Point& p) const;
    // The leaf holding the orthant (s[d] = +-1 per axis) next to the grid point.
    int leaf_at(int64_t ix, int64_t iy, int sx, int sy) const { return leaf_at(Point{ix, iy, 0}, {sx, sy, 1}); }
    int leaf_at(const Point& p, const std::array<int, kMaxDim>& s) const;
    void split(int cell, int axis);
    // Split the leaves around the point until it is a corner of every one: the release of a
    // hanging vertex, or the birth of a new one. Returns the vertex.
    int make_corner(int64_t ix, int64_t iy) { return make_corner(Point{ix, iy, 0}); }
    int make_corner(const Point& p);
    // Pre-refinement: split every leaf until each is 2^-levels wide on every axis (or per axis).
    void refine_uniform(int levels) { refine_uniform(std::array<int, kMaxDim>{levels, levels, levels}); }
    void refine_uniform(const std::array<int, kMaxDim>& levels);

    // ---- heights and prediction
    void recompute_heights();
    double predict_index(int i) const;
    double predict(double x, double y) const { return predict(std::array<double, kMaxDim>{x, y, 0.0}); }
    double predict(const std::array<double, kMaxDim>& p) const;
    int row_cell(int i) const { return row_cell_[i]; }

    // ---- design
    // The free coefficients carrying weight to a vertex: (vertex id, weight), ids ascending.
    const std::vector<std::pair<int, double>>& free_ancestors(int v);
    // The hierarchical design row of an observation: (free vertex id, value), ids ascending.
    void design_row(int i, std::vector<std::pair<int, double>>& out);
    // A vertex's height in the free coefficients' heights: itself if free, else its parents'
    // average, recursively (ids ascending). The nodal design reads these.
    const std::vector<std::pair<int, double>>& nodal_closure(int v);
    // The nodal design row of an observation: (free vertex id, value), ids ascending.
    void nodal_row(int i, std::vector<std::pair<int, double>>& out);

    // ---- the gate's candidates
    std::vector<Candidate> frontier() const;
    // The column the candidate's coefficient would have after its move, on every row.
    SparseColumn candidate_column(const Candidate& candidate) const;
    // The candidate's current position record (an earlier admission may have created its vertex).
    Candidate refresh(const Candidate& candidate) const;
    // Realise the candidate's geometry and return its vertex (not yet free).
    int realise(const Candidate& candidate);
    // The candidate's column exactly as its move will leave it. In two dimensions this is
    // candidate_column. In three the closure's splits can reach vertices born along another axis
    // on the edges through the point, which the pre-move reading cannot see; the move is made,
    // the column read and the move undone (not thread-safe: call it serially).
    SparseColumn exact_candidate_column(const Candidate& candidate);
    // The cut through a vertex: the leaf edges through it across its segment. Their far ends,
    // below/left and above/right (-1 where no leaf edge runs that way, or for a root). Two
    // dimensions only: in three the cut is a plane (both -1).
    std::array<int, 2> chord_ends(int v) const;
    // Three dimensions: whether every vertex is a corner of the leaves around it or hangs on a leaf
    // edge along its own birth axis (the closure's invariant; the tests check it).
    bool hanging_invariant_holds() const;
    // Three dimensions: split leaves until every vertex created since the last closure satisfies
    // the hanging invariant (make_corner and refine_uniform close; a bare split does not).
    void close_pending();

private:
    Mesh(const Mesh& other, int);   // clone()'s copy
    std::size_t leaf_count_ = 0;
    struct PointHash {
        std::size_t operator()(const Point& p) const {
            uint64_t h = static_cast<uint64_t>(p[0]) * 0x9E3779B97F4A7C15ULL;
            h ^= static_cast<uint64_t>(p[1]) + 0x632BE59BD9B4E019ULL + (h << 6) + (h >> 2);
            h ^= static_cast<uint64_t>(p[2]) + 0x85EBCA77C2B2AE63ULL + (h << 6) + (h >> 2);
            return static_cast<std::size_t>(h);
        }
    };
    std::unordered_map<Point, int, PointHash> vertex_index_;
    std::vector<std::vector<int>> corner_cells_;    // vertex -> the leaves with it as a corner
    std::vector<int> row_cell_;
    std::vector<std::array<double, 8>> row_weight_;
    // the per-row data read: this mesh's own, or (a clone) the original's
    const std::vector<double>* coords_[kMaxDim] = {&px, &py, &pz};
    const std::vector<std::array<double, 8>>* weights_ = &row_weight_;
    // a row's corner weights in a cell: the computation place_rows stores
    void weights_in(const Cell& c, int i, std::array<double, 8>& w) const;
    std::vector<int> pending_;                      // three dimensions: new vertices the closure has not seen
    // the undo log of a dry-run move: what split and make_vertex change in existing structures
    struct Journal {
        std::size_t vertices = 0, cells = 0, leaf_count = 0, topology = 0;
        std::vector<std::pair<int, Cell>> cells_before;
        std::vector<std::pair<int, std::vector<int>>> corners_before;
        std::vector<std::pair<int, std::size_t>> children_before;
        std::unordered_set<int> cells_seen, corners_seen, children_seen;
        std::vector<int> pending;
    };
    Journal* journal_ = nullptr;
    void rollback(Journal& journal);
    std::size_t ancestors_version_ = ~std::size_t(0);
    std::vector<std::vector<std::pair<int, double>>> ancestors_;
    std::vector<char> ancestors_done_;
    std::size_t closure_version_ = ~std::size_t(0);
    std::vector<std::vector<std::pair<int, double>>> closure_;
    std::vector<char> closure_done_;

    int make_vertex(const Point& p, int p0, int p1, int axis);
    void place_rows(int cell);
    bool is_corner(const Cell& c, const Point& p) const;
    // the orthant signs of number o (axis 0 the most significant), and whether it leaves the grid
    std::array<int, kMaxDim> orthant(int o) const;
    bool outside(const Point& p, const std::array<int, kMaxDim>& s) const;
    // the axes along which p lies strictly inside the cell's range
    int interior_axes(const Cell& c, const Point& p, std::array<int, kMaxDim>& axes) const;
};

// The prior class of a vertex bisecting a segment of `length` grid units.
int depth_of_segment(int64_t length);

}  // namespace rect
