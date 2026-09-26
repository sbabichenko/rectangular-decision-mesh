// The live system of the rectangular mesh, in nodal coordinates: the free coefficients' heights,
// the padded training design B of their nodal columns (a constrained vertex carries its parents'
// average, a free one stops the propagation), and the depth prior on the surpluses d = R h as the
// dense penalty R' diag(1/tau^2) R. The same model as the surplus basis X = B R^-1 with a diagonal
// prior (the triangular engine fits that one), but B's rows are at most a few entries wide where
// the hierarchical rows are tens: every Gram, refit and candidate product scales with the width
// squared.
#pragma once
#include "estimator/padded.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace rect { class Mesh; }

namespace estimator {

struct SparseRow { std::vector<std::pair<int64_t, double>> entries; };   // sorted by column

struct LiveSystem {
    std::vector<int> free;                 // vertex ids by (depth, id): the roots first
    std::vector<int> children;             // the free coefficients with a prior, in free order
    std::size_t root_count = 0;
    std::vector<SparseRow> transform;      // p rows: the surpluses d = R h (a root's is its height)
    std::shared_ptr<const PaddedDesign> X; // training rows by free coefficients: the nodal design B
    std::vector<double> initial;           // the free vertices' current heights
    std::vector<double> precision;         // per surplus: 1 for roots (sd 1 logit), 1/tau(depth) for children
    std::vector<double> penalty;           // p x p, row-major: R' diag(precision) R
};

// The structural parts are reused while the topology, the free set and the rows are unchanged.
struct LiveCache {
    bool have = false;
    std::vector<int64_t> key;
    LiveSystem system;
};

LiveSystem live_system(rect::Mesh& mesh, bool heldout_split, const std::vector<int64_t>& rows, LiveCache* cache = nullptr);

// R h: the surpluses of the free coefficients from their heights.
std::vector<double> surpluses_from(const LiveSystem& system, const std::vector<double>& heights);

}  // namespace estimator
