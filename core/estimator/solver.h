// pool.fit_surface: the centered conditional-mode surface, analytic-Jacobian Newton with local
// rows. Every expression is the Python's, in its order, on the same BLAS/LAPACK calls.
#pragma once
#include "estimator/padded.h"
#include "estimator/quadrature.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace estimator {

// pool.fit_surface's row_cache dict: an exact base state per row at one variance and order pair.
struct RowCache {
    bool valid = false;
    double variance = 0.;
    int count_order = 0, pool_order = 0;
    std::vector<double> eta, s, derivative;
    std::shared_ptr<const Moments> moments;   // shared with the evaluations built from it; replaced, never mutated, once shared
    void clear() { valid = false; eta.clear(); s.clear(); derivative.clear(); moments.reset(); }
};

struct SurfaceFit {
    std::vector<double> beta, eta, z;
    std::shared_ptr<const Moments> moments;   // shared by every copy of a fit; never mutated
    double score_max = 0.;
    int evaluations = 0, exact_evaluations = 0, partial_rows = 0;
    bool certified = true;
};

struct SurfaceProblem {
    const PaddedDesign* X = nullptr;
    const std::vector<double>* n = nullptr;
    const std::vector<double>* k = nullptr;
    const std::vector<double>* offset = nullptr;     // row-aligned; zeros if null
    const std::vector<double>* penalty = nullptr;    // p x p row-major (np.diag(precision)); zeros if null
    int threads = 1;
};

// The quadrature is surface_count_order() count nodes and pool_order_for(variance, 24) pool nodes.
SurfaceFit fit_surface(const SurfaceProblem& problem, double variance, const std::vector<double>& initial,
                       RowCache* row_cache = nullptr, double row_threshold = 0., bool certify = true);

// The per-row exact evaluation (pool.fit_surface's exact_rows): moments, z and dr/deta.
void exact_rows(const std::vector<double>& eta, const std::vector<double>& n, const std::vector<double>& k, double variance,
                int count_order, int pool_order, int threads, Moments& expected, std::vector<double>& z,
                std::vector<double>& derivative);

}  // namespace estimator
