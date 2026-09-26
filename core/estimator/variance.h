// pool.py's variance side: the linearized second-moment refit, accurate moments with a row
// cache, the dispersion score and the alternating solver.
#pragma once
#include "estimator/padded.h"
#include "estimator/quadrature.h"
#include "estimator/solver.h"

#include <string>
#include <vector>

namespace estimator {

struct AccurateCache {
    bool valid = false;
    double variance = 0.;
    std::vector<double> eta;
    Moments moments;
    int rows_resolved = 0, rows_total = 0;
    void clear() { valid = false; eta.clear(); moments.resize(0); rows_resolved = rows_total = 0; }
};

// pool.refit_second_moments on a padded design.
std::vector<double> refit_second_moments(const PaddedDesign& X, const std::vector<double>& beta, const Moments& expected,
                                         const std::vector<double>& penalty, int threads);
// pool.accurate_moments: 8/32 moments, re-solving only the rows that drifted when a cache is given.
const Moments& accurate_moments(const std::vector<double>& eta, const std::vector<double>& n, double variance,
                                AccurateCache* cache, double row_threshold, int threads, Moments& scratch);
// pool._dispersion_score. expected == nullptr evaluates the accurate moments at the fitted mean.
double dispersion_score(const SurfaceProblem& problem, const std::vector<double>& beta, double variance,
                        const SurfaceFit& surface, const Moments* expected = nullptr, AccurateCache* accurate_cache = nullptr,
                        double row_threshold = 0.);

struct VarianceFit {
    bool resolved = false;          // variance is not None
    double variance = 0.;
    std::string status, reason, solver_used;
    double score = 0.;
    std::vector<double> beta;
    SurfaceFit surface;
    int evaluations = 0, surface_evaluations_total = 0, coarse_evaluations = 0, high_evaluations = 0;
    double variance_tolerance = 0., variance_step = 0., relative_tolerance = 0.;
};

// The starting variance from the dispersion score at zero variance and the moments' sensitivity
// there: clamp(score / sensitivity, lo, hi), or 0 when either is not positive.
double initial_variance_guess(const SurfaceProblem& problem, const std::vector<double>& beta, double score,
                              double lo, double hi);

// The pool variance: the alternating solver (coarse dispersion scores on the surface moments
// until the step is small, then accurate moments until the root is within the tolerance).
VarianceFit fit_variance(const SurfaceProblem& problem, const std::vector<double>& initial, double upper,
                         double initial_variance, double relative_tolerance);   // initial_variance 0: start from .1 upper

}  // namespace estimator
