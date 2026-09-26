#include "estimator/variance.h"
#include "estimator/numeric.h"
#include "estimator/profile.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace estimator {

std::vector<double> refit_second_moments(const PaddedDesign& X, const std::vector<double>& beta, const Moments& expected,
                                         const std::vector<double>& penalty, int threads) {
    ScopeTimer timer("refit_second_moments");
    const std::size_t p = X.columns, rows = X.rows;
    // J = P + X^T diag(j) X; score covariance = X^T diag(a1^2 second) X
    std::vector<double> J(p * p), cov(p * p), w(rows);
    X.weighted_gram(expected.information.data(), J.data(), threads);
    for (std::size_t i = 0; i < p * p; ++i) J[i] = penalty[i] + J[i];
    for (std::size_t i = 0; i < rows; ++i) w[i] = expected.a1[i] * expected.a1[i] * expected.second[i];
    X.weighted_gram(w.data(), cov.data(), threads);
    std::vector<double> factor = J;
    if (!cho_factor(p, factor.data())) throw std::runtime_error("refit information is not positive definite");
    std::vector<double> inverse(p * p);
    cho_inverse(p, factor.data(), inverse.data());
    // scipy returns the inverse Fortran-ordered; numpy's products then run on that buffer with
    // transpose flags. quadratic_forms sees a C-contiguous copy (the same values).
    std::vector<double> inverse_t(p * p);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j) inverse_t[j * p + i] = inverse[i * p + j];
    std::vector<double> leverage(rows), sandwich(rows), tmp(p * p), sand(p * p);
    X.quadratic_forms(inverse.data(), leverage.data(), threads);
    dgemm_tn(p, inverse_t.data(), cov.data(), tmp.data());      // inverse @ score_covariance
    dgemm_nt(p, tmp.data(), inverse_t.data(), sand.data());     // (...) @ inverse
    X.quadratic_forms(sand.data(), sandwich.data(), threads);
    std::vector<double> pb(p), ipb(p), bias(rows), corrected(rows);
    dgemv(p, penalty.data(), beta.data(), pb.data());
    dgemv_fortran(p, inverse_t.data(), pb.data(), ipb.data());   // inverse @ (P @ beta)
    X.matvec(ipb.data(), bias.data(), threads);
    for (std::size_t i = 0; i < rows; ++i) {
        const double derivative = expected.information[i] / expected.a1[i];
        const double b = derivative * bias[i];
        corrected[i] = expected.second[i] - 2 * expected.information[i] * leverage[i] * expected.second[i]
                       + derivative * derivative * sandwich[i] + b * b;
    }
    for (double c : corrected) if (c < -1e-8) throw std::runtime_error("negative linearized second moment");
    for (double& c : corrected) c = std::max(c, 0.);
    return corrected;
}

const Moments& accurate_moments(const std::vector<double>& eta, const std::vector<double>& n, double variance,
                                AccurateCache* cache, double row_threshold, int threads, Moments& scratch) {
    ScopeTimer timer("accurate_moments");
    if (cache == nullptr || row_threshold <= 0) {
        scratch = moments(eta, n, variance, 8, pool_order_for(variance, 32), threads);
        return scratch;
    }
    const bool usable = cache->valid && cache->variance == variance && cache->eta.size() == eta.size();
    if (!usable) {
        Moments expected = moments(eta, n, variance, 8, pool_order_for(variance, 32), threads);
        cache->clear();
        cache->valid = true; cache->variance = variance; cache->eta = eta; cache->moments = std::move(expected);
        cache->rows_resolved = cache->rows_total = static_cast<int>(eta.size());
        return cache->moments;
    }
    std::vector<int64_t> idx;
    for (std::size_t i = 0; i < eta.size(); ++i)
        if (std::fabs(eta[i] - cache->eta[i]) > row_threshold) idx.push_back(static_cast<int64_t>(i));
    if (!idx.empty()) {
        std::vector<double> eta_rows(idx.size()), n_rows(idx.size());
        for (std::size_t r = 0; r < idx.size(); ++r) { eta_rows[r] = eta[idx[r]]; n_rows[r] = n[idx[r]]; }
        Moments expected = moments(eta_rows, n_rows, variance, 8, pool_order_for(variance, 32), threads);
        for (std::size_t r = 0; r < idx.size(); ++r) cache->eta[idx[r]] = eta_rows[r];
        cache->moments.assign_rows(idx, expected);
    }
    cache->rows_resolved = static_cast<int>(idx.size()); cache->rows_total = static_cast<int>(eta.size());
    return cache->moments;
}

double dispersion_score(const SurfaceProblem& problem, const std::vector<double>& beta, double variance,
                        const SurfaceFit& surface, const Moments* expected, AccurateCache* accurate_cache,
                        double row_threshold) {
    const PaddedDesign& X = *problem.X;
    const std::vector<double>& z = surface.z;
    Moments scratch;
    const Moments* used = expected;
    if (used == nullptr) {
        std::vector<double> eta(X.rows);
        X.matvec(beta.data(), eta.data(), problem.threads);
        for (std::size_t i = 0; i < X.rows; ++i) eta[i] = (*problem.offset)[i] + eta[i];
        used = &accurate_moments(eta, *problem.n, variance, accurate_cache, row_threshold, problem.threads, scratch);
    }
    const std::vector<double> second = refit_second_moments(X, beta, *used, *problem.penalty, problem.threads);
    std::vector<double> diff(X.rows);
    for (std::size_t i = 0; i < X.rows; ++i) diff[i] = z[i] * z[i] - second[i];
    return ddot(X.rows, used->weight.data(), diff.data());
}

double initial_variance_guess(const SurfaceProblem& problem, const std::vector<double>& beta, double score,
                              double lo, double hi) {
    std::vector<double> eta(problem.X->rows);
    problem.X->matvec(beta.data(), eta.data(), problem.threads);
    for (std::size_t i = 0; i < eta.size(); ++i) eta[i] = (*problem.offset)[i] + eta[i];
    Moments high = moments(eta, *problem.n, 0., 8, 32, problem.threads);
    const double sensitivity = ddot(high.size(), high.weight.data(), high.sensitivity.data());
    return (sensitivity > 0 && score > 0) ? std::clamp(score / sensitivity, lo, hi) : 0.;
}

namespace {

VarianceFit alternating_variance(const SurfaceProblem& problem, const std::vector<double>& initial, double upper,
                                 double initial_variance, double relative_tolerance) {
    VarianceFit out;
    std::vector<double> beta = initial;
    double v = std::clamp(initial_variance ? initial_variance : .1 * upper, upper * 1e-4, upper * .95);
    SurfaceFit zero = fit_surface(problem, 0., beta);
    double previous_v = 0.;
    double previous_score = dispersion_score(problem, zero.beta, 0., zero);
    int calls = zero.evaluations;
    if (previous_score <= 0) {
        // the dispersion equation is nonpositive already at zero variance: the root is the boundary
        out.resolved = true; out.variance = 0.; out.status = "lower_boundary"; out.score = previous_score; out.beta = zero.beta;
        out.surface = zero; out.evaluations = 1; out.surface_evaluations_total = calls;
        out.variance_tolerance = relative_tolerance * .01; out.variance_step = 0.; out.relative_tolerance = relative_tolerance;
        out.coarse_evaluations = 0; out.high_evaluations = 1;
        return out;
    }
    if (!initial_variance) {
        const double guess = initial_variance_guess(problem, zero.beta, previous_score, upper * 1e-4, upper * .95);
        if (guess > 0) v = guess;
    }
    double lo = 0., hi = upper;
    bool coarse = true;
    SurfaceFit previous_surface = zero;
    int coarse_evaluations = 0, high_evaluations = 1;
    std::string reason;
    int iteration = 0;
    // The trial fits only guide the search: they reuse rows through a row cache (rows whose
    // predictor drifted less than the threshold keep their moments, corrected to first order),
    // and only the surface at the root is certified on exact moments before it is returned.
    RowCache trial_cache;
    constexpr double kTrialRowThreshold = 1e-2;
    for (; iteration < 20; ++iteration) {
        try {
            SurfaceFit surface = fit_surface(problem, v, beta, &trial_cache, kTrialRowThreshold, false);
            calls += surface.evaluations;
            beta = surface.beta;
            double score = dispersion_score(problem, beta, v, surface, coarse ? surface.moments.get() : nullptr);
            coarse_evaluations += coarse ? 1 : 0;
            high_evaluations += coarse ? 0 : 1;
            double slope = (score - previous_score) / (v - previous_v);
            if (!std::isfinite(score)) { reason = "nonfinite variance score"; break; }
            double step = (std::isfinite(slope) && slope < 0) ? -score / slope : std::numeric_limits<double>::infinity();
            if (coarse && std::fabs(step) < 1e-3 * std::max(v, .01)) {
                previous_score = dispersion_score(problem, previous_surface.beta, previous_v, previous_surface);
                score = dispersion_score(problem, beta, v, surface);
                high_evaluations += 2;
                if (!std::isfinite(previous_score) || !std::isfinite(score)) throw std::runtime_error("nonfinite accurate dispersion score");
                coarse = false;
                lo = 0.; hi = upper;
                if (previous_score > 0) lo = previous_v;
                else if (previous_score < 0) hi = previous_v;
                slope = (score - previous_score) / (v - previous_v);
                step = (std::isfinite(slope) && slope < 0) ? -score / slope : std::numeric_limits<double>::infinity();
            }
            const double variance_tolerance = relative_tolerance * std::max(v, .01);
            if (!coarse && std::fabs(step) < variance_tolerance && surface.score_max < 2e-9) {
                surface = fit_surface(problem, v, beta, &trial_cache, kTrialRowThreshold, true);   // certified
                calls += surface.evaluations;
                beta = surface.beta;
                out.resolved = true; out.variance = v; out.status = "root"; out.score = score; out.beta = beta;
                out.surface = surface;
                out.evaluations = iteration + 2; out.surface_evaluations_total = calls;
                out.variance_tolerance = variance_tolerance; out.variance_step = step; out.relative_tolerance = relative_tolerance;
                out.coarse_evaluations = coarse_evaluations; out.high_evaluations = high_evaluations;
                return out;
            }
            if (score > 0) lo = v; else hi = v;
            if (lo >= hi) { reason = "inconsistent variance bracket"; break; }
            double trial = v + step;
            if (!(lo < trial && trial < hi)) trial = .5 * (lo + hi);
            if (trial == v) { reason = "variance iteration stagnated"; break; }
            previous_v = v; previous_score = score;
            previous_surface = surface;
            v = trial;
        } catch (const std::exception& exc) {
            reason = exc.what();
            break;
        }
    }
    if (iteration == 20) reason = "alternating iteration limit";
    out.reason = reason; out.evaluations = iteration + 2; out.surface_evaluations_total = calls;
    return out;
}

}  // namespace

VarianceFit fit_variance(const SurfaceProblem& problem, const std::vector<double>& initial, double upper,
                         double initial_variance, double relative_tolerance) {
    if (!std::isfinite(upper) || upper <= 0) throw std::invalid_argument("positive finite variance upper bound required");
    if (!std::isfinite(initial_variance) || initial_variance < 0) throw std::invalid_argument("finite nonnegative starting variance required");
    if (!std::isfinite(relative_tolerance) || !(0 < relative_tolerance && relative_tolerance <= 1e-2)) throw std::invalid_argument("relative variance tolerance must lie in (0, 1e-2]");
    bool any_pair = false;
    for (double v : *problem.n) if (v > 1) { any_pair = true; break; }
    if (!any_pair) { VarianceFit out; out.status = "unidentified"; return out; }
    VarianceFit out = alternating_variance(problem, initial, upper, initial_variance, relative_tolerance);
    if (out.resolved) out.solver_used = "alternating";
    return out;
}

}  // namespace estimator
