#include "estimator/solver.h"
#include "estimator/numeric.h"
#include "estimator/profile.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace estimator {

void exact_rows(const std::vector<double>& eta, const std::vector<double>& n, const std::vector<double>& k, double variance,
                int count_order, int pool_order, int threads, Moments& expected, std::vector<double>& z,
                std::vector<double>& derivative) {
    { ScopeTimer t("  moments"); expected = moments(eta, n, variance, count_order, pool_order, threads); }
    std::vector<double> residual, effective;
    { ScopeTimer t("  scaled_mode"); scaled_mode(expected.location, n, k, variance, residual, effective); }
    const std::size_t rows = eta.size();
    z.resize(rows); derivative.resize(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        z[i] = residual[i] - expected.mean[i];
        // dr/deta = -A_eta*w_conditional/(1+v*w_conditional)
        derivative[i] = expected.a2[i] * z[i] + expected.a1[i] * (-expected.a1[i] * effective[i] - expected.mean_derivative[i]);
    }
}

namespace {

struct Evaluation {
    bool have = false;
    std::vector<double> beta, score, jacobian, eta, z;
    std::shared_ptr<const Moments> moments;
    bool exact = false;
};

}  // namespace

SurfaceFit fit_surface(const SurfaceProblem& problem, double variance, const std::vector<double>& initial,
                       RowCache* row_cache, double row_threshold, bool certify) {
    ScopeTimer timer_fit("fit_surface");
    const PaddedDesign& X = *problem.X;
    const std::vector<double>& n = *problem.n;
    const std::vector<double>& k = *problem.k;
    const std::size_t rows = X.rows, p = X.columns;
    const int count_order = surface_count_order();
    const int pool_order = pool_order_for(variance, 24);
    if (initial.size() != p || n.size() != rows || k.size() != rows) throw std::invalid_argument("incompatible surface design and counts");
    for (std::size_t i = 0; i < rows; ++i)
        if (!std::isfinite(k[i]) || k[i] < 0 || k[i] > n[i] || k[i] != std::floor(k[i])) throw std::invalid_argument("valid integer loan counts required");
    std::vector<double> zeros_rows, zeros_penalty;
    const std::vector<double>* offset = problem.offset;
    if (offset == nullptr) { zeros_rows.assign(rows, 0.); offset = &zeros_rows; }
    if (offset->size() != rows) throw std::invalid_argument("finite row-aligned offsets required");
    for (double v : *offset) if (!std::isfinite(v)) throw std::invalid_argument("finite row-aligned offsets required");
    const std::vector<double>* penalty = problem.penalty;
    if (penalty == nullptr) { zeros_penalty.assign(p * p, 0.); penalty = &zeros_penalty; }
    if (penalty->size() != p * p) throw std::invalid_argument("symmetric positive semidefinite penalty required");
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            if (!std::isfinite((*penalty)[i * p + j]) || (*penalty)[i * p + j] != (*penalty)[j * p + i])
                throw std::invalid_argument("symmetric positive semidefinite penalty required");
    // relative to the penalty's scale: the nodal prior R' diag(1/tau^2) R mixes precisions up to
    // 1/tau_floor^2 = 4e4 with small ones, and an eigenvalue that is zero or near it rounds to
    // about 1e-11 either side (four fits failed the absolute test when the roots had no prior)
    double scale = 0.0;
    for (double v : *penalty) scale = std::max(scale, std::fabs(v));
    if (min_eigenvalue(p, penalty->data()) < -1e-12 * std::max(1.0, scale * static_cast<double>(p)))
        throw std::invalid_argument("symmetric positive semidefinite penalty required");
    if (row_threshold < 0 || !std::isfinite(row_threshold)) throw std::invalid_argument("row threshold must be a finite nonnegative logit distance");
    const bool local = row_threshold > 0 && row_cache != nullptr;
    const int threads = problem.threads;
    Evaluation last;
    int evaluations = 0, exact_evaluations = 0, partial_rows = 0;
    auto cache_usable = [&]() {
        return local && row_cache->valid && row_cache->variance == variance && row_cache->count_order == count_order
               && row_cache->pool_order == pool_order && row_cache->eta.size() == rows;
    };
    // evaluate(beta, exact): score and jacobian at beta; the memo of the last evaluation applies
    // when beta is unchanged and the memo is exact or exactness is not required
    std::vector<double> score(p), jacobian(p * p);
    auto evaluate = [&](const std::vector<double>& beta, bool exact) {
        if (last.have && beta == last.beta && (last.exact || !exact)) { score = last.score; jacobian = last.jacobian; return; }
        ++evaluations;
        std::vector<double> eta(rows);
        X.matvec(beta.data(), eta.data(), threads);
        for (std::size_t i = 0; i < rows; ++i) eta[i] = (*offset)[i] + eta[i];
        std::vector<double> s(rows), derivative, z;
        std::shared_ptr<const Moments> expected;
        bool was_exact;
        if (!exact && cache_usable()) {
            std::vector<double> drift(rows);
            std::vector<int64_t> idx;
            for (std::size_t i = 0; i < rows; ++i) {
                drift[i] = eta[i] - row_cache->eta[i];
                if (std::fabs(drift[i]) > row_threshold) idx.push_back(static_cast<int64_t>(i));
            }
            if (!idx.empty()) {
                std::vector<double> eta_rows(idx.size()), n_rows(idx.size()), k_rows(idx.size());
                for (std::size_t r = 0; r < idx.size(); ++r) { eta_rows[r] = eta[idx[r]]; n_rows[r] = n[idx[r]]; k_rows[r] = k[idx[r]]; }
                Moments expected_rows; std::vector<double> z_rows, derivative_rows;
                exact_rows(eta_rows, n_rows, k_rows, variance, count_order, pool_order, threads, expected_rows, z_rows, derivative_rows);
                for (std::size_t r = 0; r < idx.size(); ++r) {
                    row_cache->eta[idx[r]] = eta_rows[r];
                    row_cache->s[idx[r]] = expected_rows.a1[r] * z_rows[r];
                    row_cache->derivative[idx[r]] = derivative_rows[r];
                    drift[idx[r]] = 0.;
                }
                // copy-on-write: an evaluation may still hold the cache's moments
                auto updated = std::make_shared<Moments>(*row_cache->moments);
                updated->assign_rows(idx, expected_rows);
                row_cache->moments = std::move(updated);
                partial_rows += static_cast<int>(idx.size());
            }
            for (std::size_t i = 0; i < rows; ++i) s[i] = row_cache->s[i] + row_cache->derivative[i] * drift[i];
            derivative = row_cache->derivative;
            expected = row_cache->moments;
            z.resize(rows);
            for (std::size_t i = 0; i < rows; ++i) z[i] = s[i] / expected->a1[i];
            was_exact = false;
        } else {
            ++exact_evaluations;
            Moments computed;
            exact_rows(eta, n, k, variance, count_order, pool_order, threads, computed, z, derivative);
            for (std::size_t i = 0; i < rows; ++i) s[i] = computed.a1[i] * z[i];
            expected = std::make_shared<const Moments>(std::move(computed));
            if (local) {
                row_cache->clear();
                row_cache->valid = true; row_cache->variance = variance; row_cache->count_order = count_order; row_cache->pool_order = pool_order;
                row_cache->eta = eta; row_cache->s = s; row_cache->derivative = derivative; row_cache->moments = expected;
            }
            was_exact = true;
        }
        // score = X^T s - P beta; jacobian = X^T diag(derivative) X - P
        std::vector<double> xs(p), pb(p);
        X.rmatvec(s.data(), xs.data());
        dgemv(p, penalty->data(), beta.data(), pb.data());
        for (std::size_t j = 0; j < p; ++j) score[j] = xs[j] - pb[j];
        { ScopeTimer t("  gram"); X.weighted_gram(derivative.data(), jacobian.data(), threads); }
        for (std::size_t j = 0; j < p * p; ++j) jacobian[j] = jacobian[j] - (*penalty)[j];
        last.have = true; last.beta = beta; last.score = score; last.jacobian = jacobian; last.eta = std::move(eta);
        last.moments = std::move(expected); last.z = std::move(z); last.exact = was_exact;
    };
    auto newton = [&](std::vector<double>& beta, bool exact) {
        // 40 iterations and halvings down to 2^-12: a fit that converges within the old 12 steps
        // and 4 halvings takes exactly the same path; the extra room is for designs whose
        // starting score is of order 1e5 (Ginnie January 2020 Majors: pools of 1e5 loans)
        for (int it = 0; it < 40; ++it) {
            if (max_abs(score) < 2e-9) break;
            std::vector<double> step = score;
            { ScopeTimer t("  solve"); if (!solve(p, jacobian.data(), step.data())) break; }
            const double old_norm = norm2(score);
            bool accepted = false;
            for (double scale = 1.0; scale >= 1.0 / 4096; scale *= 0.5) {
                std::vector<double> trial(p);
                bool finite = true;
                for (std::size_t j = 0; j < p; ++j) { trial[j] = beta[j] - scale * step[j]; if (!std::isfinite(trial[j])) finite = false; }
                if (!finite) continue;
                // the memo is moved aside, not copied: a trial differs from last.beta (the step is
                // nonzero), so it can never hit the memo, and a rejected trial moves it back
                std::vector<double> saved_score = score, saved_jacobian;
                saved_jacobian.swap(jacobian); jacobian.resize(p * p);
                Evaluation saved_last = std::move(last); last = Evaluation();
                try {
                    evaluate(trial, exact);
                } catch (const std::runtime_error&) {
                    // a speculative Newton move left the valid domain: rejected, state restored
                    score = saved_score; jacobian.swap(saved_jacobian); last = std::move(saved_last);
                    continue;
                }
                if (norm2(score) < old_norm) { beta = trial; accepted = true; break; }
                score = saved_score; jacobian.swap(saved_jacobian); last = std::move(saved_last);
            }
            if (!accepted) break;
        }
    };
    std::vector<double> beta = initial;
    evaluate(beta, false);
    newton(beta, false);
    if (max_abs(score) >= 2e-9 && local) {
        // The cached rows are linearised about their base and switch to exact moments when they
        // drift past the threshold, so the line search can stall on the switch. Re-base the cache
        // at the stalled point and continue; failing that, finish on exact moments.
        evaluate(beta, true);
        newton(beta, false);
        if (max_abs(score) >= 2e-9) { evaluate(beta, true); newton(beta, true); }
    }
    if (max_abs(score) >= 2e-9) throw std::runtime_error("centered surface score unresolved: analytic Newton did not converge");
    const bool certified = !local || certify;
    if (local && certify) {
        evaluate(beta, true);
        newton(beta, true);
        if (max_abs(score) >= 2e-9) throw std::runtime_error("centered surface score unresolved: analytic Newton did not converge");
    } else {
        evaluate(beta, false);
    }
    std::vector<double> negated(p * p);
    for (std::size_t j = 0; j < p * p; ++j) negated[j] = -jacobian[j];
    if (max_abs(score) > 2e-5 || min_eigenvalue(p, negated.data()) <= 0) throw std::runtime_error("centered surface score unresolved: analytic Newton converged");
    SurfaceFit fit;
    fit.beta = beta; fit.eta = std::move(last.eta); fit.z = std::move(last.z); fit.moments = last.moments;
    fit.score_max = max_abs(score); fit.evaluations = evaluations; fit.exact_evaluations = exact_evaluations;
    fit.partial_rows = partial_rows; fit.certified = certified;
    return fit;
}

}  // namespace estimator
