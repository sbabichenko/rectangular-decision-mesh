#include "estimator/score.h"
#include "estimator/numeric.h"
#include "estimator/profile.h"

#include <stdexcept>

namespace estimator {

ScoreSystem::ScoreSystem(const PaddedDesign& X, const std::vector<double>& beta, const std::vector<double>& penalty,
                         const Moments& expected, const std::vector<double>& score, int threads)
    : X_(X), j_(expected.information), score_(score) {
    ScopeTimer timer("system.build");
    const std::size_t p = X.columns, rows = X.rows;
    v_.resize(rows);
    for (std::size_t i = 0; i < rows; ++i) v_[i] = expected.a1[i] * expected.a1[i] * expected.second[i];
    factor_.resize(p * p);
    X.weighted_gram(j_.data(), factor_.data(), threads);
    for (std::size_t i = 0; i < p * p; ++i) factor_[i] = factor_[i] + penalty[i];
    if (!cho_factor(p, factor_.data())) throw std::runtime_error("scoring system is not positive definite");
    factor_t_ = transpose_factor(p, factor_.data());
    omega_.resize(p * p);
    X.weighted_gram(v_.data(), omega_.data(), threads);
    bias_direction_.resize(p);
    dgemv(p, penalty.data(), beta.data(), bias_direction_.data());
    cho_solve(p, factor_.data(), bias_direction_.data());
}

CandidateStats ScoreSystem::candidate(const std::vector<double>& x, const std::vector<int64_t>& rows) const {
    ScopeTimer timer("system.candidate");
    const std::size_t p = X_.columns, m = rows.size();
    std::vector<double> jx(m), vx(m), s(m);
    for (std::size_t r = 0; r < m; ++r) {
        const std::size_t i = static_cast<std::size_t>(rows[r]);
        s[r] = score_[i];
        jx[r] = j_[i] * x[r]; vx[r] = v_[i] * x[r];
    }
    std::vector<double> cross(p), adjustment(p), noise_cross(p), tmp(p);
    X_.rmatvec_rows(rows, jx.data(), cross.data());
    adjustment = cross;
    cho_solve_transposed(p, factor_t_.data(), adjustment.data());
    CandidateStats out;
    out.information = ddot(m, x.data(), jx.data()) - ddot(p, cross.data(), adjustment.data());
    X_.rmatvec_rows(rows, vx.data(), noise_cross.data());
    dgemv_t(p, omega_.data(), adjustment.data(), tmp.data());
    // x@(v*x) - 2*adjustment@noise_cross + adjustment@omega@adjustment: * and @ associate left to
    // right in Python, so the middle term is (2*adjustment)@noise_cross
    std::vector<double> twice(p);
    for (std::size_t i = 0; i < p; ++i) twice[i] = 2 * adjustment[i];
    out.variance = ddot(m, x.data(), vx.data()) - ddot(p, twice.data(), noise_cross.data()) + ddot(p, tmp.data(), adjustment.data());
    out.raw_score = ddot(m, x.data(), s.data());
    out.shrinkage_bias = ddot(p, cross.data(), bias_direction_.data());
    return out;
}

}  // namespace estimator
