// engine.ScoreSystem: first-order nuisance refit, sampling variance and shrinkage bias of a
// candidate column against the fitted surface.
#pragma once
#include "estimator/padded.h"
#include "estimator/quadrature.h"

#include <cstdint>
#include <vector>

namespace estimator {

struct CandidateStats {
    double information = 0., variance = 0., raw_score = 0., shrinkage_bias = 0.;
};

class ScoreSystem {
public:
    ScoreSystem(const PaddedDesign& X, const std::vector<double>& beta, const std::vector<double>& penalty,
                const Moments& expected, const std::vector<double>& score, int threads = 1);
    // rows: training row positions of the candidate's entries (sorted, unique), x their values.
    CandidateStats candidate(const std::vector<double>& x, const std::vector<int64_t>& rows) const;
    // The Cholesky factor of X'JX + P (J the centered information): the fit's working posterior
    // precision, which the depth variance's second moments are read from.
    const std::vector<double>& factor() const { return factor_; }

private:
    const PaddedDesign& X_;
    std::vector<double> j_, v_, score_, factor_, omega_, bias_direction_;
    std::vector<double> factor_t_;   // the factor in column order, for the per-candidate solves
};

}  // namespace estimator
