#pragma once
// The binomial working quantities every diagnostic recomputes: the logit at an
// observation from the mesh's prediction (logits), the fitted probability,
// and the Pearson residual with its information. One spelling, so the scoring,
// the dumps and the diagnostics agree to the bit.
#include "fit/observation.h"

#include <cmath>

inline double eta_of(const Observation& o, double prediction) {
    return o.offset + prediction;
}

inline double logistic(double eta) { return 1.0 / (1.0 + std::exp(-eta)); }

struct PearsonTerms {
    double info;       // n p (1 - p)
    double residual;   // (k - n p) / sqrt(info)
};

inline PearsonTerms pearson_terms(const Observation& o, double p) {
    const double info = o.exposure * p * (1.0 - p);
    return {info, (o.events - o.exposure * p) / std::sqrt(info)};
}
