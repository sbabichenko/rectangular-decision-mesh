#pragma once

#include "fit/observation.h"
#include "fit/config.h"

#include <array>
#include <string>
#include <vector>

struct Dataset {
    std::vector<Observation> observations;
    int pool_count = 0;
    double baseline_logit = 0.0;
    bool has_offset = false;                  // the design carried an offset column
    std::string baseline_source = "training"; // "training" or "external" (DMESH_BASELINE)
    // Covariate-only domain, fixed across splits; no outcomes enter these bounds.
    std::array<double, 4> bounds{}; // wala min/max, wac min/max
    // A three-dimensional mesh: DMESH_AXIS3 names the design column of the third axis, scaled to
    // the unit interval by its own range as wala and wac are; empty for the usual two dimensions.
    std::string axis3;
    std::array<double, 2> axis3_bounds{};
    int dim() const { return axis3.empty() ? 2 : 3; }
    // 1 for fitting rows, 0 for held-out rows. Empty only before split
    // assignment; refine() always receives a mask of observations.size().
    std::vector<unsigned char> training_mask;
};

Dataset build_dataset(const RunConfig& config);
