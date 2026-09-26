#pragma once

#include <vector>

enum class LindseyStatus {
    valid,
    empty_input,
    nonfinite_coefficient,
    nonconverged,
    invalid_density,
    mass_out_of_range,
    invalid_central_fit,
};

const char* lindsey_status_name(LindseyStatus status);

struct LfdrResult {
    std::vector<double> lfdr;
    double empirical_mean = 0.0;
    double empirical_sd = 1.0;
    double pi0 = 1.0;
    // True only when both the Poisson density fit and central matching pass
    // numerical validity checks. Callers must fall back to a direct
    // multiple-testing rule when false; an invalid fitted density must never
    // be used to form local FDRs.
    bool used_lindsey = false;
    LindseyStatus status = LindseyStatus::empty_input;
    int iterations = 0;
    double fitted_mass = 0.0;
    double central_curvature = 0.0;
};

[[nodiscard]] double normal_survival(double z);
[[nodiscard]] LfdrResult lindsey_lfdr(std::vector<double> z,
                                      bool free_empirical_null = false);
