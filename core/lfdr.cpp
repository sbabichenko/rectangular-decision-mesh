// Adapted from triangular-decision-mesh core/lfdr.cpp (Lindsey's method
// with clamped empirical null). Defaults here are the production gate
// configuration from run_ginnie_benchmark.sh: sigma0 in [0.3, 6],
// pi0 >= 0.05, |delta0| <= 0.5; env overrides retained.
#include "lfdr.h"
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cmath>

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
double normal_pdf(double z) { return std::exp(-0.5*z*z)/std::sqrt(2.0*kPi); }
void solve_small(int n, double A[8][8], double b[8], double out[8]) {
    for (int c=0;c<n;++c) {
        int piv=c;
        for (int r=c+1;r<n;++r) if (std::fabs(A[r][c])>std::fabs(A[piv][c])) piv=r;
        std::swap(A[c],A[piv]); std::swap(b[c],b[piv]);
        for (int r=c+1;r<n;++r) {
            const double factor=A[r][c]/A[c][c];
            for (int j=c;j<n;++j) A[r][j]-=factor*A[c][j];
            b[r]-=factor*b[c];
        }
    }
    for (int r=n-1;r>=0;--r) {
        double rhs=b[r];
        for (int j=r+1;j<n;++j) rhs-=A[r][j]*out[j];
        out[r]=rhs/A[r][r];
    }
}
}  // namespace

double normal_survival(double z) { return 0.5*std::erfc(z/std::sqrt(2.0)); }

const char* lindsey_status_name(LindseyStatus status) {
    switch (status) {
        case LindseyStatus::valid: return "valid";
        case LindseyStatus::empty_input: return "empty-input";
        case LindseyStatus::nonfinite_coefficient: return "nonfinite-coefficient";
        case LindseyStatus::nonconverged: return "nonconverged";
        case LindseyStatus::invalid_density: return "invalid-density";
        case LindseyStatus::mass_out_of_range: return "mass-out-of-range";
        case LindseyStatus::invalid_central_fit: return "invalid-central-fit";
    }
    return "unknown";
}
LfdrResult lindsey_lfdr(std::vector<double> z, bool free_empirical_null) {
    const int NB = 72, DEG = 6;
    LfdrResult R;
    R.lfdr.assign(z.size(), 1.0);  // fail closed; caller may replace with BH
    R.status = LindseyStatus::empty_input;
    if (z.empty()) return R;
    for (auto& v : z) v = std::clamp(v, -9.0, 9.0);
    double lo = -9.5, hi = 9.5, d = (hi - lo) / NB;
    std::vector<double> counts(NB, 0.0), mids(NB);
    for (int b = 0; b < NB; ++b) mids[b] = lo + (b + 0.5) * d;
    for (double v : z) {
        int b = std::clamp((int)((v - lo) / d), 0, NB - 1);
        counts[b] += 1;
    }
    double mm = 0, ms = 0;
    for (double m : mids) mm += m;
    mm /= NB;
    for (double m : mids) ms += (m - mm) * (m - mm);
    ms = std::sqrt(ms / (NB - 1));
    // basis
    std::vector<std::array<double, 7>> B(NB);
    for (int b = 0; b < NB; ++b) {
        double t = (mids[b] - mm) / ms, p = 1;
        for (int k = 0; k <= DEG; ++k) { B[b][k] = p; p *= t; }
    }
    double beta[8] = {0};
    bool converged = false;
    for (int it = 0; it < 100; ++it) {
        double AtA[8][8] = {{0}}, Atb[8] = {0};
        double maxdiff = 0;
        std::vector<double> mu(NB);
        for (int b = 0; b < NB; ++b) {
            double eta = 0;
            for (int k = 0; k <= DEG; ++k) eta += B[b][k] * beta[k];
            eta = std::clamp(eta, -30.0, 30.0);
            mu[b] = std::exp(eta);
            double zw = eta + (counts[b] - mu[b]) / std::max(mu[b], 1e-10);
            for (int k = 0; k <= DEG; ++k) {
                Atb[k] += B[b][k] * mu[b] * zw;
                for (int l = 0; l <= DEG; ++l) AtA[k][l] += B[b][k] * mu[b] * B[b][l];
            }
        }
        for (int k = 0; k <= DEG; ++k) AtA[k][k] += 1e-8;
        double nb[8];
        solve_small(DEG + 1, AtA, Atb, nb);
        R.iterations = it + 1;
        for (int k = 0; k <= DEG; ++k) {
            if (!std::isfinite(nb[k])) {
                R.status = LindseyStatus::nonfinite_coefficient;
                return R;
            }
            maxdiff = std::max(maxdiff, std::fabs(nb[k] - beta[k]));
            beta[k] = nb[k];
        }
        if (maxdiff < 1e-9) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        R.status = LindseyStatus::nonconverged;
        return R;
    }
    std::vector<double> fbins(NB);
    double fitted_mass = 0.0;
    for (int b = 0; b < NB; ++b) {
        double eta = 0;
        for (int k = 0; k <= DEG; ++k) eta += B[b][k] * beta[k];
        fbins[b] = std::exp(std::clamp(eta, -30.0, 30.0)) / (z.size() * d);
        if (!std::isfinite(fbins[b]) || fbins[b] <= 0.0) {
            R.status = LindseyStatus::invalid_density;
            return R;
        }
        fitted_mass += fbins[b] * d;
    }
    R.fitted_mass = fitted_mass;
    // A Poisson histogram fit should integrate to one after division by n*d.
    // A wide tolerance allows harmless numerical error but rejects the
    // catastrophic nonconverged branch that can otherwise make every lfdr
    // essentially zero.
    if (!(fitted_mass >= 0.98 && fitted_mass <= 1.02)) {
        R.status = LindseyStatus::mass_out_of_range;
        return R;
    }
    // central matching: quadratic fit of log fbins on |mids|<=2
    double A3[8][8] = {{0}}, b3[8] = {0}, th[8];
    for (int b = 0; b < NB; ++b) {
        if (std::fabs(mids[b]) > 2.0) continue;
        double lx = std::log(std::max(fbins[b], 1e-300));
        double xs[3] = {1, mids[b], mids[b] * mids[b]};
        for (int k = 0; k < 3; ++k) {
            b3[k] += xs[k] * lx;
            for (int l = 0; l < 3; ++l) A3[k][l] += xs[k] * xs[l];
        }
    }
    solve_small(3, A3, b3, th);
    double a0 = th[0], a1 = th[1], a2 = th[2];
    R.central_curvature = a2;
    if (!std::isfinite(a0) || !std::isfinite(a1) || !std::isfinite(a2)
        || !(a2 < -1e-10)) {
        R.status = LindseyStatus::invalid_central_fit;
        return R;
    }
    double d0 = 0, s0 = 1, pi0 = 1;
    {
        // Ablation hooks: the [1,1.4] sigma0 box and the 0.5 pi0 floor are the
        // least principled clamps in the pipeline; with a correctly calibrated
        // candidate null they should not be load-bearing.
        // B3 diagnostic: once candidate-specific sandwich scaling has put the
        // exact score on a common null scale, allow central matching to measure
        // the remaining family-level fit-state contraction instead of forcing
        // sigma0 >= 1.  This is opt-in until the null regressions are frozen.
        // Two documented modes (writeup + triangular lfdr.cpp): the
        // non-free mode keeps the central-matching constraint sigma0 >= 1
        // (retraction item 5); the free mode is the B3 box the benchmark
        // config sets (sigma0 in [0.3, 6]) — legitimate only when the
        // family is expected to be fit-state contracted (sandwich-scaled
        // scores in triangular B3; raw underdispersed score families in
        // the rectangular port).
        const double default_smin = free_empirical_null ? 0.3 : 1.0;
        const double default_smax = free_empirical_null ? 6.0 : 1.4;
        // Only the scale box is released in B3.  Keep the conservative
        // location cap and pi0 floor: a coarse underfit can create widespread
        // same-sign signal, and the empirical null must not absorb that signal
        // by drifting its mean or declaring almost the whole family non-null.
        const double default_pi0min = free_empirical_null ? 0.05 : 0.5;
        const double default_dmax = 0.5;
        const double S0MIN = std::getenv("DMESH_SIGMA0_MIN")
            ? std::atof(std::getenv("DMESH_SIGMA0_MIN")) : default_smin;
        const double S0MAX = std::getenv("DMESH_SIGMA0_MAX")
            ? std::atof(std::getenv("DMESH_SIGMA0_MAX")) : default_smax;
        const double PI0MIN = std::getenv("DMESH_PI0_MIN")
            ? std::atof(std::getenv("DMESH_PI0_MIN")) : default_pi0min;
        const double D0MAX = std::getenv("DMESH_DELTA0_MAX")
            ? std::atof(std::getenv("DMESH_DELTA0_MAX")) : default_dmax;
        s0 = std::clamp(std::sqrt(-1.0 / (2 * a2)), S0MIN, S0MAX);
        d0 = std::clamp(a1 * s0 * s0, -D0MAX, D0MAX);
        pi0 = std::clamp(std::exp(a0 + d0 * d0 / (2 * s0 * s0))
                             * std::sqrt(2 * M_PI) * s0,
                         PI0MIN, 1.0);
    }
    R.empirical_mean = d0;
    R.empirical_sd = s0;
    R.pi0 = pi0;
    R.used_lindsey = true;
    R.status = LindseyStatus::valid;
    for (size_t i = 0; i < z.size(); ++i) {
        double fh = fbins[std::clamp((int)((z[i] - lo) / d), 0, NB - 1)];
        double f0 = normal_pdf((z[i] - d0) / s0) / s0;
        R.lfdr[i] = std::clamp(pi0 * f0 / std::max(fh, 1e-300), 0.0, 1.0);
    }
    return R;
}
