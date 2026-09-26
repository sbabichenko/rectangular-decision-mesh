#include "fit/gate.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

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

// Retry only failed full-step IRLS fits, on the same ridge-Poisson objective.
bool damped_poisson_retry(const std::vector<std::array<double, 7>>& B,
                          const std::vector<double>& counts, double beta[8], int& iterations) {
    constexpr int size = 7;
    constexpr double ridge = 1e-8;
    std::fill(beta, beta + 8, 0.0);
    double total = 0;
    for (double count : counts) total += count;
    beta[0] = std::log(total / counts.size());
    auto objective = [&](const double* b) {
        double norm = 0;
        for (int k = 0; k < size; ++k) norm += b[k] * b[k];
        double value = 0.5 * ridge * norm;
        for (std::size_t i = 0; i < B.size(); ++i) {
            double eta = 0;
            for (int k = 0; k < size; ++k) eta += B[i][k] * b[k];
            if (!std::isfinite(eta) || std::fabs(eta) > 700)
                return std::numeric_limits<double>::infinity();
            value += std::exp(eta) - counts[i] * eta;
        }
        return value;
    };
    for (int it = 0; it < 100; ++it) {
        ++iterations;
        double H[8][8] = {{0}}, score[8] = {0}, rhs[8] = {0}, delta[8] = {0};
        for (std::size_t i = 0; i < B.size(); ++i) {
            double eta = 0;
            for (int k = 0; k < size; ++k) eta += B[i][k] * beta[k];
            const double mu = std::exp(eta);
            for (int k = 0; k < size; ++k) {
                score[k] += B[i][k] * (counts[i] - mu);
                for (int j = 0; j < size; ++j) H[k][j] += B[i][k] * mu * B[i][j];
            }
        }
        for (int k = 0; k < size; ++k) {
            H[k][k] += ridge;
            score[k] -= ridge * beta[k];
            rhs[k] = score[k];
        }
        solve_small(size, H, rhs, delta);
        double decrement = 0;
        for (int k = 0; k < size; ++k) {
            if (!std::isfinite(delta[k])) return false;
            decrement += score[k] * delta[k];
        }
        if (!std::isfinite(decrement) || decrement < 0) return false;
        if (decrement <= 1e-10) return true;
        const double value = objective(beta);
        bool accepted = false;
        for (double step = 1; step >= std::ldexp(1.0, -30); step *= 0.5) {
            double trial[8];
            for (int k = 0; k < 8; ++k) trial[k] = beta[k] + step * delta[k];
            if (objective(trial) <= value - 1e-4 * step * decrement) {
                std::copy(trial, trial + 8, beta);
                accepted = true;
                break;
            }
        }
        if (!accepted) return false;
    }
    return false;
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
    R.lfdr.assign(z.size(), 1.0);  // invalid nulls defer admissions
    R.status = LindseyStatus::empty_input;
    if (z.empty()) return R;
    for (auto& v : z) v = std::clamp(v, -9.0, 9.0);
    double lo = -9.5, hi = 9.5, d = (hi - lo) / NB;
    std::vector<double> counts(NB, 0.0), mids(NB);
    for (int b = 0; b < NB; ++b) mids[b] = lo + (b + 0.5) * d;
    for (double v : z) {
        // Linear binning: the unit is split between the two bins whose midpoints bracket v, in
        // proportion to distance, so the counts (and everything fitted from them) move
        // continuously with v. With whole-bin counts a score crossing an edge moved a full count,
        // which could shift the null's sd by 0.3% and reorder near-tied candidates (2026-09-22).
        const double t = (v - lo) / d - 0.5;                  // position in bin-midpoint units
        const int b0 = std::clamp((int)std::floor(t), 0, NB - 2);
        const double w = std::clamp(t - b0, 0.0, 1.0);
        counts[b0] += 1 - w;
        counts[b0 + 1] += w;
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
    if (!converged && !damped_poisson_retry(B, counts, beta, R.iterations)) {
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
        // The scale box and the pi0 floor are the least principled clamps in
        // the pipeline; with a correctly calibrated candidate null they should
        // not be load-bearing.
        // The empirical null's box. The upper scale bound (6) and the pi0
        // floor (0.05) are the values every launcher set (DMESH_SIGMA0_MAX,
        // DMESH_PI0_MIN, baked September 2026). The sigma0 floor is the
        // code's own default: 0.35 when central matching is free (every
        // caller), 1.0 otherwise. The July launchers overrode it with
        // DMESH_SIGMA0_MIN (0.3 on the benchmark, 1.0 on the SMM launcher);
        // the September promotion returned to this default deliberately
        // (measured: the 1.0 floor costs 0.02 held-out deviance on SMM under
        // the no-term gate).
        const double S0MIN = free_empirical_null ? 0.35 : 1.0;
        // 3 rather than the triangular engine's 6 (2026-09-23): when nearly every candidate carries
        // signal, as in the first round of a fit far from its coarse start (the SMM design, seed 1),
        // the central fit takes the signal for a null of sd 4 and the gate admits nothing until it
        // stalls. The cap bound on no other fit of the year battery, the cohort split or the Ginnie
        // design (their results are identical).
        const double S0MAX = 3.0;
        const double PI0MIN = 0.05;
        const double D0MAX = 0.5;
        s0 = std::clamp(std::sqrt(-1.0 / (2 * a2)), S0MIN, S0MAX);
        d0 = std::clamp(a1 * s0 * s0, -D0MAX, D0MAX);
        pi0 = std::clamp(std::exp(a0 + d0 * d0 / (2 * s0 * s0))
                             * std::sqrt(2 * kPi) * s0,
                         PI0MIN, 1.0);
    }
    R.empirical_mean = d0;
    R.empirical_sd = s0;
    R.pi0 = pi0;
    R.used_lindsey = true;
    R.status = LindseyStatus::valid;
    for (size_t i = 0; i < z.size(); ++i) {
        // the fitted density at z itself (the same polynomial the bins sample), not its bin's value
        const double t = (std::clamp(z[i], -9.0, 9.0) - mm) / ms;
        double eta = 0, p = 1;
        for (int k = 0; k <= DEG; ++k) { eta += p * beta[k]; p *= t; }
        const double fh = std::exp(std::clamp(eta, -30.0, 30.0)) / (z.size() * d);
        double f0 = normal_pdf((z[i] - d0) / s0) / s0;
        R.lfdr[i] = std::clamp(pi0 * f0 / std::max(fh, 1e-300), 0.0, 1.0);
    }
    return R;
}
