// Pool-effect quadrature: the moments, the zero-variance exact branch and the row-level mode
// solve of pymesh/estimator/pool.py.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace estimator {

constexpr int kEnumerationLimit = 128;
constexpr double kEnumerationPrune = 1e-18;   // counts outside the contiguous range above this share of a pool's weight are skipped

struct Moments {
    std::vector<double> mean, second, weight, sensitivity, mean_derivative, information, a1, a2, location;
    std::size_t size() const { return mean.size(); }
    void resize(std::size_t n);
    void assign_rows(const std::vector<int64_t>& rows, const Moments& source);   // this[rows] = source (row-aligned)
};

// pool_quadrature_moments: output columns per row and a status (0 ok, larger worse).
int pool_quadrature_moments(int rows, const double* n, const double* a, const double* a1, double variance,
                            int count_order, int pool_order, const double* gh_nodes, const double* gh_weights,
                            int threads, double* output);
int mean_neutral_rows(int rows, const double* eta, const double* sd, int order, const double* gh_nodes,
                      const double* gh_weights, int threads, double* output);

// pool.moments(eta, n, variance, count_order, pool_order): throws on a kernel failure.
Moments moments(const std::vector<double>& eta, const std::vector<double>& n, double variance, int count_order,
                int pool_order, int threads);
// pool._scaled_mode: (residual, effective) with the vectorised iteration's global stop.
void scaled_mode(const std::vector<double>& a, const std::vector<double>& n, const std::vector<double>& k,
                 double variance, std::vector<double>& residual, std::vector<double>& effective);
int pool_order_for(double variance, int base);
int surface_count_order();
int neutral_order_for(double variance);   // Gauss-Hermite nodes for the mean-neutral location
int threads_setting();

}  // namespace estimator
