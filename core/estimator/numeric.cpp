#include "estimator/numeric.h"

#include <algorithm>
#include <cmath>
#include <cblas.h>
#include <lapacke.h>

namespace estimator {

double norm2(const std::vector<double>& x) {
    // np.linalg.norm's 1-d fast path: sqrt(x.dot(x)), a BLAS dot
    return std::sqrt(ddot(x.size(), x.data(), x.data()));
}

double expit(double x) { return 1.0 / (1.0 + std::exp(-x)); }

double max_abs(const std::vector<double>& x) {
    double m = 0.;
    for (double v : x) m = std::max(m, std::fabs(v));
    return m;
}

double ddot(std::size_t n, const double* x, const double* y) {
    return cblas_ddot(static_cast<int>(n), x, 1, y, 1);
}

void dgemv(std::size_t p, const double* A, const double* x, double* y) {
    cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(p), static_cast<int>(p), 1.0, A, static_cast<int>(p), x, 1, 0.0, y, 1);
}

void dgemv_t(std::size_t p, const double* A, const double* x, double* y) {
    cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(p), static_cast<int>(p), 1.0, A, static_cast<int>(p), x, 1, 0.0, y, 1);
}

void dgemm_tn(std::size_t p, const double* At, const double* B, double* C) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, static_cast<int>(p), static_cast<int>(p), static_cast<int>(p),
                1.0, At, static_cast<int>(p), B, static_cast<int>(p), 0.0, C, static_cast<int>(p));
}

void dgemm_nt(std::size_t p, const double* A, const double* Bt, double* C) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, static_cast<int>(p), static_cast<int>(p), static_cast<int>(p),
                1.0, A, static_cast<int>(p), Bt, static_cast<int>(p), 0.0, C, static_cast<int>(p));
}

void dgemv_fortran(std::size_t p, const double* At, const double* x, double* y) {
    cblas_dgemv(CblasRowMajor, CblasTrans, static_cast<int>(p), static_cast<int>(p), 1.0, At, static_cast<int>(p), x, 1, 0.0, y, 1);
}

bool cho_factor(std::size_t p, double* A) {
    // scipy hands LAPACK a Fortran-ordered copy and factors 'U'; the matrix is symmetric, so
    // a row-major 'U' factorisation reads the same triangle in the same order.
    return LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', static_cast<int>(p), A, static_cast<int>(p)) == 0;
}

void cho_solve(std::size_t p, const double* factor, double* b) {
    LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', static_cast<int>(p), 1, factor, static_cast<int>(p), b, 1);
}

std::vector<double> transpose_factor(std::size_t p, const double* factor) {
    std::vector<double> out(p * p);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j) out[j * p + i] = factor[i * p + j];
    return out;
}

void cho_solve_transposed(std::size_t p, const double* factor_t, double* b) {
    LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', static_cast<int>(p), 1, factor_t, static_cast<int>(p), b, static_cast<int>(p));
}

void cho_inverse(std::size_t p, const double* factor, double* inverse) {
    std::fill(inverse, inverse + p * p, 0.);
    for (std::size_t i = 0; i < p; ++i) inverse[i * p + i] = 1.;
    // scipy: dpotrs on the Fortran factor with the identity as p right-hand sides. LAPACKE's
    // row-major entry transposes the factor and the identity into that same call and
    // transposes the solution back, so inverse[i][j] is scipy's X[i, j].
    LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'U', static_cast<int>(p), static_cast<int>(p), factor, static_cast<int>(p), inverse, static_cast<int>(p));
}

bool solve(std::size_t p, const double* A, double* b) {
    // numpy linearises A into Fortran order and calls dgesv.
    std::vector<double> col(p * p);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j) col[j * p + i] = A[i * p + j];
    std::vector<int> ipiv(p);
    return LAPACKE_dgesv(LAPACK_COL_MAJOR, static_cast<int>(p), 1, col.data(), static_cast<int>(p), ipiv.data(), b, static_cast<int>(p)) == 0;
}

double min_eigenvalue(std::size_t p, const double* A) {
    std::vector<double> copy(A, A + p * p), w(p);
    // numpy: dsyevd, UPLO 'L' on the Fortran-ordered (i.e. transposed) array
    LAPACKE_dsyevd(LAPACK_COL_MAJOR, 'N', 'L', static_cast<int>(p), copy.data(), static_cast<int>(p), w.data());
    return w.empty() ? 0. : w[0];
}

}  // namespace estimator
