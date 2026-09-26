// Numerical primitives whose arithmetic order the Python engine fixes: numpy's
// np.linalg.norm, scipy's expit, and the BLAS/LAPACK calls numpy and
// scipy make (the same routines from the same library give the same doubles).
#pragma once
#include <cstddef>
#include <vector>

namespace estimator {

// np.linalg.norm(x) for a 1-d array: sqrt(x.dot(x)) (numpy's fast path, a BLAS dot).
double norm2(const std::vector<double>& x);
// scipy.special.expit: 1/(1+exp(-x)) with libm's exp, every x.
double expit(double x);
double max_abs(const std::vector<double>& x);

// BLAS/LAPACK as numpy/scipy call them. Matrices are row-major p x p.
double ddot(std::size_t n, const double* x, const double* y);
// y = A x (numpy: M @ v, cblas_dgemv row-major no-trans).
void dgemv(std::size_t p, const double* A, const double* x, double* y);
// y = x^T A (numpy: v @ M, cblas_dgemv row-major trans).
void dgemv_t(std::size_t p, const double* A, const double* x, double* y);
// numpy's matmul on a Fortran-ordered operand passes its buffer with a transpose flag: the
// buffer of a Fortran-ordered matrix M is the row-major buffer of M^T. C = At^T B (left
// operand Fortran-ordered), C = A Bt^T (right operand Fortran-ordered), y = At^T x.
void dgemm_tn(std::size_t p, const double* At, const double* B, double* C);
void dgemm_nt(std::size_t p, const double* A, const double* Bt, double* C);
void dgemv_fortran(std::size_t p, const double* At, const double* x, double* y);
// scipy.linalg.cho_factor (upper) in place on a row-major matrix; false if not PD.
bool cho_factor(std::size_t p, double* A);
// scipy.linalg.cho_solve with one right-hand side (in place on b).
void cho_solve(std::size_t p, const double* factor, double* b);
// The same solve on the factor already transposed to column order (transpose_factor): LAPACKE's
// row-major entry transposes the factor on every call; for many solves against one factor that
// is repeated work. LAPACK receives the same data either way.
std::vector<double> transpose_factor(std::size_t p, const double* factor);
void cho_solve_transposed(std::size_t p, const double* factor_t, double* b);
// scipy.linalg.cho_solve(c, np.eye(p)): the inverse, row-major.
void cho_inverse(std::size_t p, const double* factor, double* inverse);
// np.linalg.solve(A, b): false if singular. A is copied; b solved in place.
bool solve(std::size_t p, const double* A, double* b);
// np.linalg.eigvalsh(A).min() (LAPACK dsyevd, lower triangle).
double min_eigenvalue(std::size_t p, const double* A);

}  // namespace estimator
