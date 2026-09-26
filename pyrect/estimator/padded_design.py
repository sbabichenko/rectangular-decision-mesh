"""Bounded-support row design.

A hierarchical mesh design row touches only the coefficients whose support
covers its point: about two per depth level, at most a few dozen, independent
of the coefficient total. Storing each row as a fixed-width list of (column,
value) pairs makes every product the fit needs cost O(rows * width) or
O(rows * width^2) rather than O(rows * columns) or O(rows * columns^2), and
never forms a rows-by-columns matrix. Dense NumPy designs still work through
the same helpers, unchanged in arithmetic, so existing callers are unaffected.
"""
import numpy as np
from scipy.sparse import csr_matrix, eye as sparse_eye, issparse


class PaddedDesign:
    """Rows as (index, value) pairs padded to a common width; padding is zero."""

    def __init__(self, index, value, columns):
        index = np.asarray(index)
        value = np.asarray(value, float)
        if index.shape != value.shape or index.ndim != 2:
            raise ValueError('index and value must be rows-by-width arrays')
        if not np.isfinite(value).all():
            raise ValueError('finite design values required')
        if index.size and (index.min() < 0 or index.max() >= columns):
            raise ValueError('column index out of range')
        # 32-bit column numbers, as the binary's (a design has far fewer than 2^31 columns): half the
        # index memory of intp; only lookups read them, so no arithmetic changes
        self.index = np.ascontiguousarray(index, dtype=np.int32)
        self.value = np.ascontiguousarray(value)
        self.shape = (int(index.shape[0]), int(columns))
        self.width = int(index.shape[1])

    @classmethod
    def from_csr(cls, matrix):
        matrix = csr_matrix(matrix)
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        rows, columns = matrix.shape
        counts = np.diff(matrix.indptr)
        width = int(counts.max()) if rows else 0
        index = np.zeros((rows, width), dtype=np.intp)
        value = np.zeros((rows, width))
        starts = matrix.indptr[:-1]
        for j in range(width):
            take = counts > j
            index[take, j] = matrix.indices[starts[take]+j]
            value[take, j] = matrix.data[starts[take]+j]
        return cls(index, value, columns)


    def rows(self, rows):
        return PaddedDesign(self.index[rows], self.value[rows], self.shape[1])

    def digest(self, hasher):
        for array in (self.index, self.value):
            hasher.update(memoryview(np.ascontiguousarray(array)).cast('B'))
            hasher.update(str(array.shape).encode())
        hasher.update(str(self.shape).encode())

    def matvec(self, beta):
        beta = np.ascontiguousarray(beta, dtype=float)
        return np.einsum('rk,rk->r', self.value, beta[self.index])

    def rmatvec(self, vector):
        vector = np.ascontiguousarray(vector, dtype=float)
        return np.bincount(self.index.ravel(), weights=(self.value*vector[:, None]).ravel(),
                           minlength=self.shape[1])

    def weighted_gram(self, weights):
        """sum_i w_i x_i x_i^T, as a sparse product (agrees with the C++'s column-pair loops to
        rounding; the fits and held-out numbers are unchanged, measured 2026-09-22)."""
        weights = np.ascontiguousarray(weights, dtype=float)
        rows, width = self.index.shape
        indptr = np.arange(0, rows*width+1, width)
        X = csr_matrix((self.value.ravel(), self.index.ravel(), indptr), shape=self.shape)
        Xw = csr_matrix(((self.value*weights[:, None]).ravel(), self.index.ravel(), indptr), shape=self.shape)
        return (X.T@Xw).toarray()

    def quadratic_forms(self, matrix, block=4096):
        """x_i^T M x_i for every row, gathering only the row's own block of M."""
        matrix = np.ascontiguousarray(matrix, dtype=float)
        out = np.empty(self.shape[0])
        for start in range(0, self.shape[0], block):
            sl = slice(start, start+block)
            idx, val = self.index[sl], self.value[sl]
            out[sl] = np.einsum('rk,rl,rkl->r', val, val, matrix[idx[:, :, None], idx[:, None, :]])
        return out


def structural_inverse(transform, depth_limit=None):
    """Inverse of I-L for a nilpotent L (child rows subtract ancestor columns).

    Returns a sparse matrix with exactly the structural fill of the Neumann
    series I+L+L^2+..., without forming a dense inverse.
    """
    transform = csr_matrix(transform)
    p = transform.shape[0]
    L = (sparse_eye(p, format='csr')-transform).tocsr()
    L.eliminate_zeros()
    result = sparse_eye(p, format='csr')
    power = sparse_eye(p, format='csr')
    limit = p if depth_limit is None else depth_limit
    for _ in range(limit):
        power = (power@L).tocsr()
        power.eliminate_zeros()
        if power.nnz == 0:
            # canonical (sorted) columns: a product with this inverse then sums each row's terms
            # in column order, the order the C++ core uses (scipy's additions above leave the
            # columns in insertion order)
            result = result.tocsr()
            result.sort_indices()
            return result
        result = result+power
    raise ValueError('transform is not I minus a nilpotent matrix')


# Dispatch helpers: dense arrays keep their existing arithmetic exactly.
def is_padded(X):
    return isinstance(X, PaddedDesign)


def matvec(X, beta):
    return X.matvec(beta) if is_padded(X) else np.asarray(X)@np.asarray(beta, float)


def rmatvec(X, vector):
    return X.rmatvec(vector) if is_padded(X) else np.asarray(X).T@np.asarray(vector, float)


def take_rows(X, rows):
    return X.rows(rows) if is_padded(X) else np.asarray(X)[rows]


def digest(X, hasher):
    if is_padded(X):
        X.digest(hasher)
    else:
        array = np.ascontiguousarray(X)
        hasher.update(memoryview(array).cast('B'))
        hasher.update(str(array.shape).encode())


def as_design(X):
    """Accept dense arrays, SciPy sparse matrices or padded designs."""
    if is_padded(X):
        return X
    if issparse(X):
        return PaddedDesign.from_csr(X)
    return np.asarray(X, float)


