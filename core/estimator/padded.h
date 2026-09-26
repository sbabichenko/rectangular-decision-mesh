// Bounded-support row design: each row is a fixed-width list of (column, value) pairs, padding
// zero. pymesh/estimator/padded_design.py computes the same products (the gram as a sparse
// product, equal to rounding).
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace estimator {

struct PaddedDesign {
    std::size_t rows = 0, columns = 0;
    int width = 0;
    using Index = int32_t;        // a column number; the designs have far fewer than 2^31 columns
    std::vector<Index> index;     // rows*width
    std::vector<double> value;    // rows*width

    PaddedDesign() = default;
    PaddedDesign(std::size_t rows, int width, std::size_t columns)
        : rows(rows), columns(columns), width(width), index(rows * width, 0), value(rows * width, 0.) {
        if (columns > static_cast<std::size_t>(std::numeric_limits<Index>::max())) throw std::invalid_argument("too many design columns");
    }

    // PaddedDesign.from_csr: sorted columns per row, no explicit zeros, width = the longest row.
    static PaddedDesign from_sparse_rows(const std::vector<std::vector<std::pair<int64_t, double>>>& sparse_rows,
                                         std::size_t columns);
    PaddedDesign take_rows(const std::vector<int64_t>& which) const;
    PaddedDesign with_extra_column(const std::vector<int64_t>& extra_index, const std::vector<double>& extra_value,
                                   std::size_t new_columns) const;
    // several at once: the layout of successive with_extra_column calls (the extras in order after
    // the row's entries) in one copy, not one per column
    PaddedDesign with_extra_columns(const std::vector<std::vector<int64_t>>& extra_index,
                                    const std::vector<std::vector<double>>& extra_value, std::size_t new_columns) const;

    void matvec(const double* beta, double* out, int threads) const;
    void rmatvec(const double* vector, double* out) const;          // out zeroed here
    // rmatvec of the given rows only (vector[r] for rows[r]), in that order: take_rows(rows).rmatvec
    // without the copy
    void rmatvec_rows(const std::vector<int64_t>& rows, const double* vector, double* out) const;
    // out zeroed here, columns x columns row-major. Each output row belongs to one thread, so
    // every entry accumulates its rows in the same order whatever the thread count.
    void weighted_gram(const double* weights, double* out, int threads = 1) const;
    void quadratic_forms(const double* matrix, double* out, int threads) const;
    bool same_as(const PaddedDesign& other) const;
};

}  // namespace estimator
