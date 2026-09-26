#include "estimator/padded.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>

namespace estimator {

PaddedDesign PaddedDesign::from_sparse_rows(const std::vector<std::vector<std::pair<int64_t, double>>>& sparse_rows,
                                            std::size_t columns) {
    // csr_matrix(...).sum_duplicates(): sort by column, sum equal columns in order;
    // eliminate_zeros(): drop exact zeros. The rows are cleaned one at a time into a design
    // padded to the longest raw row, then narrowed if cleaning shortened every longest row.
    int bound = 0;
    for (const auto& row : sparse_rows) bound = std::max(bound, static_cast<int>(row.size()));
    PaddedDesign design(sparse_rows.size(), bound, columns);
    std::vector<std::pair<int64_t, double>> row;
    int width = 0;
    for (std::size_t i = 0; i < sparse_rows.size(); ++i) {
        row = sparse_rows[i];
        std::stable_sort(row.begin(), row.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        Index* idx = design.index.data() + i * static_cast<std::size_t>(bound);
        double* val = design.value.data() + i * static_cast<std::size_t>(bound);
        int a = 0;
        for (const auto& [c, v] : row) {
            if (a > 0 && idx[a - 1] == c) val[a - 1] += v;
            else { idx[a] = static_cast<Index>(c); val[a] = v; ++a; }
        }
        int kept = 0;
        for (int b = 0; b < a; ++b) if (val[b] != 0.) { idx[kept] = idx[b]; val[kept] = val[b]; ++kept; }
        for (int b = kept; b < a; ++b) { idx[b] = 0; val[b] = 0.; }
        width = std::max(width, kept);
    }
    if (width == bound) return design;
    PaddedDesign narrow(sparse_rows.size(), width, columns);
    for (std::size_t i = 0; i < sparse_rows.size(); ++i) {
        std::copy_n(design.index.data() + i * static_cast<std::size_t>(bound), width, narrow.index.data() + i * static_cast<std::size_t>(width));
        std::copy_n(design.value.data() + i * static_cast<std::size_t>(bound), width, narrow.value.data() + i * static_cast<std::size_t>(width));
    }
    return narrow;
}

PaddedDesign PaddedDesign::take_rows(const std::vector<int64_t>& which) const {
    PaddedDesign out(which.size(), width, columns);
    for (std::size_t r = 0; r < which.size(); ++r) {
        std::memcpy(&out.index[r * width], &index[static_cast<std::size_t>(which[r]) * width], sizeof(Index) * width);
        std::memcpy(&out.value[r * width], &value[static_cast<std::size_t>(which[r]) * width], sizeof(double) * width);
    }
    return out;
}

PaddedDesign PaddedDesign::with_extra_column(const std::vector<int64_t>& extra_index, const std::vector<double>& extra_value,
                                             std::size_t new_columns) const {
    PaddedDesign out(rows, width + 1, new_columns);
    for (std::size_t r = 0; r < rows; ++r) {
        std::memcpy(&out.index[r * (width + 1)], &index[r * width], sizeof(Index) * width);
        std::memcpy(&out.value[r * (width + 1)], &value[r * width], sizeof(double) * width);
        out.index[r * (width + 1) + width] = static_cast<Index>(extra_index[r]);
        out.value[r * (width + 1) + width] = extra_value[r];
    }
    return out;
}

PaddedDesign PaddedDesign::with_extra_columns(const std::vector<std::vector<int64_t>>& extra_index,
                                              const std::vector<std::vector<double>>& extra_value, std::size_t new_columns) const {
    const int F = static_cast<int>(extra_index.size());
    PaddedDesign out(rows, width + F, new_columns);
    const std::size_t w = static_cast<std::size_t>(width + F);
    for (std::size_t r = 0; r < rows; ++r) {
        std::memcpy(&out.index[r * w], &index[r * width], sizeof(Index) * width);
        std::memcpy(&out.value[r * w], &value[r * width], sizeof(double) * width);
        for (int f = 0; f < F; ++f) {
            out.index[r * w + width + f] = static_cast<Index>(extra_index[f][r]);
            out.value[r * w + width + f] = extra_value[f][r];
        }
    }
    return out;
}

void PaddedDesign::matvec(const double* beta, double* out, int threads) const {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (long i = 0; i < static_cast<long>(rows); ++i) {
        const Index* idx = index.data() + static_cast<std::size_t>(i) * width;
        const double* val = value.data() + static_cast<std::size_t>(i) * width;
        double total = 0.;
        for (int a = 0; a < width; ++a) total += val[a] * beta[idx[a]];
        out[i] = total;
    }
}

void PaddedDesign::rmatvec(const double* vector, double* out) const {
    std::fill(out, out + columns, 0.);
    for (std::size_t i = 0; i < rows; ++i) {
        const Index* idx = index.data() + i * width;
        const double* val = value.data() + i * width;
        const double v = vector[i];
        for (int a = 0; a < width; ++a) out[idx[a]] += val[a] * v;
    }
}

void PaddedDesign::rmatvec_rows(const std::vector<int64_t>& subset, const double* vector, double* out) const {
    std::fill(out, out + columns, 0.);
    for (std::size_t r = 0; r < subset.size(); ++r) {
        const std::size_t i = static_cast<std::size_t>(subset[r]);
        const Index* idx = index.data() + i * width;
        const double* val = value.data() + i * width;
        const double v = vector[r];
        for (int a = 0; a < width; ++a) out[idx[a]] += val[a] * v;
    }
}

void PaddedDesign::weighted_gram(const double* weights, double* out, int threads) const {
    std::fill(out, out + columns * columns, 0.);
    if (threads <= 1 || columns < 64) {
        for (std::size_t i = 0; i < rows; ++i) {
            const Index* idx = index.data() + i * width;
            const double* val = value.data() + i * width;
            const double w = weights[i];
            for (int a = 0; a < width; ++a) {
                const double wa = w * val[a];
                if (wa == 0.) continue;
                double* row = out + static_cast<std::size_t>(idx[a]) * columns;
                for (int b = 0; b < width; ++b) row[idx[b]] += wa * val[b];
            }
        }
        return;
    }
    // Thread t owns the output rows j with j % threads == t. Every thread scans all rows in
    // order and touches only its own output rows, so no entry's summation order changes.
    #pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num(), T = omp_get_num_threads();
        for (std::size_t i = 0; i < rows; ++i) {
            const Index* idx = index.data() + i * width;
            const double* val = value.data() + i * width;
            const double w = weights[i];
            for (int a = 0; a < width; ++a) {
                if (idx[a] % T != t) continue;
                const double wa = w * val[a];
                if (wa == 0.) continue;
                double* row = out + static_cast<std::size_t>(idx[a]) * columns;
                for (int b = 0; b < width; ++b) row[idx[b]] += wa * val[b];
            }
        }
    }
}

void PaddedDesign::quadratic_forms(const double* matrix, double* out, int threads) const {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (long i = 0; i < static_cast<long>(rows); ++i) {
        const Index* idx = index.data() + static_cast<std::size_t>(i) * width;
        const double* val = value.data() + static_cast<std::size_t>(i) * width;
        double total = 0.;
        for (int a = 0; a < width; ++a) {
            const double* row = matrix + static_cast<std::size_t>(idx[a]) * columns;
            double inner = 0.;
            for (int b = 0; b < width; ++b) inner += row[idx[b]] * val[b];
            total += val[a] * inner;
        }
        out[i] = total;
    }
}

bool PaddedDesign::same_as(const PaddedDesign& other) const {
    return rows == other.rows && columns == other.columns && width == other.width && index == other.index && value == other.value;
}

}  // namespace estimator
