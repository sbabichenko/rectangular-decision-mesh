#include "fit/dataset.h"
#include "fit/env.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <sstream>
#include <string>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
// The local split's blocks: a median k-d partition down to at most this many pools per block.
constexpr int kSplitBlockSize = 24;


// Column lookup by header name. The four design columns are required under
// their canonical names (aliases accepted); everything else is optional and
// unknown columns are ignored, so a design may carry the pool's attributes for
// the Python side without the binary caring. Values are numeric; text columns
// must not contain commas (the panel writer drops issuer names for that reason).
struct DesignColumns {
    int wala = -1, wac = -1, exposure = -1, events = -1;
    int offset = -1;
    int pool_id = -1;   // rows sharing a value are one pool observed more than once (a panel)
    int axis3 = -1;     // the column DMESH_AXIS3 names: a three-dimensional mesh's third axis
};

DesignColumns parse_header(const std::string& header) {
    DesignColumns c;
    std::vector<std::string> names;
    std::string cell;
    for (std::istringstream h(header); std::getline(h, cell, ',');) {
        cell.erase(std::remove_if(cell.begin(), cell.end(), [](unsigned char ch) { return std::isspace(ch) || ch == '"'; }), cell.end());
        names.push_back(cell);
    }
    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
        const std::string& n = names[i];
        if (n == "wala") c.wala = i;
        else if (n == "wac") c.wac = i;
        else if (n == "n0" || n == "exposure" || n == "n") c.exposure = i;
        else if (n == "k_term" || n == "events" || n == "k") c.events = i;
        else if (n == "offset" || n == "off5") c.offset = i;
        else if (n == "pool_id" || n == "cusip") c.pool_id = i;
    }
    if (const char* axis3 = std::getenv("DMESH_AXIS3"); axis3 && *axis3) {
        for (int i = 0; i < static_cast<int>(names.size()); ++i) if (names[i] == axis3) c.axis3 = i;
        if (c.axis3 < 0) throw std::runtime_error(std::string("DMESH_AXIS3 names a column the design does not have: ") + axis3);
    }
    // Headerless legacy files: the first line is data. Fall back to positions.
    if (c.wala < 0 && c.wac < 0 && names.size() >= 4) {
        bool numeric = true;
        for (const auto& n : names) { char* end = nullptr; std::strtod(n.c_str(), &end); numeric = numeric && end != n.c_str(); }
        if (numeric) { c.wala = 0; c.wac = 1; c.exposure = 2; c.events = 3; }
    }
    if (c.wala < 0 || c.wac < 0 || c.exposure < 0 || c.events < 0)
        throw std::runtime_error("design header must name wala, wac, n0 and k_term (aliases: exposure/n, events/k)");
    return c;
}

Dataset load_real_data(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("cannot open data file: " + path);

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("data file is empty: " + path);
    }
    // A column named "offset" (or "off5") is a per-pool linear-predictor offset; the mesh then
    // models only the residual surface.
    if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF && static_cast<unsigned char>(line[1]) == 0xBB &&
        static_cast<unsigned char>(line[2]) == 0xBF)
        line.erase(0, 3);                                   // a UTF-8 byte order mark before the header
    const DesignColumns cols = parse_header(line);
    const bool headerless = cols.wala == 0 && cols.wac == 1 && cols.exposure == 2 && cols.events == 3 &&
        line.find("wala") == std::string::npos;
    if (headerless) { file.clear(); file.seekg(0); }

    struct Row { double wala, wac, exposure, events, offset, axis3; int pool; };
    std::vector<Row> rows;
    std::vector<std::string> fields;
    std::string cell;
    std::unordered_map<std::string, int> pool_ids;   // pool_id text -> consecutive id, by first appearance
    double total_events = 0.0;
    double total_exposure = 0.0;
    double min_wala = 1e9, max_wala = -1e9;
    double min_wac = 1e9, max_wac = -1e9;
    double min_axis3 = 1e300, max_axis3 = -1e300;
    int line_number = headerless ? 0 : 1;
    auto number = [&](int index, double fallback, const char* what) {
        if (index < 0) return fallback;
        if (index >= static_cast<int>(fields.size()) || fields[index].empty())
            throw std::runtime_error(std::string("row ") + std::to_string(line_number) + " has no " + what + " in " + path);
        // The whole field must be a plain decimal number (digits, sign, point, exponent, or
        // inf/nan): a numeric prefix with trailing text, whitespace, hexadecimal, a nan payload
        // in parentheses, or a non-ASCII digit is not a value. The Python port applies the
        // same rule, so both sides accept and refuse the same fields.
        const std::string& field = fields[index];
        bool plain = !field.empty();
        for (unsigned char ch : field)
            if (ch >= 0x80 || std::isspace(ch) || ch == 'x' || ch == 'X' || ch == '(' || ch == '_') { plain = false; break; }
        char* end = nullptr;
        const double v = plain ? std::strtod(field.c_str(), &end) : 0.0;
        if (!plain || end == field.c_str() || *end != '\0')
            throw std::runtime_error(std::string("row ") + std::to_string(line_number) + " has a non-numeric " + what + " in " + path);
        return v;
    };
    while (std::getline(file, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();   // a CRLF line ending, before the blank test
        if (line.empty()) continue;
        fields.clear();
        for (std::istringstream parser(line); std::getline(parser, cell, ',');) fields.push_back(cell);
        Row r;
        r.wala = number(cols.wala, 0.0, "wala");
        r.wac = number(cols.wac, 0.0, "wac");
        r.exposure = number(cols.exposure, 0.0, "exposure");
        r.events = number(cols.events, 0.0, "events");
        r.offset = number(cols.offset, 0.0, "offset");
        r.axis3 = number(cols.axis3, 0.0, "third-axis value");
        // A row is rejected, never repaired: a non-finite value, a count that is not an
        // integer with n0 >= 1 and 0 <= k_term <= n0, or a blank pool_id where the file has
        // that column (the pool index would otherwise fall outside the pool arrays).
        if (!std::isfinite(r.wala) || !std::isfinite(r.wac) || !std::isfinite(r.exposure) || !std::isfinite(r.events) ||
            !std::isfinite(r.offset) || !std::isfinite(r.axis3))
            throw std::runtime_error("row " + std::to_string(line_number) + " has a non-finite value in " + path);
        if (!(r.exposure >= 1.0) || r.exposure > 2147483647.0 || !(r.events >= 0.0) || r.events > r.exposure ||
            r.exposure != std::floor(r.exposure) || r.events != std::floor(r.events))
            throw std::runtime_error("row " + std::to_string(line_number) + " has invalid counts (need integer 1 <= n0 <= 2147483647 and 0 <= k_term <= n0) in " + path);
        if (cols.pool_id >= 0) {
            if (cols.pool_id >= static_cast<int>(fields.size()) || fields[cols.pool_id].empty())
                throw std::runtime_error("row " + std::to_string(line_number) + " has no pool_id in " + path);
            r.pool = static_cast<int>(pool_ids.emplace(fields[cols.pool_id], static_cast<int>(pool_ids.size())).first->second);
        } else {
            r.pool = static_cast<int>(rows.size());   // one pool per row
        }
        rows.push_back(r);
        total_events += r.events;
        total_exposure += r.exposure;
        min_wala = std::min(min_wala, r.wala);
        max_wala = std::max(max_wala, r.wala);
        min_wac = std::min(min_wac, r.wac);
        max_wac = std::max(max_wac, r.wac);
        min_axis3 = std::min(min_axis3, r.axis3);
        max_axis3 = std::max(max_axis3, r.axis3);
    }
    if (cols.axis3 >= 0 && !(max_axis3 > min_axis3))
        throw std::runtime_error("the third axis (DMESH_AXIS3) is constant in " + path);

    if (rows.empty() || total_exposure <= 0.0 || total_events <= 0.0 ||
        total_events >= total_exposure || max_wala <= min_wala || max_wac <= min_wac) {
        throw std::runtime_error("data file has invalid or degenerate contents: " + path);
    }

    const double pooled_probability = total_events / total_exposure;
    const bool use_offset = cols.offset >= 0;

    Dataset dataset;
    dataset.has_offset = use_offset;
    dataset.bounds = {min_wala, max_wala, min_wac, max_wac};
    if (cols.axis3 >= 0) { dataset.axis3 = std::getenv("DMESH_AXIS3"); dataset.axis3_bounds = {min_axis3, max_axis3}; }
    dataset.observations.reserve(rows.size());
    for (const auto& row : rows) {
        const double x = (row.wala - min_wala) / (max_wala - min_wala);
        const double y = (row.wac - min_wac) / (max_wac - min_wac);
        const int n = static_cast<int>(row.exposure);
        const int k = static_cast<int>(row.events);
        // the baseline and the working data at it are set once the split is known (build_dataset)
        Observation o{x, y, 0.0, 0.0, 0.0, k, n, row.pool, 0.0, row.offset};
        if (cols.axis3 >= 0) o.z = (row.axis3 - min_axis3) / (max_axis3 - min_axis3);
        dataset.observations.push_back(o);
    }
    dataset.pool_count = cols.pool_id >= 0 ? static_cast<int>(pool_ids.size()) : static_cast<int>(rows.size());

    std::fprintf(stderr,
                 "[data] %zu rows, %d pools, all-row rate %.4f (descriptive; the fit's baseline is the training rate, see [split]), WALA [%g,%g], WAC [%g,%g]%s\n",
                 rows.size(), dataset.pool_count, pooled_probability, min_wala, max_wala, min_wac, max_wac,
                 use_offset ? ", offset column" : "");
    return dataset;
}

struct SplitUnit {
    int pool_id = -1;
    double x = 0.0;
    double y = 0.0;
    double exposure = 0.0;
    std::uint64_t tie = 0;
};

void assign_split_mask(const RunConfig& config, Dataset& dataset) {
    const int n = static_cast<int>(dataset.observations.size());
    dataset.training_mask.assign(n, 1);
    if (!config.fit.heldout_split || n == 0) return;

    // External mask replay (DMESH_SPLIT_MASK): one 0/1 per data row, 1 = train.
    // Used to rerun the 2026-08-05 MC-10 diagnostic splits bit-identically.
    if (const char* mask_path = env_cstr("DMESH_SPLIT_MASK")) {
        std::ifstream mask_file(mask_path);
        if (!mask_file) throw std::runtime_error("DMESH_SPLIT_MASK: cannot open file");
        int value = 0, count = 0;
        while (mask_file >> value) {
            if (count >= n) throw std::runtime_error("DMESH_SPLIT_MASK: more entries than rows");
            dataset.training_mask[count++] = (value != 0);
        }
        if (count != n) throw std::runtime_error("DMESH_SPLIT_MASK: entry count != row count");
        return;
    }

    // Split whole pools, not rows (one row per pool on a cross-section,
    // so the unit is then the row). A median k-d partition forms small adaptive spatial blocks;
    // exposure-ranked pairs inside each block are sent to opposite sides.
    const int pool_count = std::max(dataset.pool_count, 0);
    std::vector<double> sx(pool_count, 0.0), sy(pool_count, 0.0), sw(pool_count, 0.0);
    std::vector<std::vector<int>> rows(pool_count);
    for (int i = 0; i < n; ++i) {
        const Observation& o = dataset.observations[i];
        if (o.pool_id < 0 || o.pool_id >= pool_count) continue;
        const double w = std::max(o.exposure, 1);
        sx[o.pool_id] += w * o.x;
        sy[o.pool_id] += w * o.y;
        sw[o.pool_id] += w;
        rows[o.pool_id].push_back(i);
    }

    std::mt19937_64 rng(config.seed);
    std::vector<SplitUnit> units;
    units.reserve(pool_count);
    for (int pool = 0; pool < pool_count; ++pool) {
        if (rows[pool].empty()) continue;
        units.push_back({pool, sx[pool] / sw[pool], sy[pool] / sw[pool], sw[pool], rng()});
    }
    std::vector<int> order(units.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<unsigned char> pool_train(pool_count, 0);
    std::size_t block_count = 0;
    double max_block_exposure_imbalance = 0.0;

    std::function<void(int,int)> split_block = [&](int begin, int end) {
        const int count = end - begin;
        if (count <= kSplitBlockSize) {
            ++block_count;
            std::sort(order.begin() + begin, order.begin() + end,
                      [&](int a, int b) {
                          if (units[a].exposure != units[b].exposure)
                              return units[a].exposure > units[b].exposure;
                          return units[a].tie < units[b].tie;
                      });
            // Greedy equal-cardinality partition minimizes exposure
            // imbalance inside the local block. A random block-wide label
            // flip supplies seeded variation without sacrificing balance.
            const int target_a = (count + 1) / 2;
            const int target_b = count / 2;
            std::vector<unsigned char> side_a(count, 0);
            double exp_a = 0.0, exp_b = 0.0;
            int na = 0, nb = 0;
            for (int j = 0; j < count; ++j) {
                const int u = order[begin + j];
                bool to_a;
                if (na >= target_a) to_a = false;
                else if (nb >= target_b) to_a = true;
                else if (std::fabs(exp_a - exp_b) <= 1e-12) to_a = (rng() & 1ULL) != 0;
                else to_a = exp_a < exp_b;
                side_a[j] = to_a ? 1 : 0;
                if (to_a) { exp_a += units[u].exposure; ++na; }
                else { exp_b += units[u].exposure; ++nb; }
            }
            const bool flip_labels = (rng() & 1ULL) != 0;
            for (int j = 0; j < count; ++j) {
                const int u = order[begin + j];
                const bool train = (side_a[j] != 0) ^ flip_labels;
                pool_train[units[u].pool_id] = train ? 1 : 0;
            }
            const double train_exp = flip_labels ? exp_b : exp_a;
            const double held_exp = flip_labels ? exp_a : exp_b;
            const double total = train_exp + held_exp;
            if (total > 0.0)
                max_block_exposure_imbalance = std::max(
                    max_block_exposure_imbalance,
                    std::fabs(train_exp - held_exp) / total);
            return;
        }

        double xmin = 1.0, xmax = 0.0, ymin = 1.0, ymax = 0.0;
        for (int j = begin; j < end; ++j) {
            const SplitUnit& u = units[order[j]];
            xmin = std::min(xmin, u.x); xmax = std::max(xmax, u.x);
            ymin = std::min(ymin, u.y); ymax = std::max(ymax, u.y);
        }
        const bool x_axis = (xmax - xmin) >= (ymax - ymin);
        const int mid = begin + count / 2;
        std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end,
                         [&](int a, int b) {
                             const double ca = x_axis ? units[a].x : units[a].y;
                             const double cb = x_axis ? units[b].x : units[b].y;
                             if (ca != cb) return ca < cb;
                             const double oa = x_axis ? units[a].y : units[a].x;
                             const double ob = x_axis ? units[b].y : units[b].x;
                             if (oa != ob) return oa < ob;
                             return units[a].tie < units[b].tie;
                         });
        split_block(begin, mid);
        split_block(mid, end);
    };
    split_block(0, static_cast<int>(order.size()));

    for (int pool = 0; pool < pool_count; ++pool)
        for (int i : rows[pool]) dataset.training_mask[i] = pool_train[pool];

    std::fprintf(stderr,
                 "[split] local seed %u, block<=%d, %zu blocks, max block exposure imbalance %.3f\n",
                 config.seed, kSplitBlockSize, block_count,
                 max_block_exposure_imbalance);

    long train_rows = 0, held_rows = 0;
    double train_n = 0.0, held_n = 0.0, train_k = 0.0, held_k = 0.0;
    for (int i = 0; i < n; ++i) {
        const Observation& o = dataset.observations[i];
        if (dataset.training_mask[i]) {
            ++train_rows; train_n += o.exposure; train_k += o.events;
        } else {
            ++held_rows; held_n += o.exposure; held_k += o.events;
        }
    }
    std::fprintf(stderr,
                 "[split] mode local: train %ld rows / %.0f exposure / rate %.6f; "
                 "held %ld rows / %.0f exposure / rate %.6f\n",
                 train_rows, train_n, train_k / std::max(train_n, 1.0),
                 held_rows, held_n, held_k / std::max(held_n, 1.0));
}
}  // namespace

Dataset build_dataset(const RunConfig& config) {
    if (!config.data_file) throw std::runtime_error("DMESH_DATA must name the design file");
    Dataset dataset = load_real_data(*config.data_file);
    assign_split_mask(config, dataset);
    double k = 0.0, n = 0.0;
    for (std::size_t i = 0; i < dataset.observations.size(); ++i) {
        if (!dataset.training_mask[i]) continue;
        k += dataset.observations[i].events;
        n += dataset.observations[i].exposure;
    }
    if (!(n > 0.0)) throw std::runtime_error("split has no training exposure");
    if (config.fit.baseline == "external") {
        // The offset column is the whole baseline logit; the fit adds nothing to it.
        if (!dataset.has_offset)
            throw std::runtime_error("DMESH_BASELINE=external needs an offset column carrying each row's baseline logit");
        dataset.baseline_logit = 0.0;
        dataset.baseline_source = "external";
    } else {
        const double p = std::clamp(k / n, 1e-9, 1.0 - 1e-9);
        dataset.baseline_logit = std::log(p / (1.0 - p));
        dataset.baseline_source = "training";
    }
    for (auto& o : dataset.observations) {
        o.offset = dataset.baseline_logit + o.attribute_offset;
        const double p0 = 1.0 / (1.0 + std::exp(-o.offset));
        const double information = o.exposure * p0 * (1.0 - p0);
        o.working_weight = information;
        o.working_response = (o.events - o.exposure * p0) / information;
    }
    return dataset;
}
