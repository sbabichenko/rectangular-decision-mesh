#include "fit/dumps.h"
#include "fit/binomial.h"

#include "mesh/rect.h"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// The per-observation dumps format millions of numbers: std::to_chars in the general format at the
// stream's precision produces the same text as operator<< (printf %.{precision}g in both), and
// appending into one buffer per file avoids a stream call per field.
struct RowBuffer {
    std::string text;
    int precision;
    explicit RowBuffer(int precision) : precision(precision) { text.reserve(1 << 20); }
    void number(double v) {
        char buf[64];
        const auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::general, precision);
        text.append(buf, static_cast<std::size_t>(r.ptr - buf));
    }
    void integer(long long v) {
        char buf[32];
        const auto r = std::to_chars(buf, buf + sizeof buf, v);
        text.append(buf, static_cast<std::size_t>(r.ptr - buf));
    }
    void comma() { text.push_back(','); }
    void flush_to(std::ostream& out, bool force) {
        if (force || text.size() > (1u << 20) - 4096) { out.write(text.data(), static_cast<std::streamsize>(text.size())); text.clear(); }
    }
};

std::vector<std::string>& failed_outputs() {
    static std::vector<std::string> failed;
    return failed;
}

CheckedOutput open_output(const std::string& filename) {
    CheckedOutput stream(filename);
    if (!stream) { stream.disarm(); throw std::runtime_error("unable to open output file: " + filename); }
    stream << std::setprecision(10);
    return stream;
}

}  // namespace

CheckedOutput::~CheckedOutput() {
    if (!live_) return;
    flush();
    if (!*this) failed_outputs().push_back(name_);
}

void throw_if_output_failed() {
    auto& failed = failed_outputs();
    if (failed.empty()) return;
    std::string names;
    for (const auto& n : failed) names += (names.empty() ? "" : ", ") + n;
    failed.clear();
    throw std::runtime_error("write failed (disk full or I/O error) on: " + names);
}

void write_candidates(const OutputConfig& output, const std::vector<CandidateRow>& rows, int dim) {
    if (!output.prefix) return;
    auto file = open_output(*output.prefix + "_candidates.csv");
    // z is the gate's coordinate; a three-dimensional mesh's third position is x3
    file << (dim == 3 ? "round,x,y,x3," : "round,x,y,") << "vertex,axis,depth,parent0,parent1,z,sd,beta,gain,selected,admitted\n";
    for (const CandidateRow& r : rows) {
        file << r.round << ',' << r.x << ',' << r.y << ',';
        if (dim == 3) file << r.x3 << ',';
        file << r.vertex << ',' << r.axis << ',' << r.depth << ','
             << r.parent0 << ',' << r.parent1 << ',' << r.z << ',' << r.sd << ',' << r.beta << ',' << r.gain << ','
             << r.selected << ',' << r.admitted << '\n';
    }
}

void write_dumps(const OutputConfig& output,
                 const rect::Mesh& mesh,
                 const std::vector<Observation>& observations,
                 const std::vector<double>& x,
                 const std::vector<double>& y,
                 int pool_count,
                 double pool_variance) {
    if (!output.prefix) return;
    const std::string& base = *output.prefix;

    {
        auto file = open_output(base + "_surface.csv");
        file << (mesh.dim == 3 ? "gx,gy,gz,g_hat\n" : "gx,gy,g_hat\n");
        constexpr int grid_size = 60;
        // three dimensions: the same grid on five slices of the third axis
        const std::vector<double> slices = mesh.dim == 3 ? std::vector<double>{0.0, 0.25, 0.5, 0.75, 1.0} : std::vector<double>{0.0};
        for (double gz : slices)
            for (int a = 0; a < grid_size; ++a) {
                for (int b = 0; b < grid_size; ++b) {
                    const double gx = (a + 0.5) / grid_size;
                    const double gy = (b + 0.5) / grid_size;
                    if (mesh.dim == 3) file << gx << ',' << gy << ',' << gz << ',' << mesh.predict({gx, gy, gz}) << '\n';
                    else file << gx << ',' << gy << ',' << mesh.predict(gx, gy) << '\n';
                }
            }
    }

    {
        auto file = open_output(base + "_obs.csv");
        file << (mesh.dim == 3 ? "x,y,z,pid,n,k,p_hat,resid,info,g_hat,is_train,offset\n" : "x,y,pid,n,k,p_hat,resid,info,g_hat,is_train,offset\n");
        RowBuffer row(10);
        const bool have_mask = mesh.training_mask.size() == observations.size();
        for (std::size_t i = 0; i < observations.size(); ++i) {
            const double fitted = mesh.predict_index(static_cast<int>(i));
            const double probability = logistic(eta_of(observations[i], fitted));
            const auto [information, residual] = pearson_terms(observations[i], probability);
            row.number(x[i]); row.comma(); row.number(y[i]); row.comma();
            if (mesh.dim == 3) { row.number(observations[i].z); row.comma(); }
            row.integer(observations[i].pool_id); row.comma(); row.integer(observations[i].exposure); row.comma();
            row.integer(observations[i].events); row.comma(); row.number(probability); row.comma();
            row.number(residual); row.comma(); row.number(information); row.comma(); row.number(fitted); row.comma();
            row.integer(have_mask ? int(mesh.training_mask[i]) : 1); row.comma();
            row.number(observations[i].offset); row.text.push_back('\n');
            row.flush_to(file, false);
        }
        row.flush_to(file, true);
    }

    {
        auto file = open_output(base + "_hier_vertices.csv");
        file << (mesh.dim == 3 ? "id,x,y,z," : "id,x,y,") << "height,surplus,depth,axis,parent0,parent1,gate_admitted,free_coefficient,backbone,"
                "sigma_sq,sigma_pooled,lambda_v,em_second,current_fit_points,current_xTx,current_keff,"
                "admit_round,admit_beta,admit_sd,admit_z,admit_c,admit_xTx,admit_keff,"
                "admit_gain,admit_signed_root,admit_info_null,"
                "cand_round,cand_beta,cand_sd,cand_z,cand_keff,admit_move,admit_splits,admit_chord_lo,admit_chord_hi\n";
        for (const rect::Vertex& v : mesh.vertices) {
            file << v._id << ',' << v.x << ',' << v.y << ',';
            if (mesh.dim == 3) file << v.z << ',';
            file << v.height << ',' << v.surplus << ','
                 << v.depth << ',' << v.axis << ',' << v.p0 << ',' << v.p1 << ','
                 << (v.gate_admitted ? 1 : 0) << ',' << (v.free_coefficient ? 1 : 0) << ',' << (v.backbone ? 1 : 0) << ','
                 << v.sigma_sq << ',' << v.sigma_pooled << ',' << v.lambda_v << ',' << v.em_second << ','
                 << v.current_fit_points << ',' << v.current_xTx << ',' << v.current_keff << ','
                 << v.admit_round << ',' << v.admit_beta << ',' << v.admit_sd << ',' << v.admit_z << ','
                 << v.admit_c << ',' << v.admit_xTx << ',' << v.admit_keff << ','
                 << v.admit_gain << ',' << v.admit_signed_root << ',' << v.admit_info_null << ','
                 << v.cand_round << ',' << v.cand_beta << ',' << v.cand_sd << ',' << v.cand_z << ',' << v.cand_keff << ','
                 << v.admit_move << ',' << v.admit_splits << ',' << v.admit_chord_lo << ',' << v.admit_chord_hi << '\n';
        }
    }

    {
        auto file = open_output(base + "_mesh.csv");
        // corner k's name spells its side per axis in axis order (h10: x1, y0; h011: x0, y1, z1)
        const int K = mesh.corner_count();
        auto name = [&](int k) { std::string s; for (int d = 0; d < mesh.dim; ++d) s += (k >> d) & 1 ? '1' : '0'; return s; };
        file << (mesh.dim == 3 ? "x0,y0,z0,x1,y1,z1" : "x0,y0,x1,y1");
        for (int k = 0; k < K; ++k) file << ",h" << name(k);
        for (int k = 0; k < K; ++k) file << ",v" << name(k);
        file << ",n_pts\n";
        for (const rect::Cell& c : mesh.cells) {
            if (!c.leaf()) continue;
            for (int d = 0; d < mesh.dim; ++d) file << (d ? "," : "") << static_cast<double>(c.lo[d]) / rect::kSide;
            for (int d = 0; d < mesh.dim; ++d) file << ',' << static_cast<double>(c.hi[d]) / rect::kSide;
            for (int k = 0; k < K; ++k) file << ',' << mesh.vertices[c.corner[k]].height;
            for (int k = 0; k < K; ++k) file << ',' << c.corner[k];
            file << ',' << c.rows.size() << '\n';
        }
    }

    {
        std::vector<double> weighted_error(pool_count, 0.0);
        std::vector<double> information(pool_count, 0.0);
        const double floored_variance = std::max(pool_variance, 1e-4);
        for (std::size_t i = 0; i < observations.size(); ++i) {
            const double fitted = mesh.predict_index(static_cast<int>(i));
            const double probability = logistic(eta_of(observations[i], fitted));
            const double obs_information = observations[i].exposure * probability * (1.0 - probability);
            const double working_residual =
                (observations[i].events - observations[i].exposure * probability) / obs_information;
            const int pool = observations[i].pool_id;
            weighted_error[pool] += obs_information * working_residual;
            information[pool] += obs_information;
        }
        auto file = open_output(base + "_pools.csv");
        file << "pid,u_hat,W\n";
        RowBuffer row(10);
        for (int pool = 0; pool < pool_count; ++pool) {
            if (information[pool] <= 0.0) continue;
            const double estimate = weighted_error[pool] /
                                    (information[pool] + 1.0 / floored_variance);
            row.integer(pool); row.comma(); row.number(estimate); row.comma();
            row.number(information[pool]); row.text.push_back('\n');
            row.flush_to(file, false);
        }
        row.flush_to(file, true);
    }
}
