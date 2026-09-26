// The refinement driver. It builds the coarse mesh, runs the gate rounds (score the candidate
// frontier against the frozen fit, calibrate an empirical null, admit the selected prefix),
// refits, and writes the report and the dumps. Every fit is the engine's. The triangular
// driver's rounds, null and selection, with the rectangular mesh's candidates: an existing
// constrained vertex, or the midpoint of a leaf edge whose admission splits the cells around it
// (a hanging vertex's release is the same move). Splits whose new vertices are constrained leave
// the surface unchanged, so admitting a candidate changes the design by its one column.
#include "fit/refine.h"

#include "fit/binomial.h"
#include "estimator/engine.h"
#include "estimator/profile.h"
#include "fit/gate.h"
#include "fit/observation.h"
#include "fit/dumps.h"
#include "fit/units.h"
#include "mesh/rect.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using estimator::CandidateCensus;
using estimator::CandidateGain;
using estimator::Engine;
using estimator::ScoreContext;

namespace {

// The refinement budget and the stopping rules of the gate rounds.
constexpr int kPreRefineLevels = 3;          // the coarse geometry: an 8 x 8 grid of cells
constexpr int kPreRefineLevelsAxis3 = 2;     // three dimensions: 8 x 8 x 4, the third axis starting coarser
constexpr int kBackboneDepth = 1;            // pre-refined vertices this deep start as free coefficients
constexpr int kMinimumCellCap = 600;         // cell budget: max(this, N / kRowsPerCell)
constexpr int kRowsPerCell = 24;             // about one cell per this many rows: the data-density ceiling
constexpr int kMaxRounds = 80;
constexpr int kStallLimit = 3;               // consecutive rounds admitting nothing
constexpr int kRefreshPeriod = 3;            // rounds between refits of the whole surface
constexpr int kLindseyMinimumCandidates = 50; // below this the round uses direct BH

// One round's scored candidates, one row per scoreable candidate.
struct RoundCandidates {
    std::vector<rect::Candidate> candidates;
    std::vector<double> z;                 // the calibrated gate coordinate
    std::vector<double> z_sd;              // its calibration scale
    std::vector<CandidateGain> exact;

    void push(const rect::Candidate& candidate, double z_value, double sd, const CandidateGain& profile) {
        candidates.push_back(candidate);
        z.push_back(z_value);
        z_sd.push_back(sd);
        exact.push_back(profile);
    }
    std::size_t size() const { return candidates.size(); }
};

// Pre-refinement builds the coarse geometry and frees the roots and the depth-1 vertices (a
// bilinear surface and its midline corrections), as the triangular engine frees its level and
// tilt. A depth-2 backbone interpolated better on the Ginnie design (-0.57%) but doubled the
// held-out deviance when the newest cohorts are held out (the cohort split, February 2020: 3.40
// against 1.57): free quarter-scale heights at the edge of the data extrapolate steeply. The
// deeper pre-refined vertices stay constrained
// at zero surplus on the candidate frontier and earn a coefficient only by passing the gate.
void pre_refine_mesh(rect::Mesh& mesh) {
    if (mesh.dim == 3) mesh.refine_uniform(std::array<int, rect::kMaxDim>{kPreRefineLevels, kPreRefineLevels, kPreRefineLevelsAxis3});
    else mesh.refine_uniform(kPreRefineLevels);
    for (rect::Vertex& v : mesh.vertices) {
        if (v.has_prior() && v.depth > kBackboneDepth) continue;
        v.free_coefficient = true;
        v.backbone = true;
    }
    ++mesh.coefficient_version;
    mesh.recompute_heights();
}

bool height_is_valid(double height) { return std::isfinite(height) && std::fabs(height) <= kHeightLimitUnits; }

// Admit the selected candidates, best first. order holds the selected prefix, indexing into
// scored; c_round is the round's admission scale. Each candidate is re-profiled here rather than
// trusted from selection, because admitting one can move the surface under another: the profile
// is recomputed at the actual commit state, on the realised geometry.
// Returns how many were admitted.
int admit_selected_candidates(int round, int kk, const std::vector<int>& order, const RoundCandidates& scored,
                              double c_round, rect::Mesh& mesh, Engine& engine, CandidateCensus& census,
                              std::vector<char>& admitted_flags) {
    int admitted = 0;
    for (int i = 0; i < kk; ++i) {
        const int index = order[i];
        const CandidateGain& scored_exact = scored.exact[index];
        rect::Candidate candidate = mesh.refresh(scored.candidates[index]);
        if (candidate.vertex >= 0 && mesh.vertices[candidate.vertex].free_coefficient) continue;
        if (!scored_exact.ok || scored_exact.raw_gain_nats < -1e-8 || !(scored_exact.gain_nats > 1e-12)) continue;

        // Geometry first, at zero surplus: the splits are inert, so the surface is unchanged and
        // the statistical decision is profiled on the vertex's realised column.
        const bool existed = candidate.vertex >= 0;
        const std::size_t topology_before = mesh.topology_version;
        const int id = mesh.realise(candidate);
        const int splits = static_cast<int>(mesh.topology_version - topology_before);
        mesh.recompute_heights();
        candidate = mesh.refresh(candidate);

        ++census.reprofiled;
        const CandidateGain exact = engine.candidate_gain(mesh, candidate);
        if (!exact.ok || exact.raw_gain_nats < -1e-8 || !(exact.gain_nats > 1e-12)) { ++census.reprofile_rejected; continue; }
        rect::Vertex& v = mesh.vertices[id];
        const double base = v.height;   // zero surplus: the parents' average
        if (!height_is_valid(base + exact.optimum_units)) { ++census.reprofile_rejected; continue; }

        v.free_coefficient = true;
        v.gate_admitted = true;
        v.surplus = exact.optimum_units;
        v.lambda_v = exact.lambda_logit;
        v.sigma_sq = exact.data_information_optimum > 0.0 ? 1.0 / exact.data_information_optimum : 1e300;
        v.sigma_pooled = exact.data_information_optimum + v.lambda_v > 0.0 ? 1.0 / (exact.data_information_optimum + v.lambda_v) : 1e300;
        // the split as the gate scored it, frozen for the star plots
        v.admit_round = v.cand_round = round;
        v.admit_beta = scored_exact.optimum_units - scored_exact.null_units;
        v.admit_sd = scored.z_sd[index];
        v.admit_z = scored.z[index];
        v.admit_c = c_round;
        v.admit_xTx = scored_exact.xTx; v.admit_keff = scored_exact.keff;
        v.admit_gain = scored_exact.gain_nats; v.admit_signed_root = scored_exact.signed_root_gain;
        v.admit_info_null = scored_exact.information_at_null;
        v.admit_move = !existed ? 0 : (splits == 0 ? 1 : 2);
        v.admit_splits = splits;
        const auto chord = mesh.chord_ends(id);
        v.admit_chord_lo = chord[0]; v.admit_chord_hi = chord[1];
        ++mesh.coefficient_version;
        mesh.recompute_heights();
        admitted_flags[index] = 1;
        ++admitted;
    }
    return admitted;
}

// The round's calibration line: score moments, the fitted empirical null, the selection
// method and the admitted prefix.
void report_calibration(int rnd, const RoundCandidates& scored, const LfdrResult& lfdr_result,
                        bool lindsey_fallback, int kk, Engine& engine) {
    double zmean = 0.0, zsq = 0.0;
    for (double z : scored.z) zmean += z;
    zmean /= std::max<std::size_t>(scored.z.size(), 1);
    for (double z : scored.z) zsq += (z - zmean) * (z - zmean);
    const double round_z_sd = scored.z.size() > 1 ? std::sqrt(zsq / (scored.z.size() - 1)) : 0.0;
    std::fprintf(stderr,
        "[calibration] round %d M %zu score mean/sd %.6g/%.6g emp-null mean/sd/pi0 %.6g/%.6g/%.6g method %s admitted-prefix %d "
        "lindsey-status %s iter %d mass %.9g curvature %.9g\n",
        rnd, scored.z.size(), zmean, round_z_sd,
        lfdr_result.empirical_mean, lfdr_result.empirical_sd, lfdr_result.pi0,
        lfdr_result.used_lindsey ? "lindsey"
            : (lindsey_fallback ? (lfdr_result.status == LindseyStatus::invalid_central_fit ? "BH-fallback" : "defer-invalid-null") : "BH"),
        kk, lindsey_status_name(lfdr_result.status), lfdr_result.iterations, lfdr_result.fitted_mass, lfdr_result.central_curvature);
    engine.note_round(scored.size(), kk, static_cast<int>(lfdr_result.status), lfdr_result.empirical_sd, lfdr_result.empirical_mean);
}

// The gate rounds to convergence. A round scores the candidate frontier, calibrates an
// empirical null, selects under it, and admits. The loop ends when the cell budget is spent,
// when a round finds no scoreable candidate, or after kStallLimit consecutive rounds that
// admit nothing.
void run_gate_rounds(rect::Mesh& mesh, std::size_t cell_cap, double fdr_level, Engine& engine, CandidateCensus& census,
                     std::vector<CandidateRow>& records) {
    int stalls = 0;
    for (int rnd = 0; rnd < kMaxRounds; ++rnd) {
        if (mesh.leaf_count() > cell_cap) break;
        { estimator::ScopeTimer t_("refine: round start refit"); mesh.recompute_tau_sq(); engine.refit(mesh); }
        RoundCandidates scored;

        estimator::ScopeTimer t_frontier("refine: frontier");
        const std::vector<rect::Candidate> frontier = mesh.frontier();
        t_frontier.stop();

        ScoreContext context{mesh, rnd, census,
                             [&](std::size_t index, double z, double sd, const CandidateGain& e) { scored.push(frontier[index], z, sd, e); }};
        engine.score(frontier, context);
        if (scored.candidates.empty()) break;
        for (std::size_t q = 0; q < scored.size(); ++q)
            if (scored.candidates[q].vertex >= 0) {
                rect::Vertex& v = mesh.vertices[scored.candidates[q].vertex];
                v.cand_round = rnd; v.cand_z = scored.z[q]; v.cand_sd = scored.z_sd[q];
                v.cand_beta = scored.exact[q].optimum_units - scored.exact[q].null_units;
                v.cand_xTx = scored.exact[q].xTx; v.cand_keff = scored.exact[q].keff;
                v.cand_gain = scored.exact[q].gain_nats; v.cand_signed_root = scored.exact[q].signed_root_gain;
                v.cand_info_null = scored.exact[q].information_at_null;
            }

        std::vector<double> lf;
        LfdrResult lfdr_result;
        std::vector<int> direct_bh_order;
        int direct_bh_k = -1;
        bool lindsey_fallback = false;
        auto run_direct_bh = [&]() {
            std::vector<int> op(scored.z.size());
            std::vector<double> pv(scored.z.size());
            for (std::size_t i = 0; i < scored.z.size(); ++i) {
                pv[i] = 2 * normal_survival(std::fabs(scored.z[i]));
                op[i] = static_cast<int>(i);
            }
            // Ties broken by index: the order among equal p-values is load-bearing (admissions
            // commit sequentially), so it must not depend on the standard library's sort.
            std::sort(op.begin(), op.end(), [&](int a, int b) { return pv[a] < pv[b] || (pv[a] == pv[b] && a < b); });
            int kbh = 0;
            for (std::size_t i = 0; i < op.size(); ++i)
                if (pv[op[i]] <= fdr_level * (i + 1) / op.size()) kbh = static_cast<int>(i) + 1;
            direct_bh_order = std::move(op);
            direct_bh_k = kbh;
            lf.assign(scored.z.size(), 1.0);
            for (int i = 0; i < kbh; ++i) lf[direct_bh_order[i]] = 0.0;
        };

        if (scored.z.size() >= static_cast<std::size_t>(kLindseyMinimumCandidates)) {
            lfdr_result = lindsey_lfdr(scored.z, true);
            if (lfdr_result.used_lindsey) {
                lf = lfdr_result.lfdr;
            } else {
                lindsey_fallback = true;
                if (lfdr_result.status == LindseyStatus::invalid_central_fit) run_direct_bh();
                else lf.assign(scored.z.size(), 1.0);
            }
        } else {
            run_direct_bh();
        }
        std::vector<int> order;
        int kk = 0;
        if (direct_bh_k >= 0) {
            order = direct_bh_order;
            kk = direct_bh_k;
        } else {
            order.resize(lf.size());
            for (std::size_t i = 0; i < lf.size(); ++i) order[i] = static_cast<int>(i);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return lf[a] < lf[b] || (lf[a] == lf[b] && a < b); });
            double cum = 0.0;
            for (std::size_t i = 0; i < order.size(); ++i) {
                cum += lf[order[i]];
                if (cum / (i + 1) <= fdr_level) kk = static_cast<int>(i) + 1;
                else break;
            }
        }
        report_calibration(rnd, scored, lfdr_result, lindsey_fallback, kk, engine);
        int admitted = 0;
        std::vector<char> admitted_flags(scored.size(), 0);
        if (kk > 0) {
            double c_round = 1e300;
            for (int i = 0; i < kk; ++i) c_round = std::min(c_round, std::fabs(scored.z[order[i]]));
            estimator::ScopeTimer t_("refine: admit_selected_candidates");
            admitted = admit_selected_candidates(rnd, kk, order, scored, c_round, mesh, engine, census, admitted_flags);
        }
        {   // the round's candidates, as scored, for the star plots
            std::vector<char> selected(scored.size(), 0);
            for (int i = 0; i < kk; ++i) selected[order[i]] = 1;
            for (std::size_t q = 0; q < scored.size(); ++q) {
                const rect::Candidate& c = scored.candidates[q];
                CandidateRow row;
                row.round = rnd; row.x = static_cast<double>(c.ix) / rect::kSide; row.y = static_cast<double>(c.iy) / rect::kSide;
                row.x3 = static_cast<double>(c.iz) / rect::kSide;
                row.vertex = c.vertex; row.axis = c.axis; row.depth = c.depth; row.parent0 = c.p0; row.parent1 = c.p1;
                row.z = scored.z[q]; row.sd = scored.z_sd[q];
                row.beta = scored.exact[q].optimum_units - scored.exact[q].null_units; row.gain = scored.exact[q].gain_nats;
                row.selected = selected[q]; row.admitted = admitted_flags[q];
                records.push_back(row);
            }
        }
        // A round is a stall when nothing was admitted, whether nothing was selected or every
        // selected candidate failed its re-profile: the triangular driver counts only the first,
        // and a candidate selected and rejected in every round keeps it going to kMaxRounds with
        // nothing changing (the SMM design, seed 1: 70 idle rounds).
        if (admitted == 0) {
            stalls++;
            if (stalls >= kStallLimit) break;
            engine.refit(mesh);
            continue;
        }
        stalls = 0;
        if (rnd % kRefreshPeriod == kRefreshPeriod - 1) engine.refit(mesh);
    }
}

// The held-out deviance of the final fit.
void report_heldout(const RunConfig& config, const rect::Mesh& mesh, const std::vector<Observation>& obs) {
    if (!config.fit.heldout_split) return;
    double dev = 0, x2 = 0, wsum = 0;
    long m = 0;
    for (int i = 0; i < static_cast<int>(obs.size()); ++i) {
        if (mesh.training_mask[i]) continue;
        const double ph = std::clamp(logistic(eta_of(obs[i], mesh.predict_index(i))), 1e-6, 1 - 1e-6);
        const double k = obs[i].events, n = obs[i].exposure;
        if (k > 0) dev += 2 * k * std::log(k / (n * ph));
        if (n - k > 0) dev += 2 * (n - k) * std::log((n - k) / (n * (1 - ph)));
        const double inf = n * ph * (1 - ph);
        x2 += (k - n * ph) * (k - n * ph) / inf; wsum += inf; m++;
    }
    if (m > 0 && wsum > 0)
        std::printf("HELDOUT: mean deviance/pool %.3f | X2/info %.4f | %ld pools\n", dev / m, x2 / wsum, m);
    else
        std::printf("HELDOUT: no held-out pools\n");
}

}  // namespace

int refine(const RunConfig& config, Dataset& dataset, Engine& engine) {
    std::vector<Observation>& obs = dataset.observations;
    const int N = static_cast<int>(obs.size());
    std::printf("pool-months %d, pools %d\n", N, dataset.pool_count);
    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
    const bool heldout_split = config.fit.heldout_split;

    std::vector<unsigned char> mask = heldout_split ? dataset.training_mask : std::vector<unsigned char>(N, 1);
    std::unique_ptr<rect::Mesh> mesh;
    { estimator::ScopeTimer t_("refine: mesh ctor"); mesh = std::make_unique<rect::Mesh>(obs, mask, dataset.dim()); }

    pre_refine_mesh(*mesh);
    mesh->recompute_tau_sq();
    engine.refit(*mesh);
    std::printf("  [diag pre-rounds] cells %zu, vertices %zu, pred(0.5,0.5)=%.3f\n", mesh->leaf_count(), mesh->vertices.size(), mesh->predict(0.5, 0.5));
    const std::size_t cell_cap = static_cast<std::size_t>(std::max(kMinimumCellCap, N / kRowsPerCell));
    CandidateCensus census;
    std::vector<CandidateRow> candidate_rows;
    run_gate_rounds(*mesh, cell_cap, config.fit.fdr_level, engine, census, candidate_rows);
    engine.refit(*mesh);
    int free_count = 0;
    for (const rect::Vertex& v : mesh->vertices) free_count += v.free_coefficient ? 1 : 0;
    std::printf("refinement done: cells %zu, vertices %zu, free %d  [%.1f s]\n", mesh->leaf_count(), mesh->vertices.size(), free_count, elapsed());

    { estimator::ScopeTimer t_("refine: final refit"); engine.final_refit(*mesh); }
    { estimator::ScopeTimer t_("refine: audit"); engine.audit(*mesh); }

    report_heldout(config, *mesh, obs);
    std::fprintf(stderr, "[candidate census] attempted %lld scored %lld unscoreable %lld | reprofiled %lld rejected %lld\n",
                 census.attempted, census.scored, census.unscoreable, census.reprofiled, census.reprofile_rejected);
    std::printf("TOTAL %.1f s\n", elapsed());

    std::vector<double> sx(N), sy(N);
    for (int i = 0; i < N; ++i) { sx[i] = obs[i].x; sy[i] = obs[i].y; }
    estimator::ScopeTimer timer_dumps("refine: write_dumps");
    write_dumps(config.output, *mesh, obs, sx, sy, dataset.pool_count, engine.pool_variance());
    write_candidates(config.output, candidate_rows, mesh->dim);
    throw_if_output_failed();
    return 0;
}
