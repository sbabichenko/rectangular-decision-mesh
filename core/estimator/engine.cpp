#include "estimator/engine.h"

#include "estimator/system.h"
#include "estimator/json.h"
#include "estimator/profile.h"
#include "estimator/numeric.h"
#include "estimator/padded.h"
#include "estimator/quadrature.h"
#include "estimator/score.h"
#include "estimator/solver.h"
#include "estimator/variance.h"

#include <cblas.h>
#include <omp.h>

#include "mesh/rect.h"
#include "fit/refine.h"
#include "fit/units.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace estimator {

namespace {
// The depth variance's statistic for each child (EM): its fitted surplus squared plus its posterior
// variance in the working Gaussian, the diagonal of (X'WX + P)^-1 with W the centered
// information and P the depth priors (population levels included, unpenalised); the factor is the
// scoring system's, the same matrix. Stored on the vertex; rect::Mesh::recompute_tau_sq averages
// it per depth. Adding the posterior variance back is what keeps a depth off the floor: shrinkage
// moves mass from the squared mean to the variance. (pymesh/estimator/engine.py _store_second_moments.)
void store_second_moments(rect::Mesh& mesh, const LiveSystem& system, const std::vector<double>& factor,
                          const std::vector<double>& beta, std::size_t p) {
    // the system is in nodal heights: a child's surplus is its transform row r applied to them, and
    // its posterior variance r' (X'JX + P)^-1 r
    std::vector<double> inverse(p * p);
    cho_inverse(p, factor.data(), inverse.data());
    for (std::size_t c = 0; c < system.children.size(); ++c) {
        const auto& row = system.transform[system.root_count + c].entries;
        double mean = 0.0, variance = 0.0;
        for (const auto& [a, wa] : row) {
            mean += wa * beta[static_cast<std::size_t>(a)];
            for (const auto& [b, wb] : row) variance += wa * wb * inverse[static_cast<std::size_t>(a) * p + static_cast<std::size_t>(b)];
        }
        mesh.vertices[system.children[c]].em_second = mean * mean + std::max(variance, 0.0);
    }
}
}  // namespace

namespace {

std::string env_or(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string(fallback);
}

double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct FitRecord {
    double seconds = 0.0; int coefficients = 0; double pool_variance = 0.0;
    std::optional<std::vector<double>> levels;
    double score_max = 0.0; int surface_evaluations = 0; bool surface_fit_reused = false;
    int exact_evaluations = 0, partial_rows = 0; bool certified = true;
    std::optional<JsonValue> variance_fit; bool variance_fit_reused = false; bool skipped_unchanged = false;
    JsonValue json() const {
        JsonValue o;
        o.set("seconds", JsonValue::of(seconds)); o.set("coefficients", JsonValue::of(coefficients));
        o.set("pool_variance", JsonValue::of(pool_variance));
        if (levels) { JsonValue l = JsonValue::list(); for (double v : *levels) l.push(JsonValue::of(v)); o.set("levels", l); }
        else o.set("levels", JsonValue::null());
        o.set("score_max", JsonValue::of(score_max)); o.set("surface_evaluations", JsonValue::of(surface_evaluations));
        o.set("surface_fit_reused", JsonValue::of(surface_fit_reused)); o.set("exact_evaluations", JsonValue::of(exact_evaluations));
        o.set("partial_rows", JsonValue::of(partial_rows)); o.set("certified", JsonValue::of(certified));
        if (variance_fit) { o.set("variance_fit", *variance_fit); o.set("variance_fit_reused", JsonValue::of(variance_fit_reused)); }
        if (skipped_unchanged) o.set("skipped_unchanged", JsonValue::of(true));
        return o;
    }
};

struct CenteredState {
    bool have = false;
    // fit key: the training inputs and the variance
    std::shared_ptr<const PaddedDesign> design;   // the fit key's design and the scoring system's
    std::shared_ptr<const std::vector<double>> key_n, key_k, key_offset, key_penalty; double key_variance = 0.0;
    std::vector<int64_t> row_lookup;    // global row -> training position or -1
    std::vector<int64_t> train_rows;    // training positions -> global rows
    std::unique_ptr<ScoreSystem> system;
    std::vector<double> prediction, info;
    SurfaceFit fit;
    int refresh_epoch = 0; long long w_version = 0;
    std::vector<double> all_heights;
    FitRecord record;
    JsonValue record_json;
};

struct VarianceCacheEntry {
    bool have = false;
    std::shared_ptr<const PaddedDesign> design; std::shared_ptr<const std::vector<double>> n, k, offset, penalty;
    VarianceFit fit;
};

}  // namespace

struct Engine::Impl {
    Impl(const RunConfig& config, Dataset& dataset)
        : config_(config), dataset_(dataset) {
        read_settings();
    }
    int run(Engine& engine);

    void read_settings();
    FitRecord fit(rect::Mesh& mesh, bool estimate, double tolerance, bool final);
    CandidateGain candidate(const rect::Mesh& mesh, const rect::SparseColumn& column,
                            double current, double null, double lam_units);
    void score(const std::vector<rect::Candidate>& frontier, ScoreContext& context);
    void save_trace();
    void write_metadata();

    // fit()'s phases, each named for what it hands back to the phase after it.
    void load_level_groups(std::size_t N, const std::vector<double>& offset_all);

    // The training design fit() solves against: the live hierarchical system's design and
    // priors, widened by a joint-levels column when DMESH_LEVEL_GROUPS is set.
    struct TrainingProblem {
        std::shared_ptr<const PaddedDesign> Xt_shared;
        std::shared_ptr<const std::vector<double>> nt_shared, kt_shared, ot_shared, penalty_shared;
        std::vector<double> initial;
    };
    TrainingProblem build_training_problem(const LiveSystem& system, const std::vector<double>& offset,
                                           const std::vector<double>& n, const std::vector<double>& k,
                                           const std::vector<int64_t>& train_rows);

    // The pool variance this fit uses, and the surface fitted while finding it (empty when the
    // step held and the surface below is fitted separately).
    struct VarianceStep {
        double variance = 0.0;
        std::vector<double> initial;
        bool stepped = false;
        bool have_stepped_surface = false;
        SurfaceFit stepped_surface;
        JsonValue stepped_record;
    };
    VarianceStep step_or_start_variance(bool estimate, bool final, double tolerance, SurfaceProblem& problem,
                                        const PaddedDesign& Xt, std::size_t surface_count, double variance,
                                        std::vector<double> initial, const TrainingProblem& training,
                                        RowCache* row_cache, AccurateCache* accurate_cache);

    // Commit a fitted surface to the mesh's vertex heights, after checking every height is
    // finite and within the runaway guard (a near-empty region can honestly reach the proposal
    // limit, but never three times over it).
    void commit_surface_heights(rect::Mesh& mesh, const LiveSystem& system, const SurfaceFit& fitted,
                                std::size_t surface_count);

    // Each free coefficient's support at the fit: the information and rows of its nodal column (the
    // design is nodal), and its prior precision.
    void update_information(rect::Mesh& mesh, const LiveSystem& system, const PaddedDesign& Xt,
                            const std::vector<double>& info, std::size_t surface_count, std::size_t T, std::size_t p);

    const RunConfig& config_;
    Dataset& dataset_;
    Engine* engine_ = nullptr;   // set by run(): score() profiles candidates through the public method
    CandidateGain candidate_gain(const rect::Mesh& mesh, const rect::Candidate& c) { return engine_->candidate_gain(mesh, c); }
    CandidateGain candidate_gain(const rect::Mesh& mesh, const rect::Candidate& c, const rect::SparseColumn& column) { return engine_->candidate_gain(mesh, c, column); }
    // the frozen settings (2026-09-21): local rows within 1e-2 logits of their cached base, the
    // step schedule for the pool variance between gate rounds with a hold of 1e-2 (relative),
    // coarse moments first and accurate ones when the coarse step is small, the alternating
    // solver for the initial and final variance, intermediate surfaces uncertified, the
    // scoring state reused when nothing moved
    int threads_ = 1;
    static constexpr double row_threshold_ = 1e-2, step_hold_ = 1e-2, final_tolerance_ = 1e-5, gate_tolerance_ = 1e-3;
    // state
    std::vector<JsonValue> trace_;
    std::optional<std::vector<double>> last_levels_;
    std::optional<std::string> audit_path_;
    // mesh-attached state (one mesh at a time; a new mesh resets it)
    rect::Mesh* state_mesh_ = nullptr;
    RowCache row_cache_; AccurateCache accurate_cache_;
    // the level factors: level_groups_ is rows x factors (row-major), each factor's codes 0 (its
    // reference) to level_sizes_[f]; the factors' columns follow one another after the surface's
    std::vector<int64_t> level_groups_; std::vector<int> level_sizes_; int level_count_ = 0; std::vector<double> level_values_, base_offsets_;
    double row_level_of(std::size_t i) const {
        const std::size_t F = level_sizes_.size();
        double level = 0.0; std::size_t start = 0;
        for (std::size_t f = 0; f < F; ++f) {
            const int64_t g = level_groups_[i * F + f];
            if (g > 0) level += level_values_[start + g - 1];
            start += level_sizes_[f];
        }
        return level;
    }
    bool levels_loaded_ = false;
    double centered_variance_ = 0.0;
    VarianceCacheEntry variance_cache_;
    CenteredState state_;
    LiveCache live_cache_;
    long long w_version_ = 0;
    void reset_mesh_state(rect::Mesh& mesh);
};

void Engine::Impl::read_settings() {
    threads_ = threads_setting();
}

void Engine::Impl::reset_mesh_state(rect::Mesh& mesh) {
    if (state_mesh_ == &mesh) return;
    state_mesh_ = &mesh;
    row_cache_.clear(); accurate_cache_.clear();
    levels_loaded_ = false; level_count_ = 0; level_values_.clear(); base_offsets_.clear(); level_groups_.clear();
    centered_variance_ = 0.0;
    variance_cache_ = VarianceCacheEntry(); state_ = CenteredState(); live_cache_ = LiveCache(); w_version_ = 0;
}

void Engine::Impl::save_trace() {
    if (!audit_path_) return;
    std::ofstream out(*audit_path_);
    JsonValue rows = JsonValue::list();
    for (const JsonValue& row : trace_) rows.push(row);
    out << rows.dumps() << "\n";
}

FitRecord Engine::Impl::fit(rect::Mesh& mesh, bool estimate, double tolerance, bool final) {
    ScopeTimer timer_total("engine.fit (total)");
    reset_mesh_state(mesh);
    const double started = now();
    const std::vector<Observation>& obs = *mesh.obs_store;
    const std::size_t N = static_cast<std::size_t>(mesh.n_points);
    std::vector<double> offset_all(N), n(N), k(N);
    for (std::size_t i = 0; i < N; ++i) { offset_all[i] = obs[i].offset; n[i] = obs[i].exposure; k[i] = obs[i].events; }
    load_level_groups(N, offset_all);
    const std::vector<double>& offset = level_count_ ? base_offsets_ : offset_all;
    std::vector<int64_t> train_rows;
    std::vector<char> train(N, 1);
    if (config_.fit.heldout_split)
        for (std::size_t i = 0; i < N; ++i) train[i] = mesh.training_mask[i] != 0;
    for (std::size_t i = 0; i < N; ++i) if (train[i]) train_rows.push_back(static_cast<int64_t>(i));
    LiveSystem system = live_system(mesh, config_.fit.heldout_split, train_rows, &live_cache_);
    const std::size_t surface_count = system.free.size();
    const std::size_t T = train_rows.size();
    TrainingProblem training = build_training_problem(system, offset, n, k, train_rows);
    std::shared_ptr<const PaddedDesign> Xt_shared = training.Xt_shared;
    const std::shared_ptr<const std::vector<double>> nt_shared = training.nt_shared, kt_shared = training.kt_shared,
        ot_shared = training.ot_shared, penalty_shared = training.penalty_shared;
    const std::vector<double>& nt = *nt_shared; const std::vector<double>& kt = *kt_shared; const std::vector<double>& ot = *ot_shared;
    const PaddedDesign& Xt = *Xt_shared;
    const std::size_t p = Xt.columns;
    const std::vector<double>& penalty = *penalty_shared;
    std::vector<double> initial = training.initial;
    // the same vectors compare equal to a cache entry by pointer, else by value
    auto same = [](const std::shared_ptr<const std::vector<double>>& a, const std::vector<double>& b) { return a && (a.get() == &b || *a == b); };
    SurfaceProblem problem{&Xt, &nt, &kt, &ot, &penalty, threads_};
    RowCache* row_cache = row_threshold_ > 0 ? &row_cache_ : nullptr;
    AccurateCache* accurate_cache = row_threshold_ > 0 ? &accurate_cache_ : nullptr;
    double variance = centered_variance_;
    VarianceStep vstep = step_or_start_variance(estimate, final, tolerance, problem, Xt, surface_count,
                                                variance, initial, training, row_cache, accurate_cache);
    variance = vstep.variance; initial = vstep.initial;
    const bool stepped = vstep.stepped;
    const bool have_stepped_surface = vstep.have_stepped_surface;
    const SurfaceFit& stepped_surface = vstep.stepped_surface;
    const JsonValue& stepped_record = vstep.stepped_record;
    const bool key_same = state_.have && (state_.design == Xt_shared || state_.design->same_as(Xt)) && same(state_.key_n, nt) && same(state_.key_k, kt) && same(state_.key_offset, ot)
                          && same(state_.key_penalty, penalty) && state_.key_variance == variance;
    const bool surface_reused = !have_stepped_surface && key_same;
    if (state_.have && !surface_reused) state_.system.reset();
    std::vector<double> heights_now;
    heights_now.reserve(mesh.vertices.size());
    for (const rect::Vertex& v : mesh.vertices) heights_now.push_back(v.height);
    if (surface_reused && state_.w_version == w_version_ && state_.refresh_epoch == mesh.refresh_epoch && state_.all_heights == heights_now) {
        FitRecord record = state_.record;
        record.seconds = now() - started; record.surface_evaluations = 0; record.surface_fit_reused = true; record.skipped_unchanged = true;
        return record;
    }
    SurfaceFit fitted = have_stepped_surface ? stepped_surface
                        : surface_reused ? state_.fit
                        : fit_surface(problem, variance, initial, row_cache, row_threshold_, final);
    std::vector<double> row_level;
    if (level_count_) {
        level_values_.assign(fitted.beta.begin() + surface_count, fitted.beta.end());
        row_level.assign(N, 0.0);
        for (std::size_t i = 0; i < N; ++i) row_level[i] = row_level_of(i);
        auto& observations = const_cast<std::vector<Observation>&>(obs);
        for (std::size_t i = 0; i < N; ++i) observations[i].offset = base_offsets_[i] + row_level[i];
    }
    commit_surface_heights(mesh, system, fitted, surface_count);
    std::vector<double> pred(N);
    { ScopeTimer t("  predict all"); for (std::size_t i = 0; i < N; ++i) pred[i] = mesh.predict_index(static_cast<int>(i)); }
    {   // the consistency check on the training rows
        std::vector<double> row_shift(T);
        std::vector<double> beta_full = fitted.beta;
        Xt.matvec(beta_full.data(), row_shift.data(), threads_);
        for (std::size_t r = 0; r < T; ++r) {
            const double shift = row_shift[r] - (level_count_ ? (obs[train_rows[r]].offset - base_offsets_[train_rows[r]]) : 0.0);
            if (std::fabs(pred[train_rows[r]] - shift) > 1e-8) throw std::runtime_error("centered predictions disagree with the design");
        }
    }
    const Moments& expected = *fitted.moments;
    std::vector<double> score(T), info(T);
    for (std::size_t r = 0; r < T; ++r) { score[r] = expected.a1[r] * fitted.z[r]; info[r] = expected.information[r]; }
    for (double v : info) if (!(v > 0) || !std::isfinite(v)) throw std::runtime_error("nonpositive centered working information");
    mesh.refresh_epoch += 1;
    w_version_ += 1;
    update_information(mesh, system, Xt, info, surface_count, T, p);
    // the scoring state
    state_.have = true;
    state_.system.reset();
    state_.design = Xt_shared; state_.key_n = nt_shared; state_.key_k = kt_shared; state_.key_offset = ot_shared; state_.key_penalty = penalty_shared; state_.key_variance = variance;
    state_.row_lookup.assign(N, -1);
    for (std::size_t r = 0; r < T; ++r) state_.row_lookup[train_rows[r]] = static_cast<int64_t>(r);
    state_.train_rows = train_rows;
    state_.system = std::make_unique<ScoreSystem>(*state_.design, fitted.beta, penalty, expected, score, threads_);
    store_second_moments(mesh, system, state_.system->factor(), fitted.beta, p);
    state_.prediction = std::move(pred); state_.info = std::move(info); state_.fit = fitted;
    state_.refresh_epoch = mesh.refresh_epoch; state_.w_version = w_version_;
    state_.all_heights.clear();
    for (const rect::Vertex& v : mesh.vertices) state_.all_heights.push_back(v.height);
    FitRecord record;
    record.seconds = now() - started; record.coefficients = static_cast<int>(surface_count); record.pool_variance = variance;
    if (level_count_) { record.levels = level_values_; last_levels_ = level_values_; }
    record.score_max = fitted.score_max; record.surface_evaluations = surface_reused ? 0 : fitted.evaluations;
    record.surface_fit_reused = surface_reused; record.exact_evaluations = fitted.exact_evaluations; record.partial_rows = fitted.partial_rows;
    record.certified = fitted.certified;
    if (have_stepped_surface) {
        JsonValue vrec = stepped_record;
        bool reused_flag = false;
        for (auto& [k, v] : vrec.object) if (k == "__reused") reused_flag = v.kind == JsonValue::Bool && v.boolean;
        vrec.object.erase(std::remove_if(vrec.object.begin(), vrec.object.end(), [](const auto& kv) { return kv.first == "__reused"; }), vrec.object.end());
        record.variance_fit = vrec; record.variance_fit_reused = stepped ? false : reused_flag;
    }
    state_.record = record;
    return record;
}

// fit()'s phases -------------------------------------------------------------------------

void Engine::Impl::load_level_groups(std::size_t N, const std::vector<double>& offset_all) {
    if (levels_loaded_) return;
    levels_loaded_ = true;
    const char* groups_file = std::getenv("DMESH_LEVEL_GROUPS");
    if (!groups_file || !*groups_file) return;
    std::ifstream in(groups_file);
    if (!in) throw std::runtime_error(std::string(groups_file) + " not found.");   // numpy's message, as the port reports it
    // one line per row, one column per factor
    std::string line; std::size_t factors = 0, lines = 0;
    while (std::getline(in, line)) {
        std::istringstream tokens(line);
        std::size_t count = 0; long long value;
        while (tokens >> value) { level_groups_.push_back(value); ++count; }
        if (count == 0) continue;
        if (factors == 0) factors = count;
        if (count != factors) throw std::invalid_argument("DMESH_LEVEL_GROUPS needs the same number of factors on every row");
        ++lines;
    }
    if (lines != N || factors == 0) throw std::invalid_argument("DMESH_LEVEL_GROUPS needs one nonnegative integer per row");
    for (long long g : level_groups_) if (g < 0) throw std::invalid_argument("DMESH_LEVEL_GROUPS needs one nonnegative integer per row");
    level_sizes_.assign(factors, 0);
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t f = 0; f < factors; ++f)
            level_sizes_[f] = std::max(level_sizes_[f], static_cast<int>(level_groups_[i * factors + f]));
    level_count_ = 0;
    for (int s : level_sizes_) level_count_ += s;
    level_values_.assign(level_count_, 0.0);
    base_offsets_ = offset_all;
}

Engine::Impl::TrainingProblem Engine::Impl::build_training_problem(
    const LiveSystem& system, const std::vector<double>& offset,
    const std::vector<double>& n, const std::vector<double>& k, const std::vector<int64_t>& train_rows) {
    const std::size_t surface_count = system.free.size();
    const std::size_t T = train_rows.size();
    std::vector<double> nt(T), kt(T), ot(T);
    for (std::size_t r = 0; r < T; ++r) { nt[r] = n[train_rows[r]]; kt[r] = k[train_rows[r]]; ot[r] = offset[train_rows[r]]; }
    TrainingProblem out;
    out.nt_shared = std::make_shared<std::vector<double>>(std::move(nt));
    out.kt_shared = std::make_shared<std::vector<double>>(std::move(kt));
    out.ot_shared = std::make_shared<std::vector<double>>(std::move(ot));
    out.Xt_shared = system.X;
    out.initial = system.initial;

    if (level_count_) {
        // one more padded entry per row for each factor: the row's indicator in that factor's block
        // (built in one copy of the design: successive copies held three at once)
        const std::size_t F = level_sizes_.size();
        std::vector<std::vector<int64_t>> extra_index; std::vector<std::vector<double>> extra_value;
        std::size_t start = 0;
        for (std::size_t f = 0; f < F; ++f) {
            if (level_sizes_[f] == 0) continue;
            extra_index.emplace_back(T); extra_value.emplace_back(T);
            for (std::size_t r = 0; r < T; ++r) {
                const long long g = level_groups_[static_cast<std::size_t>(train_rows[r]) * F + f];
                extra_index.back()[r] = g > 0 ? static_cast<int64_t>(surface_count + start + g - 1) : 0;
                extra_value.back()[r] = g > 0 ? 1.0 : 0.0;
            }
            start += level_sizes_[f];
        }
        out.Xt_shared = std::make_shared<const PaddedDesign>(extra_index.size() == 1
            ? out.Xt_shared->with_extra_column(extra_index[0], extra_value[0], surface_count + level_count_)
            : out.Xt_shared->with_extra_columns(extra_index, extra_value, surface_count + level_count_));
        out.initial.insert(out.initial.end(), level_values_.begin(), level_values_.end());
    }
    const std::size_t p = out.Xt_shared->columns;
    auto penalty_shared = std::make_shared<std::vector<double>>(p * p, 0.0);
    std::vector<double>& penalty = *penalty_shared;
    // the depth prior in nodal coordinates (R' diag(1/tau^2) R); the level columns are unpenalised
    for (std::size_t i = 0; i < surface_count; ++i)
        for (std::size_t j = 0; j < surface_count; ++j) penalty[i * p + j] = system.penalty[i * surface_count + j];
    out.penalty_shared = penalty_shared;
    return out;
}

Engine::Impl::VarianceStep Engine::Impl::step_or_start_variance(
    bool estimate, bool final, double tolerance, SurfaceProblem& problem, const PaddedDesign& Xt,
    std::size_t surface_count, double variance, std::vector<double> initial, const TrainingProblem& training,
    RowCache* row_cache, AccurateCache* accurate_cache) {
    const std::vector<double>& nt = *problem.n; const std::vector<double>& ot = *problem.offset;
    auto same = [](const std::shared_ptr<const std::vector<double>>& a, const std::vector<double>& b) { return a && (a.get() == &b || *a == b); };
    VarianceStep result;
    bool stepped = false;
    bool any_pair = false;
    for (double v : nt) if (v > 1) { any_pair = true; break; }
    auto ddot_ws = [](const Moments& m) { return ddot(m.size(), m.weight.data(), m.sensitivity.data()); };
    if (estimate && !final && variance == 0 && any_pair) {
        std::printf("CENTERED initial variance step %zu coefficients\n", surface_count); std::fflush(stdout);
        SurfaceFit zero = fit_surface(problem, 0.0, initial);
        const double score0 = dispersion_score(problem, zero.beta, 0.0, zero);
        const double first = initial_variance_guess(problem, zero.beta, score0, 2e-4, 1.9);
        if (first > 0) {
            SurfaceFit surface = fit_surface(problem, first, zero.beta, row_cache, row_threshold_, false);
            variance = first;
            stepped = true;
            result.stepped_record.set("variance", JsonValue::of(variance)); result.stepped_record.set("status", JsonValue::of("initial_step"));
            result.stepped_record.set("score", JsonValue::of(score0)); result.stepped_record.set("variance_step", JsonValue::of(first));
            result.stepped_record.set("variance_tolerance", JsonValue::null()); result.stepped_record.set("moved", JsonValue::of(true));
            result.stepped_record.set("surface_evaluations_total", JsonValue::of(zero.evaluations + surface.evaluations));
            result.stepped_record.set("high_evaluations", JsonValue::of(1)); result.stepped_record.set("coarse_evaluations", JsonValue::of(0));
            result.stepped_record.set("evaluations", JsonValue::of(1)); result.stepped_record.set("relative_tolerance", JsonValue::null());
            result.stepped_record.set("solver_used", JsonValue::of("initial_step")); result.stepped_record.set("accurate_evaluated", JsonValue::of(true));
            result.stepped_surface = surface; result.have_stepped_surface = true;
            initial = surface.beta;
            centered_variance_ = variance;
        }
    }
    if (!stepped && estimate && !final && variance > 0) {
        std::printf("CENTERED variance step %zu coefficients\n", surface_count); std::fflush(stdout);
        SurfaceFit surface = fit_surface(problem, variance, initial, row_cache, row_threshold_, false);
        // one Newton step on the dispersion equation, the slope from the moments' sensitivity
        auto newton_step = [&](double score, const Moments& used) {
            const double sensitivity = ddot_ws(used);
            return sensitivity > 0 ? score / sensitivity : 0.0;
        };
        const double step_tolerance = step_hold_ * std::max(variance, .01);
        std::optional<std::pair<int, int>> accurate_rows;
        const double coarse_score = dispersion_score(problem, surface.beta, variance, surface, surface.moments.get());
        const double coarse_step = newton_step(coarse_score, *surface.moments);
        double score, step;
        if (std::fabs(coarse_step) <= step_tolerance) {
            // a small coarse step: decide the move on accurate moments
            score = dispersion_score(problem, surface.beta, variance, surface, nullptr, accurate_cache, row_threshold_);
            if (accurate_cache) accurate_rows = std::make_pair(accurate_cache->rows_resolved, accurate_cache->rows_total);
            step = newton_step(score, accurate_cache ? accurate_cache->moments : *surface.moments);
        } else {
            score = coarse_score; step = coarse_step;
        }
        const bool moved = std::fabs(step) > step_tolerance;
        if (moved) {
            const double new_variance = std::clamp(variance + step, .5 * variance, 2. * variance);
            surface = fit_surface(problem, new_variance, surface.beta, row_cache, row_threshold_, false);
            variance = new_variance;
        }
        stepped = true;
        result.stepped_record.set("variance", JsonValue::of(variance)); result.stepped_record.set("status", JsonValue::of("step"));
        result.stepped_record.set("score", JsonValue::of(score)); result.stepped_record.set("variance_step", JsonValue::of(step));
        result.stepped_record.set("variance_tolerance", JsonValue::of(step_tolerance)); result.stepped_record.set("moved", JsonValue::of(moved));
        result.stepped_record.set("coarse_step", JsonValue::of(coarse_step));
        result.stepped_record.set("accurate_evaluated", JsonValue::of(accurate_rows.has_value()));
        result.stepped_record.set("surface_evaluations_total", JsonValue::of(surface.evaluations));
        result.stepped_record.set("high_evaluations", JsonValue::of(1)); result.stepped_record.set("coarse_evaluations", JsonValue::of(0));
        result.stepped_record.set("evaluations", JsonValue::of(1)); result.stepped_record.set("relative_tolerance", JsonValue::of(tolerance));
        result.stepped_record.set("solver_used", JsonValue::of("step"));
        result.stepped_record.set("accurate_rows_resolved", accurate_rows ? [&] { JsonValue l = JsonValue::list(); l.push(JsonValue::of(static_cast<long long>(accurate_rows->first))); l.push(JsonValue::of(static_cast<long long>(accurate_rows->second))); return l; }() : JsonValue::null());
        result.stepped_surface = surface; result.have_stepped_surface = true;
        initial = surface.beta;
        centered_variance_ = variance;
    } else if (estimate && !stepped) {
        std::printf("CENTERED variance start %zu coefficients\n", surface_count); std::fflush(stdout);
        // a cached solve serves a request at its own precision or looser; the final fit never
        // inherits a gate round's loose root
        const bool reused = variance_cache_.have && (variance_cache_.design == training.Xt_shared || variance_cache_.design->same_as(Xt)) && same(variance_cache_.n, nt) && same(variance_cache_.k, *problem.k)
                            && same(variance_cache_.offset, ot) && same(variance_cache_.penalty, *problem.penalty)
                            && (variance_cache_.fit.relative_tolerance > 0 ? variance_cache_.fit.relative_tolerance : final_tolerance_) <= tolerance;
        VarianceFit vf;
        if (reused) vf = variance_cache_.fit;
        else {
            vf = fit_variance(problem, initial, 2.0, variance, tolerance);
            variance_cache_ = VarianceCacheEntry{true, training.Xt_shared, training.nt_shared, training.kt_shared, training.ot_shared, training.penalty_shared, vf};
        }
        if (!vf.resolved) throw std::runtime_error("pool variance update unresolved: " + vf.status + " " + vf.reason);
        variance = vf.variance;
        initial = vf.beta;
        centered_variance_ = variance;
        if (row_cache) { row_cache->clear(); accurate_cache->clear(); }   // solved at a new variance: bases are stale
        result.stepped_record.set("variance", JsonValue::of(vf.variance)); result.stepped_record.set("status", JsonValue::of(vf.status)); result.stepped_record.set("score", JsonValue::of(vf.score));
        result.stepped_record.set("evaluations", JsonValue::of(vf.evaluations)); result.stepped_record.set("surface_evaluations_total", JsonValue::of(vf.surface_evaluations_total));
        result.stepped_record.set("variance_tolerance", JsonValue::of(vf.variance_tolerance)); result.stepped_record.set("variance_step", JsonValue::of(vf.variance_step));
        result.stepped_record.set("relative_tolerance", JsonValue::of(vf.relative_tolerance)); result.stepped_record.set("coarse_evaluations", JsonValue::of(vf.coarse_evaluations));
        result.stepped_record.set("high_evaluations", JsonValue::of(vf.high_evaluations)); result.stepped_record.set("solver_used", JsonValue::of(vf.solver_used));
        result.stepped_record.set("profile_scan_performed", JsonValue::of(false));
        result.stepped_surface = vf.surface; result.have_stepped_surface = true;
        // the record flag: variance_fit_reused = reused
        result.stepped_record.set("__reused", JsonValue::of(reused));
    }
    result.variance = variance; result.initial = initial; result.stepped = stepped;
    return result;
}

void Engine::Impl::commit_surface_heights(rect::Mesh& mesh, const LiveSystem& system, const SurfaceFit& fitted,
                                          std::size_t surface_count) {
    // the fitted coefficients are the free vertices' heights; the mesh keeps surpluses (R h), and
    // every height follows from them by the birth-parent recursion
    const std::vector<double> heights(fitted.beta.begin(), fitted.beta.begin() + surface_count);
    const std::vector<double> surplus = surpluses_from(system, heights);
    std::vector<double> previous(surface_count);
    for (std::size_t j = 0; j < surface_count; ++j) {
        rect::Vertex& v = mesh.vertices[system.free[j]];
        previous[j] = v.surplus;
        v.surplus = surplus[j];
    }
    mesh.recompute_heights();
    double worst = 0.0; bool finite = true; int worst_id = -1;
    for (const rect::Vertex& v : mesh.vertices) {
        if (!std::isfinite(v.height)) finite = false;
        if (std::fabs(v.height) > worst) { worst = std::fabs(v.height); worst_id = v._id; }
    }
    if (!finite || worst > 3 * kHeightLimitUnits) {
        std::ostringstream detail;
        if (worst_id >= 0) {
            const rect::Vertex& v = mesh.vertices[worst_id];
            detail << "vertex " << v._id << " depth " << v.depth << " height " << v.height << " at (" << v.x << ", " << v.y << ")";
        }
        for (std::size_t j = 0; j < surface_count; ++j) mesh.vertices[system.free[j]].surplus = previous[j];
        mesh.recompute_heights();
        throw std::runtime_error("centered surface exceeds existing height safeguard: " + detail.str());
    }
}

void Engine::Impl::update_information(rect::Mesh& mesh, const LiveSystem& system, const PaddedDesign& Xt,
                                      const std::vector<double>& info, std::size_t surface_count,
                                      std::size_t T, std::size_t p) {
    std::vector<double> xwx(p, 0.0), xw2x(p, 0.0), rows(p, 0.0);
    for (std::size_t r = 0; r < T; ++r)
        for (int a = 0; a < Xt.width; ++a) {
            const int64_t c = static_cast<int64_t>(Xt.index[r * Xt.width + a]); const double v = Xt.value[r * Xt.width + a];
            if (v == 0.0) continue;
            xwx[c] += info[r] * v * v; xw2x[c] += info[r] * info[r] * v * v; rows[c] += 1;
        }
    for (std::size_t j = 0; j < surface_count; ++j) {
        rect::Vertex& v = mesh.vertices[system.free[j]];
        const double information = xwx[j];
        v.current_fit_points = static_cast<int>(rows[j]);
        v.current_xTx = information;
        v.current_keff = xw2x[j] > 0 ? information * information / xw2x[j] : 0.0;
        v.sigma_sq = information > 0 ? 1. / information : 1e300;
        if (v.has_prior()) {
            const double lv = 1. / mesh.tau_for_depth(v.depth);
            v.lambda_v = lv;
            v.sigma_pooled = information + lv > 0 ? 1. / (information + lv) : 1e300;
        }
    }
}


CandidateGain Engine::Impl::candidate(const rect::Mesh& mesh, const rect::SparseColumn& column,
                                      double current, double null, double lam_units) {
    ScopeTimer timer("engine.candidate");
    CandidateGain out;
    out.null_units = null;
    if (!state_.have || !state_.system) throw std::runtime_error("no centered state to score against");
    std::vector<int64_t> rows; std::vector<double> x;
    {
        // the column's entries above 1e-12 in magnitude (rows ascending, no duplicates), then the
        // training entries (training_column: keep training rows, sum duplicates in row order)
        std::vector<std::pair<int64_t, double>> kept;
        kept.reserve(column.size());
        for (const auto& [i, v] : column) {
            if (!(std::fabs(v) > 1e-12)) continue;
            const int64_t r = state_.row_lookup[static_cast<std::size_t>(i)];
            if (r >= 0) kept.emplace_back(r, v);
        }
        bool ordered = true;
        for (std::size_t q = 1; q < kept.size(); ++q) if (kept[q].first <= kept[q - 1].first) { ordered = false; break; }
        if (!ordered) {
            std::stable_sort(kept.begin(), kept.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            std::vector<std::pair<int64_t, double>> summed;
            for (const auto& [r, v] : kept) {
                if (!summed.empty() && summed.back().first == r) summed.back().second += v;
                else summed.emplace_back(r, v);
            }
            kept.swap(summed);
        }
        for (const auto& [r, v] : kept) { rows.push_back(r); x.push_back(v); }
    }
    int support = 0;
    double xTx = 0.0, xT2x = 0.0;
    for (std::size_t q = 0; q < x.size(); ++q) {
        if (std::fabs(x[q]) > 1e-12) ++support;
        const double w = state_.info[static_cast<std::size_t>(rows[q])];
        xTx += w * x[q] * x[q]; xT2x += w * w * x[q] * x[q];
    }
    out.fit_points = support;
    out.xTx = xTx; out.keff = xT2x > 0 ? xTx * xTx / xT2x : 0.0;
    if (support < rect::Vertex::kMinimumFitPoints || xTx <= 0) return out;
    const CandidateStats stats = state_.system->candidate(x, rows);
    const double h = stats.information, variance = stats.variance;
    double score = stats.raw_score;
    if (h <= 1e-12 || variance <= 1e-12) return out;
    // the displacement on the candidate's rows: the surface now against the round's frozen predictions
    std::vector<double> change(rows.size()), infox(rows.size());
    for (std::size_t q = 0; q < rows.size(); ++q) {
        const int64_t g = state_.train_rows[static_cast<std::size_t>(rows[q])];
        change[q] = mesh.predict_index(static_cast<int>(g)) - state_.prediction[static_cast<std::size_t>(g)];
        infox[q] = state_.info[static_cast<std::size_t>(rows[q])] * change[q];
    }
    score -= ddot(rows.size(), x.data(), infox.data());
    score -= h * (null - current);
    const double lam = lam_units;
    const double step = score / (h + lam);
    out.lambda_logit = lam;
    out.information_at_null = h + lam;
    out.data_information_optimum = h;
    out.optimum_units = null + step;
    out.gain_nats = out.raw_gain_nats = .5 * score * step;
    out.signed_root_gain = (score > 0 ? 1.0 : (score < 0 ? -1.0 : 0.0)) * std::sqrt(2 * out.gain_nats);
    out.ok = true;
    out.centered_gate_z = (score - stats.shrinkage_bias) / std::sqrt(variance);
    out.centered_gate_sd = std::sqrt(variance) / h;
    return out;
}

void Engine::Impl::score(const std::vector<rect::Candidate>& frontier, ScoreContext& context) {
    ScopeTimer timer("engine.score (total)");
    rect::Mesh& mesh = context.mesh;
    FitRecord record = fit(mesh, true, gate_tolerance_, false);
    JsonValue row = record.json();
    row.set("variance_update", JsonValue::of("full"));
    row.set("round", JsonValue::of(context.round));
    row.set("cells_before", JsonValue::of(static_cast<long long>(mesh.leaf_count()))); row.set("frontier", JsonValue::of(static_cast<long long>(frontier.size())));
    trace_.push_back(row);
    save_trace();
    // The candidates are independent reads of the frozen fit: score them in parallel, then
    // apply the results in frontier order (the counters and the pushes).
    std::vector<CandidateGain> scored(frontier.size());
    // Three dimensions: a candidate's exact column comes from a dry-run move (the mesh is changed
    // and restored), so each thread moves candidates on its own copy of the round's geometry and
    // scores the column at once against the frozen fit on the untouched mesh. Every dry run starts
    // from the same state, so the columns and scores are the serial ones. Two dimensions read the
    // column before the move.
    {
        ScopeTimer timer_candidates("  candidates (parallel)");
        if (mesh.dim == 3) {
            const int workers = std::max(1, std::min<int>(threads_, static_cast<int>(frontier.size() / 16) + 1));
            std::vector<std::unique_ptr<rect::Mesh>> copies(workers);
            { ScopeTimer timer_clones("    dry-run copies"); for (auto& copy : copies) copy = mesh.clone(); }
            #pragma omp parallel for num_threads(workers) schedule(dynamic, 4)
            for (long q = 0; q < static_cast<long>(frontier.size()); ++q) {
                const std::size_t i = static_cast<std::size_t>(q);
                const rect::SparseColumn column = copies[omp_get_thread_num()]->exact_candidate_column(frontier[i]);
                scored[i] = candidate_gain(mesh, frontier[i], column);
            }
        } else {
            #pragma omp parallel for num_threads(threads_) schedule(dynamic, 8)
            for (long q = 0; q < static_cast<long>(frontier.size()); ++q)
                scored[static_cast<std::size_t>(q)] = candidate_gain(mesh, frontier[static_cast<std::size_t>(q)]);
        }
    }
    std::size_t pushed = 0;
    for (std::size_t q = 0; q < frontier.size(); ++q) {
        const CandidateGain& exact = scored[q];
        if (exact.xTx <= 0) continue;
        ++context.census.attempted;
        if (!exact.ok || exact.gain_nats <= 1e-12) { ++context.census.unscoreable; continue; }
        ++context.census.scored;
        ++pushed;
        context.push(q, exact.centered_gate_z, exact.centered_gate_sd, exact);
    }
    std::printf("CENTERED round %d v %s scored %zu of %zu [%.2f s]\n", context.round, repr_double(record.pool_variance).c_str(),
                pushed, frontier.size(), record.seconds);
    std::fflush(stdout);
}

void Engine::Impl::write_metadata() {
    // the port loads the binary's one-line metadata, adds its keys and writes json.dumps(indent=2)
    if (!config_.output.prefix) return;
    JsonValue root = JsonValue::dict();
    root.set("schema_version", JsonValue::of(1));
    root.set("baseline_logit", JsonValue::of(dataset_.baseline_logit));
    root.set("wala_min", JsonValue::of(dataset_.bounds[0])); root.set("wala_max", JsonValue::of(dataset_.bounds[1]));
    root.set("wac_min", JsonValue::of(dataset_.bounds[2])); root.set("wac_max", JsonValue::of(dataset_.bounds[3]));
    if (dataset_.dim() == 3) {   // a three-dimensional mesh: the third axis and its range
        root.set("axis3", JsonValue::of(dataset_.axis3));
        root.set("axis3_min", JsonValue::of(dataset_.axis3_bounds[0])); root.set("axis3_max", JsonValue::of(dataset_.axis3_bounds[1]));
    }
    root.set("height_scale", JsonValue::of(1)); root.set("baseline_source", JsonValue::of(dataset_.baseline_source));
    root.set("blas_threads", JsonValue::of(1));
    JsonValue e = JsonValue::dict();
    e.set("fitting", JsonValue::of("penalized centered conditional-mode equation"));
    e.set("gate", JsonValue::of("first-order refit- and shrinkage-bias-corrected score"));
    e.set("proposal", JsonValue::of("local quadratic; nonlinear joint refit after admission"));
    e.set("variance_components", JsonValue::of("not separately decomposed in vertex dump"));
    e.set("selection", JsonValue::of("existing empirical-null rule; no adaptive FDR guarantee"));
    e.set("depth_variance", JsonValue::of("em"));
    JsonValue sq = JsonValue::dict(); sq.set("count_order", JsonValue::of(6)); sq.set("pool_order_base", JsonValue::of(24));   // pool_order_for(variance, 24) per solve
    e.set("surface_quadrature", sq);
    JsonValue vq = JsonValue::dict(); vq.set("count_order", JsonValue::of(8)); vq.set("pool_order", JsonValue::of(32));
    vq.set("early_search", JsonValue::of("surface moments; accurate secant reset before acceptance"));
    e.set("variance_quadrature", vq);
    e.set("surface_initializer", JsonValue::of("direct"));
    e.set("variance_schedule", JsonValue::of("full update every gate round"));
    e.set("variance_solver", JsonValue::of("alternating"));
    e.set("design", JsonValue::of("padded"));
    e.set("hierarchy", JsonValue::of("rectangular hierarchical surplus"));
    root.set("experimental_centered_score", e);
    if (last_levels_) {
        JsonValue levels = JsonValue::dict();
        levels.set("groups_file", JsonValue::of(env_or("DMESH_LEVEL_GROUPS", "")));
        JsonValue values = JsonValue::list();
        for (double v : *last_levels_) values.push(JsonValue::of(v));
        levels.set("levels", values);
        if (level_sizes_.size() > 1) {   // several factors: the levels above are their blocks in order
            JsonValue sizes = JsonValue::list();
            for (int s : level_sizes_) sizes.push(JsonValue::of(s));
            levels.set("factor_sizes", sizes);
        }
        root.set("centered_levels", levels);
    }
    JsonValue& inner = *root.get("experimental_centered_score");
    JsonValue local = JsonValue::dict();
    local.set("row_threshold_logits", JsonValue::of(row_threshold_)); local.set("certify_intermediate", JsonValue::of(false));
    local.set("variance_schedule", JsonValue::of("step")); local.set("step_hold", JsonValue::of(step_hold_));
    local.set("step_moments", JsonValue::of("coarse")); local.set("step_slope", JsonValue::of("newton"));
    local.set("initial_variance", JsonValue::of("step")); local.set("skip_unchanged", JsonValue::of(true));
    local.set("trim_caches", JsonValue::of(true));
    inner.set("local_rows", local);
    JsonValue tol = JsonValue::dict();
    tol.set("gate_rounds", JsonValue::of(gate_tolerance_)); tol.set("final_fit", JsonValue::of(final_tolerance_));
    tol.set("note", JsonValue::of("relative step in variance units; alternating solver only"));
    inner.set("variance_tolerance", tol);
    std::ofstream out(*config_.output.prefix + "_model.json");
    out << root.dumps(2) << "\n";
}

int Engine::Impl::run(Engine& engine) {
    if (dataset_.pool_count != static_cast<int>(dataset_.observations.size()))
        throw std::invalid_argument("the centered-score mesh needs one row per pool");
    if (config_.output.prefix) audit_path_ = *config_.output.prefix + "_centered_rounds.json";
    engine_ = &engine;
    ScopeTimer timer_pipeline("main: refine");
    const int result = refine(config_, dataset_, engine);
    timer_pipeline.stop();
    write_metadata();
    save_trace();
    return result;
}

Engine::Engine(const RunConfig& config, Dataset& dataset) : impl_(std::make_unique<Impl>(config, dataset)) {}
Engine::~Engine() = default;
int Engine::run() { return impl_->run(*this); }
void Engine::refit(rect::Mesh& mesh) { impl_->fit(mesh, false, Impl::final_tolerance_, false); }
void Engine::score(const std::vector<rect::Candidate>& frontier, ScoreContext& context) { impl_->score(frontier, context); }
double Engine::pool_variance() const { return impl_->centered_variance_; }

void Engine::note_round(std::size_t candidates, int admitted, int null_status, double null_sd, double null_mean) {
    auto& trace = impl_->trace_;
    if (trace.empty()) return;
    trace.back().set("candidates", JsonValue::of(static_cast<long long>(candidates)));
    trace.back().set("admitted", JsonValue::of(admitted));
    trace.back().set("null_status", JsonValue::of(null_status));
    trace.back().set("null_sd", JsonValue::of(null_sd)); trace.back().set("null_mean", JsonValue::of(null_mean));
    impl_->save_trace();
}

CandidateGain Engine::candidate_gain(const rect::Mesh& mesh, const rect::Candidate& candidate) {
    // The column the candidate's coefficient would have after its move; a candidate is never
    // free, so its surplus now and at the null are both zero.
    const rect::SparseColumn column = mesh.candidate_column(candidate);
    const double precision = 1.0 / mesh.tau_for_depth(candidate.depth);
    return impl_->candidate(mesh, column, 0.0, 0.0, precision);
}

CandidateGain Engine::candidate_gain(const rect::Mesh& mesh, const rect::Candidate& candidate, const rect::SparseColumn& column) {
    const double precision = 1.0 / mesh.tau_for_depth(candidate.depth);
    return impl_->candidate(mesh, column, 0.0, 0.0, precision);
}

void Engine::final_refit(rect::Mesh& mesh) {
    FitRecord record = impl_->fit(mesh, true, Impl::final_tolerance_, true);
    if (impl_->config_.output.prefix) {
        std::ofstream out(*impl_->config_.output.prefix + "_final_fit.json");
        out << record.json().dumps() << "\n";
    }
}

void Engine::audit(const rect::Mesh& mesh) const {
    const auto& state = impl_->state_;
    if (!state.have) throw std::runtime_error("no centered state for the final audit");
    for (int i = 0; i < mesh.n_points; ++i)
        if (std::fabs(mesh.predict_index(i) - state.prediction[static_cast<std::size_t>(i)]) > 1e-8)
            throw std::runtime_error("final centered predictions moved after the refit");
    if (state.fit.score_max > 2e-5) throw std::runtime_error("final centered equation is not stationary");
    std::printf("CENTERED final score audit %s\n", repr_double(state.fit.score_max).c_str()); std::fflush(stdout);
}

int run(const RunConfig& config, Dataset& dataset) {
    // One BLAS thread, whatever the environment says: OpenBLAS sums differently with a different
    // thread count, and the dumps must not depend on it (the port's launcher pins the same).
    openblas_set_num_threads(1);
    Engine engine(config, dataset);
    return engine.run();
}

}  // namespace estimator
