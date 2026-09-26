// The centred-score estimator: the surface fit at a fixed topology, the pool variance, and
// the candidate scores the refinement driver (refine.cpp) selects on. The triangular engine's,
// with the rectangular mesh's surplus coordinates underneath (docs/RECT_ESTIMATOR.md).
#pragma once
#include "fit/dataset.h"
#include "fit/config.h"
#include "mesh/rect.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace estimator {

// A candidate's profile from the engine: the centred score of the candidate's column against
// the frozen fit, its penalised gain and the surplus it proposes.
struct CandidateGain {
    bool ok = false;
    double null_units = 0.0;
    double optimum_units = 0.0;
    double gain_nats = 0.0;
    double signed_root_gain = 0.0;
    double raw_gain_nats = 0.0;
    double information_at_null = 0.0;
    double data_information_optimum = 0.0;
    double lambda_logit = 0.0;
    int fit_points = 0;
    double xTx = 0.0, keff = 0.0;          // the column's information at the frozen fit, and its effective rows
    // the centered gate's coordinate: standardized centered score and its scale
    double centered_gate_z = 0.0, centered_gate_sd = 0.0;
};

// The run's census of candidate profiles, reported once at the end.
struct CandidateCensus {
    long long attempted = 0;
    long long scored = 0;
    long long unscoreable = 0;
    long long reprofiled = 0;             // admitted candidates re-profiled at the commit state
    long long reprofile_rejected = 0;
};

// What one round's scoring reads and writes: the mesh, the round number, the census, and the
// sink for each scoreable candidate (its index in the frontier, its gate coordinate, that
// coordinate's scale, and the profile).
struct ScoreContext {
    rect::Mesh& mesh;
    int round;
    CandidateCensus& census;
    std::function<void(std::size_t index, double z, double sd, const CandidateGain&)> push;
};

class Engine {
public:
    Engine(const RunConfig& config, Dataset& dataset);
    ~Engine();

    // Fit the mesh end to end and write the dumps. Returns the process exit status.
    int run();

    // Refit the surface at the current topology (skipped when nothing moved).
    void refit(rect::Mesh& mesh);
    // Refit at the gate tolerance, then score the frontier against the frozen fit.
    void score(const std::vector<rect::Candidate>& frontier, ScoreContext& context);
    // The round's selection, appended to the audit trace.
    void note_round(std::size_t candidates, int admitted, int null_status, double null_sd, double null_mean);
    // The candidate's profile at the live state: its column against the frozen fit.
    CandidateGain candidate_gain(const rect::Mesh& mesh, const rect::Candidate& candidate);
    // the same on a column already computed (three dimensions: the exact column of a dry-run move)
    CandidateGain candidate_gain(const rect::Mesh& mesh, const rect::Candidate& candidate, const rect::SparseColumn& column);
    // The certified final fit at the frozen topology; writes the final-fit record.
    void final_refit(rect::Mesh& mesh);
    // The fixed-point audit of the final fit.
    void audit(const rect::Mesh& mesh) const;
    double pool_variance() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

int run(const RunConfig& config, Dataset& dataset);

}  // namespace estimator
