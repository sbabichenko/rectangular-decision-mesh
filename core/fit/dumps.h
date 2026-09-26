#pragma once

#include "fit/observation.h"
#include "fit/config.h"

#include <fstream>
#include <string>
#include <vector>

namespace rect { class Mesh; }

// An output stream that flushes and checks itself on close; a failure (disk full, I/O
// error) is recorded for throw_if_output_failed(), since a destructor cannot throw.
class CheckedOutput : public std::ofstream {
public:
    explicit CheckedOutput(const std::string& filename) : std::ofstream(filename), name_(filename) {}
    CheckedOutput(CheckedOutput&& other) noexcept
        : std::ofstream(std::move(other)), name_(std::move(other.name_)), live_(other.live_) { other.live_ = false; }
    ~CheckedOutput();
    void disarm() { live_ = false; }     // before throwing on a failed open: nothing to check on close
private:
    std::string name_;
    bool live_ = true;
};
// Every dump stream flushes and checks itself when it closes; a write failure is recorded
// and thrown by throw_if_output_failed(), which the pipeline calls after the dumps.
void throw_if_output_failed();

// The dumps: the surface on a grid, the observations with their fitted rates, the vertex
// table, the cell table and the pool-effect posteriors at the fitted pool variance.
// One scored candidate of one gate round, as <prefix>_candidates.csv lists it: its position, the
// vertex it was (-1: an edge midpoint with no vertex yet), the segment it would bisect (axis 0 along
// loan age, 1 along coupon; its prior class; its would-be parents), the gate's coordinate and scale,
// the surplus and gain it proposed, and whether the round selected and admitted it.
struct CandidateRow {
    int round = 0;
    double x = 0.0, y = 0.0, x3 = 0.0;   // x3: the third coordinate, three-dimensional meshes only
    int vertex = -1, axis = -1, depth = 0, parent0 = -1, parent1 = -1;
    double z = 0.0, sd = 0.0, beta = 0.0, gain = 0.0;
    int selected = 0, admitted = 0;
};
void write_candidates(const OutputConfig& output, const std::vector<CandidateRow>& rows, int dim = 2);

void write_dumps(const OutputConfig& output,
                 const rect::Mesh& mesh,
                 const std::vector<Observation>& observations,
                 const std::vector<double>& x,
                 const std::vector<double>& y,
                 int pool_count,
                 double pool_variance);
