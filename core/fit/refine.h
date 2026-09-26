#pragma once

#include "fit/dataset.h"
#include "fit/config.h"

namespace estimator { class Engine; }

// Fit the mesh: the coarse geometry, the gate rounds, the final retirement and refit, the
// report and the dumps. Returns the process exit status.
int refine(const RunConfig& config, Dataset& dataset, estimator::Engine& engine);
