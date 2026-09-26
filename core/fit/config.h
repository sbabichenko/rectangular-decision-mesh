#pragma once

#include <optional>
#include <string>

// The run's configuration: the one positional argument (the seed) and the DMESH_* environment.
struct FitConfig {
    bool heldout_split = false;
    // Where the pooled level comes from. "training": the training rows' pooled
    // logit, added to any offset column. "external": the offset column already
    // carries the whole baseline logit and enters unchanged; nothing is added.
    std::string baseline = "training";
    // The gate's false discovery rate (DMESH_Q), strictly between 0 and 1.
    double fdr_level = 0.10;
};

struct OutputConfig {
    std::optional<std::string> prefix;   // DMESH_DUMP
};

struct RunConfig {
    std::optional<std::string> data_file;   // DMESH_DATA, required
    unsigned seed = 7;                      // drives the held-out split
    FitConfig fit;
    OutputConfig output;
};

RunConfig parse_run_config(int argc, char** argv);
void print_usage(const char* program_name);
