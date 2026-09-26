#include "fit/config.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

bool env_flag(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) return false;
    const std::string text(value);
    return text.empty() || (text != "0" && text != "false" && text != "FALSE");
}

std::optional<std::string> env_string(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') return std::nullopt;
    return std::string(value);
}


template <typename T>
T parse_number(const char* text, const char* label);

template <>
double parse_number<double>(const char* text, const char* label) {
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !std::isfinite(value))
        throw std::runtime_error(std::string("invalid ") + label + ": " + text);
    return value;
}

template <>
int parse_number<int>(const char* text, const char* label) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
        throw std::runtime_error(std::string("invalid ") + label + ": " + text);
    return static_cast<int>(value);
}

}  // namespace

RunConfig parse_run_config(int argc, char** argv) {
    RunConfig config;
    if (argc > 2)
        throw std::runtime_error("the engine takes one optional argument, the seed (the five-argument form "
                                 "'sigma_u q scale points_per_face seed' was retired on 2026-09-22)");
    if (argc > 1) {
        const int parsed_seed = parse_number<int>(argv[1], "seed");
        if (parsed_seed < 0) throw std::runtime_error("seed must be nonnegative");
        config.seed = static_cast<unsigned>(parsed_seed);
    }

    config.data_file = env_string("DMESH_DATA");
    config.fit.heldout_split = env_flag("DMESH_SPLIT");
    config.fit.baseline = env_string("DMESH_BASELINE").value_or(config.fit.baseline);
    if (config.fit.baseline != "training" && config.fit.baseline != "external")
        throw std::runtime_error("DMESH_BASELINE must be training or external");
    config.output.prefix = env_string("DMESH_DUMP");
    if (const auto q = env_string("DMESH_Q")) {
        config.fit.fdr_level = parse_number<double>(q->c_str(), "DMESH_Q");
        if (!(config.fit.fdr_level > 0.0 && config.fit.fdr_level < 1.0))
            throw std::runtime_error("DMESH_Q must be strictly between 0 and 1");
    }

    return config;
}

void print_usage(const char* program_name) {
    std::fprintf(stderr,
                 "usage: %s [seed=7]\n",
                 program_name);
}
