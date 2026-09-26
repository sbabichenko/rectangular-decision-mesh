#include "estimator/engine.h"
#include "estimator/profile.h"
#include "fit/dataset.h"
#include "fit/config.h"

#include <cstdio>
#include <exception>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

int main(int argc, char** argv) {
#if defined(__GLIBC__)
    // The parallel candidate scoring makes many short-lived allocations on every thread; glibc's
    // default of one arena per thread kept about 7 MB of fragments per thread (123 MB, against
    // 79 MB with two arenas, at 8 threads on the pooled benchmark) at no cost in time.
    mallopt(M_ARENA_MAX, 2);
#endif
    RunConfig config;
    try {
        config = parse_run_config(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        print_usage(argv[0]);
        return 2;
    }
    try {
        estimator::ScopeTimer timer_load("main: build_dataset");
        Dataset dataset = build_dataset(config);
        timer_load.stop();
        return estimator::run(config, dataset);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
