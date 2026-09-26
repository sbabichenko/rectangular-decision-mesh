// Scope timers for the centered engine, printed at exit when DMESH_CENTERED_PROFILE=1.
#pragma once
#include <chrono>
#include <string>

namespace estimator {
void profile_add(const char* name, double seconds);
bool profile_enabled();
struct ScopeTimer {
    const char* name; std::chrono::steady_clock::time_point start; bool on;
    explicit ScopeTimer(const char* n) : name(n), start(std::chrono::steady_clock::now()), on(profile_enabled()) {}
    void stop() { if (on) profile_add(name, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()); on = false; }
    ~ScopeTimer() { stop(); }
};
}
