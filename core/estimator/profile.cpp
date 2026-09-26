#include "estimator/profile.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>
#include <algorithm>
namespace estimator {
namespace {
std::map<std::string, std::pair<double, long>>& table() { static std::map<std::string, std::pair<double, long>> t; return t; }
std::mutex& lock() { static std::mutex m; return m; }

// DMESH_MEM_PROFILE (with the profile on): the peak resident size's growth, credited to the timer
// that was ending when it was seen (inner timers end first, so they claim their own growth)
std::map<std::string, long>& growth() { static std::map<std::string, long> g; return g; }
long peak_kb() {
    std::FILE* f = std::fopen("/proc/self/status", "r"); if (!f) return 0;
    char line[256]; long kb = 0;
    while (std::fgets(line, sizeof line, f)) if (std::sscanf(line, "VmHWM: %ld kB", &kb) == 1) break;
    std::fclose(f); return kb;
}
bool mem_enabled() { static const bool on = std::getenv("DMESH_MEM_PROFILE") != nullptr; return on; }
long& last_peak() { static long v = 0; return v; }
struct Printer { Printer() { table(); lock(); growth(); last_peak(); } ~Printer() {
    if (!profile_enabled()) return;
    std::vector<std::pair<std::string, std::pair<double, long>>> rows(table().begin(), table().end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    std::fprintf(stderr, "[centered profile]\n");
    for (const auto& [name, v] : rows) std::fprintf(stderr, "  %-34s %8.3f s %8ld calls\n", name.c_str(), v.first, v.second);
    if (mem_enabled()) {
        std::vector<std::pair<std::string, long>> g(growth().begin(), growth().end());
        std::sort(g.begin(), g.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::fprintf(stderr, "[peak memory growth by the timer ending when it was seen] final peak %ld MB\n", peak_kb() / 1024);
        for (const auto& [name, kb] : g) if (kb > 512) std::fprintf(stderr, "  %-34s %8.1f MB\n", name.c_str(), kb / 1024.0);
    }
} } printer;
}
bool profile_enabled() { static const bool on = std::getenv("DMESH_CENTERED_PROFILE") != nullptr; return on; }
void profile_add(const char* name, double seconds) {
    std::lock_guard<std::mutex> g(lock()); auto& e = table()[name]; e.first += seconds; ++e.second;
    if (mem_enabled()) { const long now = peak_kb(); if (now > last_peak()) { growth()[name] += now - last_peak(); last_peak() = now; } }
}
}
