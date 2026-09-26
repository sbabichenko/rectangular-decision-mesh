#pragma once
// Environment reads outside fit/config.cpp (which validates the supported
// surface). These are read where they are used, once, and are not validated:
// a misspelled name is inert.
#include <cstdlib>

inline const char* env_cstr(const char* name) { return std::getenv(name); }
inline bool env_present(const char* name) { return std::getenv(name) != nullptr; }
