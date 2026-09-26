cmake_minimum_required(VERSION 3.14)
project(RectangularDecisionMesh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DMESH_NATIVE "Enable architecture-specific release tuning" ON)
option(DMESH_LTO "Enable interprocedural optimization in Release builds" ON)

# Sanitizers, off by default. Give it a comma-separated -fsanitize list:
#
#   cmake -S core -B build-asan -DCMAKE_BUILD_TYPE=Debug \
#         -DDMESH_SANITIZE=address,undefined
#
# Applies to every target, including the tests, so `ctest` runs sanitized too.
# Use with Debug or RelWithDebInfo: a sanitized Release build reports line
# numbers that do not correspond to anything you can read.
set(DMESH_SANITIZE "" CACHE STRING
    "Comma-separated sanitizers to enable, e.g. address,undefined")

# One place for the flags every target shares, rather than a copy per target.
add_library(dmesh_flags INTERFACE)
target_include_directories(dmesh_flags INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(dmesh_flags INTERFACE
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:GNU,Clang>>:-O3>
  # No fused multiply-add contraction: with -march=native the compiler would
  # otherwise fuse a*b+c differently depending on inlining (LTO on or off),
  # and the held-out split's tie logic then consumes the seeded RNG in a
  # different order, so two builds of the same source gave different fits.
  # Off, every build reproduces the pinned reference bit for bit.
  $<$<CXX_COMPILER_ID:GNU,Clang>:-ffp-contract=off>
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/O2;/W4>)

if(DMESH_SANITIZE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Sanitizers enabled: ${DMESH_SANITIZE}")
    target_compile_options(dmesh_flags INTERFACE
      -fsanitize=${DMESH_SANITIZE} -fno-omit-frame-pointer -g)
    target_link_options(dmesh_flags INTERFACE -fsanitize=${DMESH_SANITIZE})
    if(DMESH_NATIVE)
      message(STATUS "Sanitizers on: disabling DMESH_NATIVE for readable reports")
      set(DMESH_NATIVE OFF)
    endif()
    if(DMESH_LTO)
      message(STATUS "Sanitizers on: disabling DMESH_LTO")
      set(DMESH_LTO OFF)
    endif()
  else()
    message(WARNING "DMESH_SANITIZE ignored: unsupported compiler ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()

# Everything but main, once. The executable and every test link this library,
# so no source list is transcribed by hand.
add_library(dmesh_core STATIC
  estimator/engine.cpp
  estimator/json.cpp
  estimator/numeric.cpp
  estimator/padded.cpp
  estimator/profile.cpp
  estimator/quadrature.cpp
  estimator/score.cpp
  estimator/solver.cpp
  estimator/system.cpp
  estimator/variance.cpp
  fit/config.cpp
  fit/dataset.cpp
  fit/dumps.cpp
  fit/gate.cpp
  fit/refine.cpp
  mesh/rect.cpp)
target_link_libraries(dmesh_core PUBLIC dmesh_flags)
# Candidate gains are profiled in parallel (independent per candidate, so the
# output does not depend on the thread count). OMP_NUM_THREADS sets the count.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(dmesh_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# LAPACK/OpenBLAS for the engine's dense algebra.
find_library(DMESH_LAPACKE lapacke)
find_library(DMESH_OPENBLAS openblas)
if(DMESH_LAPACKE AND DMESH_OPENBLAS)
  target_link_libraries(dmesh_core PUBLIC ${DMESH_LAPACKE} ${DMESH_OPENBLAS})
  message(STATUS "dense algebra: LAPACK (${DMESH_LAPACKE})")
else()
  message(FATAL_ERROR "the centered estimator needs LAPACKE and OpenBLAS (the same library numpy and scipy use)")
endif()

add_executable(rect_mesh main.cpp)
target_link_libraries(rect_mesh PRIVATE dmesh_core)

if(DMESH_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(dmesh_core PRIVATE $<$<CONFIG:Release>:-march=native;-mtune=native>)
endif()
if(DMESH_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT DMESH_IPO_OK OUTPUT DMESH_IPO_ERROR)
  if(DMESH_IPO_OK)
    set_property(TARGET rect_mesh dmesh_core PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  else()
    message(WARNING "IPO/LTO unavailable: ${DMESH_IPO_ERROR}")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  foreach(t rect_geometry)
    add_executable(test_${t} tests/test_${t}.cpp)
    target_link_libraries(test_${t} PRIVATE dmesh_core)
    add_test(NAME ${t} COMMAND test_${t})
  endforeach()
endif()
