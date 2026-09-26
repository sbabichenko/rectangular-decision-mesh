cmake_minimum_required(VERSION 3.14)
project(rectangular_decision_mesh CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(rectcore rect_mesh.cpp lfdr.cpp)

add_executable(ginnie_rect ginnie_main.cpp)
target_link_libraries(ginnie_rect rectcore)

enable_testing()
add_executable(test_geometry test_geometry.cpp)
target_link_libraries(test_geometry rectcore)
add_test(NAME geometry COMMAND test_geometry)
