# Top most CMakeLists.txt file for the project
cmake_minimum_required(VERSION 3.21)
# if apple set -DCMAKE_APPLE_SILICON_PROCESSOR="arm64"
if (APPLE)
    message(STATUS "Apple Silicon detected")
    set(ENV{DCMAKE_APPLE_SILICON_PROCESSOR} "arm64")
else()
    message(STATUS "Intel detected.")
endif()

if (USE_CUDA EQUAL 0)
    message(STATUS "CUDA is disabled")
elseif (USE_CUDA EQUAL 1)
    message(STATUS "CUDA is enabled")
else()
    message(FATAL_ERROR "USE_CUDA must be 0 or 1")
endif()
add_definitions(-DUSE_CUDA=${USE_CUDA})

# Define your project and set its name
project(SmokeyRhino VERSION 0.1.0)

# Add subdirectories for the library and executable
add_subdirectory(SDDMMlib)
add_subdirectory(benchmark_pipeline)