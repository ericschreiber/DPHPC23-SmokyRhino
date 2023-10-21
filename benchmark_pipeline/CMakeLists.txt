# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.26)
# if apple set -DCMAKE_APPLE_SILICON_PROCESSOR="arm64"
if (APPLE)
    message(STATUS "Apple Silicon detected")
    set(ENV{DCMAKE_APPLE_SILICON_PROCESSOR} "arm64")
else()
    message(STATUS "Intel detected.")
endif()

# Define your project and set its name
project(Pipeline)

# We need some header files for the compilation
include_directories("../SDDMMlib/include/MatrixLib")
include_directories("../SDDMMlib/include/SDDMM")
include_directories("../SDDMMlib/include/TimingLib")

# Add subdirectories for the library and executable
add_subdirectory(src)