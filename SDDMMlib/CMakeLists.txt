# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 20)

# USE_CUDA takes values 0 or 1
if (USE_CUDA EQUAL 1)
    message(STATUS "Building with CUDA")
    # set cuda archtitecture for V100
    if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
        set(CMAKE_CUDA_ARCHITECTURES 70)
    endif()
elseif(USE_CUDA EQUAL 0)
    message(STATUS "Building without CUDA")
else()
    message(FATAL_ERROR "USE_CUDA must be set to 0 or 1")
endif()


# if apple set -DCMAKE_APPLE_SILICON_PROCESSOR="arm64"
if (APPLE)
    message(STATUS "Apple Silicon detected")
    set(ENV{DCMAKE_APPLE_SILICON_PROCESSOR} "arm64")
else()
    message(STATUS "Intel detected.")
endif()

# Define your project and set its name
project(SDDMM)

# Add subdirectories for the library and executable
add_subdirectory(src)
add_subdirectory(tests)
