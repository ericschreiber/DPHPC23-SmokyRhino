# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.25)

# Define your project and set its name
project(testcasessddmm)

# Add subdirectories for the library and executable
add_subdirectory(naive_sequential_full_SDDMM_HOST)
add_subdirectory(naive_sequential_sampled_SDDMM_HOST)

