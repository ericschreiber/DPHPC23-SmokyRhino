# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.26)

# Define your project and set its name
project(tests)

# Add subdirectories for the library and executable
add_subdirectory(MatrixLib)
add_subdirectory(SDDMM)
