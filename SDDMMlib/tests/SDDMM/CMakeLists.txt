# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.21)

# Define your project and set its name
project(testcasessddmm)

# Add subdirectories for the library and executable
if(USE_CUDA EQUAL 1)
add_subdirectory(naive_dense_dense_gpu)
add_subdirectory(initial_gpu/jiela)
#add_subdirectory(initial_gpu/niklas)
endif()