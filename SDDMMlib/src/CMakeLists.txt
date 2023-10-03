cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)
# if apple set -DCMAKE_APPLE_SILICON_PROCESSOR="arm64"
if (APPLE)
    message(STATUS "Apple Silicon detected")
    set(ENV{DCMAKE_APPLE_SILICON_PROCESSOR} "arm64")
else()
    message(STATUS "Intel detected.")
endif()
project(SDDMMlib)

add_library(SDDMMlib STATIC
            SDDMM/naive_SDDMM.cpp 
            MatrixLib/CSRMatrix.cpp
            MatrixLib/DenseMatrix.cpp
)      

# Add our own code
target_include_directories(SDDMMlib PUBLIC 
                            ../include)

# Add third party code if needed here
