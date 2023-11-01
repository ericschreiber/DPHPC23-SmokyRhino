cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)
project(MatrixLib)

add_library(MatrixLib 
            CSRMatrix.cpp
            COOMatrix.cpp
            DenseMatrix.cpp
)      

# Add our own code
target_include_directories(MatrixLib PUBLIC 
                            ../../include/MatrixLib
                            )

target_link_libraries(MatrixLib SDDMMlib)

# Add third party code if needed here