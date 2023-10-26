cmake_minimum_required(VERSION 3.25)

project(MatrixLib)


add_library(MatrixLib 
            CSRMatrix.cpp
            DenseMatrix.cpp
)      

# Add our own code
target_include_directories(MatrixLib PUBLIC 
                            ../../include/MatrixLib
                            )

target_link_libraries(MatrixLib SDDMMlib)

# Add third party code if needed here
