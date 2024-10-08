# go to cmake folder    cd cmake
# cmake ..
# cmake --build . -j20 # -j20 is optional and limiting the number of threads, Compile time ~20 min 

cmake_minimum_required(VERSION 3.21)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # create compile_commands.json file

project(testcasesMatrixLib)

set(SDDMMLIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# suppress warnings
# add_definitions(-w)

# Link the executable
add_executable(test_io_dense test_IO_Dense.cpp)
add_executable(test_io_CSR test_IO_CSR.cpp)
add_executable(test_SDDMM test_SDDMM.cpp)
add_executable(test_COO test_COO.cpp)

# Link the library
target_link_libraries(test_io_dense MatrixLib)
target_link_libraries(test_io_CSR MatrixLib)
target_link_libraries(test_SDDMM 
                       MatrixLib
                       SDDMMlib
)
target_link_libraries(test_COO 
                       MatrixLib
)