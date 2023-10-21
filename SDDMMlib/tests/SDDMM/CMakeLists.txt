# go to cmake folder    cd cmake
# cmake ..
# cmake --build . -j20 # -j20 is optional and limiting the number of threads, Compile time ~20 min 

cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # create compile_commands.json file

project(testcasesCPUSDDMM)

set(SDDMMLIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

# suppress warnings
# add_definitions(-w)

# Link the executable

add_executable(test_naive_CPU_SDDMM test_naive_CPU_SDDMM.cpp)

# adding the MatrixLib library
# add_subdirectory(${SDDMMLIB_ROOT}/src ${CMAKE_CURRENT_BINARY_DIR}/SDDMMlib)
# target_include_directories(test_io_dense PRIVATE ${SDDMMLIB_ROOT}/include)
# target_include_directories(test_io_CSR PUBLIC ${SDDMMLIB_ROOT}/include)
# target_include_directories(test_SDDMM PUBLIC ${SDDMMLIB_ROOT}/include)

# Link the library
target_link_libraries(test_io_dense MatrixLib)
target_link_libraries(test_io_CSR MatrixLib)
target_link_libraries(test_naive_CPU_SDDMM 
                        MatrixLib
                        SDDMMlib
)