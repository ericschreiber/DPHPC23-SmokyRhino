# go to cmake folder    cd cmake
# cmake ..
# cmake --build . -j20 # -j20 is optional and limiting the number of threads, Compile time ~20 min 

cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # create compile_commands.json file
project(testcases)

# suppress warnings
# add_definitions(-w)

# Link the executable
add_executable(test_io_dense test_IO_Dense.cpp)
add_executable(test_io_CSR test_IO_CSR.cpp)
add_executable(test_SDDMM test_SDDMM.cpp)

# adding the MatrixLib library
add_subdirectory(../src ${CMAKE_CURRENT_BINARY_DIR}/MatrixLib)
target_include_directories(test_io_dense PUBLIC ../include)
target_include_directories(test_io_CSR PUBLIC ../include)
target_include_directories(test_SDDMM PUBLIC ../include)

# Link the library
target_link_libraries(test_io_dense MatrixLib)
target_link_libraries(test_io_CSR MatrixLib)
target_link_libraries(test_SDDMM MatrixLib)
