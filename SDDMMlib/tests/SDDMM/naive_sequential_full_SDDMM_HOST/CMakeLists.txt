# go to cmake folder    cd cmake
# cmake ..
# cmake --build . -j20 # -j20 is optional and limiting the number of threads, Compile time ~20 min 

cmake_minimum_required(VERSION 3.21)

project(naive_sequential_full_SDDMM_HOST)



# Link the executable

add_executable(test_naive_sequential_full_SDDMM_HOST test_naive_sequential_full_SDDMM_HOST.cpp)


# Link the library
target_link_libraries(test_naive_sequential_full_SDDMM_HOST
                        MatrixLib
                        SDDMMlib)
