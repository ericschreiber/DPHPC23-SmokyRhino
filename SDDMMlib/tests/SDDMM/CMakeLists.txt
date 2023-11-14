# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.21)

# Define your project and set its name
project(testcasessddmm)

# Add subdirectories for the library and executable
if(USE_CUDA EQUAL 1)
add_subdirectory(naive_dense_dense_gpu)

add_executable(test_naive_coo_SDDMM_GPU
               test_naive_coo_SDDMM_GPU.cpp
)

add_executable(test_cache_coo_SDDMM_GPU
               test_cache_coo_SDDMM_GPU.cpp
)
target_link_libraries(test_naive_coo_SDDMM_GPU 
                       MatrixLib
                       SDDMMlib
)

target_link_libraries(test_cache_coo_SDDMM_GPU 
                       MatrixLib
                       SDDMMlib
)

endif()
# Link the executable
add_executable(test_naive_CPU_SDDMM test_naive_CPU_SDDMM.cpp)
target_link_libraries(test_naive_CPU_SDDMM
                        MatrixLib
                        SDDMMlib)
