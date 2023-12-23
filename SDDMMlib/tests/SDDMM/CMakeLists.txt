# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.21)

# Define your project and set its name
project(testcasessddmm)

# Add subdirectories for the library and executable
if(USE_CUDA EQUAL 1)
add_subdirectory(naive_dense_dense_gpu)
add_subdirectory(semi_naive_CSR_GPU)


add_executable(test_naive_coo_SDDMM_GPU
               test_naive_coo_SDDMM_GPU.cpp
)
add_executable(test_cache_coo_SDDMM_GPU
               test_cache_coo_SDDMM_GPU.cpp
)
add_executable(test_tiled_tiles
               test_tiled_tiles.cpp
)

add_executable(test_merged
               test_merged.cpp
)

add_executable(test_naive_csr_via_coo_SDDMM_GPU
               test_naive_csr_via_coo_SDDMM_GPU.cpp
)
add_executable(test_coo_opt_vectorization_SDDMM_GPU
               test_coo_opt_vectorization_SDDMM_GPU.cpp
)
add_executable(test_coo_opt_loop_unrolled_SDDMM_GPU
               test_coo_opt_loop_unrolled_SDDMM_GPU.cpp
)
add_executable(test_coo_tiling_naive_gpu
               test_coo_tiling_naive_gpu.cpp
)
add_executable(test_SM_L2
               test_SM_L2.cpp
)

target_link_libraries(test_naive_coo_SDDMM_GPU 
                       MatrixLib
                       SDDMMlib)
target_link_libraries(test_naive_csr_via_coo_SDDMM_GPU 
                       MatrixLib
                       SDDMMlib)
target_link_libraries(test_coo_opt_vectorization_SDDMM_GPU 
                       MatrixLib
                       SDDMMlib)
target_link_libraries(test_coo_opt_loop_unrolled_SDDMM_GPU
                          MatrixLib
                          SDDMMlib)
target_link_libraries(test_cache_coo_SDDMM_GPU 
                       MatrixLib
                       SDDMMlib)

target_link_libraries(test_tiled_tiles
                          MatrixLib
                          SDDMMlib)

target_link_libraries(test_merged                       
                          MatrixLib
                          SDDMMlib)

target_link_libraries(test_coo_tiling_naive_gpu
                          MatrixLib
                          SDDMMlib)
                          
target_link_libraries(test_SM_L2
                          MatrixLib
                          SDDMMlib)

endif()

add_subdirectory(naive_sequential_full_SDDMM_HOST)
add_subdirectory(naive_sequential_sampled_SDDMM_HOST)


# Link the executable
add_executable(test_naive_CPU_SDDMM test_naive_CPU_SDDMM.cpp)

target_link_libraries(test_naive_CPU_SDDMM
                        MatrixLib
                        SDDMMlib)
