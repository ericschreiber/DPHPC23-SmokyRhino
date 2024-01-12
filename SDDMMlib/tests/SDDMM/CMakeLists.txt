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
add_executable(test_coo_opt_vectorization_SDDMM_GPU
               test_coo_opt_vectorization_SDDMM_GPU.cpp
)
add_executable(test_coo_opt_loop_unrolled_SDDMM_GPU
               test_coo_opt_loop_unrolled_SDDMM_GPU.cpp
)
add_executable(test_cusparse_baseline
               test_cusparse_baseline.cpp
)
add_executable(test_SM_L2
               test_SM_L2.cpp
)

target_link_libraries(test_naive_coo_SDDMM_GPU 
                       MatrixLib
                       SDDMMlib)
target_link_libraries(test_coo_opt_vectorization_SDDMM_GPU 
                       MatrixLib
                       SDDMMlib)
target_link_libraries(test_coo_opt_loop_unrolled_SDDMM_GPU
                          MatrixLib
                          SDDMMlib)

target_link_libraries(test_cusparse_baseline
                            MatrixLib
                          SDDMMlib)
                          
target_link_libraries(test_SM_L2
                          MatrixLib
                            SDDMMlib)

endif()


# Link the executable
