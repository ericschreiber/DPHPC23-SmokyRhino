cmake_minimum_required(VERSION 3.21)
set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)

if (USE_CUDA EQUAL 0)

    add_library(SDDMMlib
                naive_SDDMM.cpp
                naive_CPU_SDDMM.cpp)   
                
    target_include_directories(SDDMMlib PUBLIC 
                                ../../include/SDDMM
    )

    target_link_libraries(SDDMMlib
                            TimingLib
                            MatrixLib)


# Build without cuda but only the non-gpu parts
else()
    find_package(CUDAToolkit REQUIRED)

    project(SDDMM_CUDA LANGUAGES CXX CUDA)

    # cuda
    set_source_files_properties(*.cu PROPERTIES LANGUAGE CUDA)

    add_library(SDDMMlib 
                naive_SDDMM.cpp

                naive_CPU_SDDMM.cpp

                naive_dense_dense_gpu/naive_SDDMM_GPU.cpp
                naive_dense_dense_gpu/naive_dense_dense.cu
                naive_dense_dense_gpu/my_naive_sampling.cu

                naive_coo_gpu/naive_coo_SDDMM_GPU.cpp
                naive_coo_gpu/naive_coo_SDDMM.cu)      

    # Add our own code
    target_include_directories(SDDMMlib PUBLIC 
                                ../../include/SDDMM
                                ${THIRD_PARTY_ROOT}/include_gpu
                                ../../include/SDDMM/naive_dense_dense_gpu
                                /usr/local/cuda-12.1/targets/x86_64-linux/include)

    target_link_libraries(SDDMMlib
                        MatrixLib
                        TimingLib
                            CUDA::cublas
                            CUDA::cudart
                            CUDA::cuda_driver)

endif()
