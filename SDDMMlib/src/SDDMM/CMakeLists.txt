cmake_minimum_required(VERSION 3.25)
set(CMAKE_CXX_STANDARD 17)
find_package(CUDAToolkit REQUIRED)

project(SDDMM_CUDA LANGUAGES CXX CUDA)

set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)


# cuda
set_source_files_properties(*.cu PROPERTIES LANGUAGE CUDA)

add_library(SDDMMlib 
            naive_SDDMM.cpp
            naive_dense_dense_gpu/naive_SDDMM_GPU.cpp
            naive_dense_dense_gpu/naive_dense_dense.cu
            naive_dense_dense_gpu/my_naive_sampling.cu
            initial_implementation_gpu_jiela/initial_SDDMM_GPU_jiela.cpp
            initial_implementation_gpu_jiela/initial_SDDMM_GPU_kernel_jiela.cu
            initial_implementation_gpu_niklas/initial_SDDMM_GPU_niklas.cpp
            initial_implementation_gpu_niklas/initial_SDDMM_solver.cu
)      

# Add our own code
target_include_directories(SDDMMlib PUBLIC 
                            ../../include/SDDMM
                            ${THIRD_PARTY_ROOT}/include_gpu
                            )

target_link_libraries(SDDMMlib
                        MatrixLib
                        CUDA::cublas)

# Add third party code if needed here
