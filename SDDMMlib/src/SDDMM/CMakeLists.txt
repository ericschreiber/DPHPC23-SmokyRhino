cmake_minimum_required(VERSION 3.21)
set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)

add_library(SDDMMlib
                naive_SDDMM.cpp
                naive_CPU_SDDMM.cpp
                
                naive_sequential_full_SDDMM_HOST/naive_sequential_full_SDDMM_HOST.cpp
                naive_sequential_sampled_SDDMM_HOST/naive_sequential_sampled_SDDMM_HOST.cpp
                )

if (USE_CUDA EQUAL 0) 
                
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
    # # Set compute architecture to V100 aka 70. This is needed for shared mem support to the limits.
    # # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    # # set_property(TARGET SDDMMlib PROPERTY CUDA_ARCHITECTURES 70)
    target_compile_options(SDDMMlib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode arch=compute_70,code=sm_70
        -use_fast_math
        -w
        --generate-line-info
        #--verbose
        -Xptxas -O3
        --ptxas-options=-v
        #-rdc=true
        --m64
        --fmad=true
        >)
    # # limit maxrregcount to 64 to avoid that the compiler uses too many registers if we use 1024 threads per block
    # target_compile_options(SDDMMlib PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas -maxrregcount=64>)

    add_compile_options("$<$COMPILE_LANGUAGE:CXX>:
        -std=c++17; 
        -v; 
        -Ofast; 
        -funroll-loop; 
        -flto; 
        -finline-functions;
        -march=native;
        -mtune=native;
        -fopenmp
        >")

    target_sources(SDDMMlib PRIVATE
                SM_L2/SM_L2_GPU.cpp
                SM_L2/SM_L2.cu
                
                coo_opt_vectorization_gpu/coo_opt_vectorization_SDDMM_GPU.cpp
                coo_opt_vectorization_gpu/coo_opt_vectorization_SDDMM.cu
                
                coo_opt_loop_unrolled_gpu/coo_opt_loop_unrolled_SDDMM_GPU.cpp
                coo_opt_loop_unrolled_gpu/coo_opt_loop_unrolled_SDDMM.cu

                coo_opt_vectorization_gpu/coo_opt_vectorization_SDDMM_GPU.cpp
                coo_opt_vectorization_gpu/coo_opt_vectorization_SDDMM.cu

                naive_csr_via_coo_gpu/naive_csr_via_coo_SDDMM_GPU.cpp
                naive_csr_via_coo_gpu/naive_csr_via_coo_SDDMM.cu

                naive_dense_dense_gpu/naive_SDDMM_GPU.cu
                naive_dense_dense_gpu/my_naive_sampling.cu
                
                better_naive_CSR_GPU/better_naive_CSR_SDDMM_GPU.cu
                better_naive_CSR_GPU/better_naive_CSR_SDDMM_GPU_kernel.cu

                semi_naive_CSR_GPU/semi_naive_CSR_SDDMM_GPU.cu
                semi_naive_CSR_GPU/semi_naive_CSR_SDDMM_GPU_kernel.cu

                naive_coo_gpu/naive_coo_SDDMM_GPU.cpp
                naive_coo_gpu/naive_coo_SDDMM.cu

                cache_coo_gpu/cache_coo_SDDMM_GPU.cpp
                cache_coo_gpu/cache_coo_SDDMM.cu

                cusparse_baseline/cusparse_baseline.cpp

                tiled_tiles/tiled_tiles.cpp
                tiled_tiles/tiled_tiles.cu

                merged/merged.cpp
                merged/merged.cu

                coo_tiling_naive_gpu/coo_tiling_naive_gpu_SDDMM_GPU.cpp
                coo_tiling_naive_gpu/coo_tiling_naive_gpu_SDDMM.cu
                )      


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
                            CUDA::cuda_driver
                            CUDA::cusparse)

endif()