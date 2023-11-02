cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD 17)

set(THIRD_PARTY_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party)

add_library(TimingLib 
            ExecutionTimer.cpp
)      

# Add our own code
if (USE_CUDA EQUAL 0)
    target_include_directories(TimingLib PUBLIC 
                                ../../include/TimingLib
                                )
else()
    find_package(CUDAToolkit REQUIRED)

    project(SDDMM_CUDA LANGUAGES CXX CUDA)

    # cuda
    set_source_files_properties(*.cu PROPERTIES LANGUAGE CUDA)


    target_include_directories(TimingLib PUBLIC 
                                ../../include/TimingLib
                                ${THIRD_PARTY_ROOT}/include_gpu
                                )
    target_link_libraries(TimingLib
                            CUDA::cuda_driver)
endif()