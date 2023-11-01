cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)


add_library(SDDMMlib 
            naive_SDDMM.cpp
            naive_sequential_full_SDDMM_HOST/naive_sequential_full_SDDMM_HOST.cpp
            naive_sequential_sampled_SDDMM_HOST/naive_sequential_sampled_SDDMM_HOST.cpp
)      

# Add our own code
target_include_directories(SDDMMlib PUBLIC 
                            ../../include/SDDMM
                            )

target_link_libraries(SDDMMlib 
                        MatrixLib
                        TimingLib)

# Add third party code if needed here
