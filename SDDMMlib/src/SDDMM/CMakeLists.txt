cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)


add_library(SDDMMlib STATIC
            naive_SDDMM.cpp
)      

# Add our own code
target_include_directories(SDDMMlib PUBLIC 
                            ../../include/SDDMM
                            )

target_link_libraries(SDDMMlib MatrixLib)

# Add third party code if needed here
