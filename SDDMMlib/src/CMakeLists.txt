cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(SDDMMlib)

set(MATRIXLIB_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../MatrixLib)

add_library(SDDMMlib STATIC
            naive_SDDMM.cpp
)      

# add_subdirectory(${MATRIXLIB_ROOT}/src ${CMAKE_CURRENT_BINARY_DIR}/MatrixLib)
# target_include_directories(SDDMMlib PUBLIC MATRIXLIB_ROOT/include)

# Add our own code
target_include_directories(SDDMMlib PUBLIC ../include)

if (NOT TARGET MatrixLib)
    message(STATUS "MatrixLib not found, adding it as a subdirectory")
    # Add the Matrix library
    add_subdirectory(${MATRIXLIB_ROOT}/src ${CMAKE_CURRENT_BINARY_DIR}/MatrixLib)
    target_include_directories(SDDMMlib PUBLIC MATRIXLIB_ROOT/include)
    target_link_libraries(SDDMMlib MatrixLib)
endif()
