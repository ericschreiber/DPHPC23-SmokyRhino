cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)
project(MatrixLib)

set(SDDMMLIB_ROOT ../../SDDMMlib)

add_library(MatrixLib STATIC
            CSRMatrix.cpp
            DenseMatrix.cpp
)

# add_subdirectory(${SDDMMLIB_ROOT}/src ${CMAKE_CURRENT_BINARY_DIR}/SDDMMlib)
# target_include_directories(MatrixLib PUBLIC ${SDDMMLIB_ROOT}/include)

# Add our code as library
target_include_directories(MatrixLib PUBLIC ../include)

if (NOT TARGET SDDMMlib)
    message("Adding SDDMMlib")
    add_subdirectory(${SDDMMLIB_ROOT}/src ${CMAKE_CURRENT_BINARY_DIR}/SDDMMlib)
    target_include_directories(MatrixLib PUBLIC ${SDDMMLIB_ROOT}/include)
    target_link_libraries(MatrixLib 
            SDDMMlib
            )
endif()
