# go to cmake folder    cd cmake
# cmake ..
# cmake --build . -j20 # -j20 is optional and limiting the number of threads, Compile time ~20 min 

cmake_minimum_required(VERSION 3.25)

project(testcasesinitialGPUniklas)



# Link the executable
add_executable(test_compile_niklas test_compile_niklas.cpp)


# Link the library
target_link_libraries(test_compile_niklas 
                        MatrixLib
                        SDDMMlib
)
