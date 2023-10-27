# go to cmake folder    cd cmake
# cmake ..
# cmake --build . -j20 # -j20 is optional and limiting the number of threads, Compile time ~20 min 

cmake_minimum_required(VERSION 3.21)

project(testcasesinitialGPUjiela)



# Link the executable
add_executable(test_compile_jiela test_compile_jiela.cpp)


# Link the library
target_link_libraries(test_compile_jiela 
                        MatrixLib
                        SDDMMlib
)
