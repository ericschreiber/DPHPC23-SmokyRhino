cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)


add_library(TimingLib 
            ExecutionTimer.cpp
)      

# Add our own code
target_include_directories(TimingLib PUBLIC 
                            ../../include/TimingLib
                            )
