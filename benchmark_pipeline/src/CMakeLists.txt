cmake_minimum_required(VERSION 3.26)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # create compile_commands.json file

# suppress warnings
# add_definitions(-w)

add_library(BenchmarkPipelinelib
            runner.cpp
            config_helper.cpp
            )
target_include_directories(BenchmarkPipelinelib PUBLIC 
                            ../include
                            )
target_link_libraries(BenchmarkPipelinelib
                        MatrixLib
                        SDDMMlib
                        TimingLib
                        )

# Link the executable
add_executable(benchmark_pipeline benchmark_pipeline.cpp)


# Link the library
target_link_libraries(benchmark_pipeline 
                        MatrixLib
                        SDDMMlib
                        BenchmarkPipelinelib
)
