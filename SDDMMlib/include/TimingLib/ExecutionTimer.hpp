// ExecutionTimer.hpp --  measurement class
#define ExecutionTimer_HPP
#ifdef ExecutionTimer_HPP

#include <chrono>
#include <optional>
#include <vector>
// #if USE_CUDA
#include "CUDATimer.cuh"
#include "utils.h"
// #endif

class ExecutionTimer
{
    public:
        ExecutionTimer();
        ~ExecutionTimer();
        void start_gpu_run();
        void start_cpu_run();
        // void interrupt_gpu_run(); // Could be useful if we want to measure the time of a single kernel
        // void resume_gpu_run();
        void stop_gpu_run();
        void stop_cpu_run();
        void reset_run();
        void reset_all();
        std::vector<double> get_runtimes();

    private:
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
        std::chrono::time_point<std::chrono::high_resolution_clock> stop_time;
        bool running;
        std::vector<double> elapsed_times;
        // #if USE_CUDA
        EventTimer cuda_timer;
        cudaStream_t s;
        // #endif
};

#endif  // ExecutionTimer__HPP