#include "ExecutionTimer.hpp"

#include <cassert>
#include <iostream>

#include "CUDATimer.cuh"
#include "utils.h"

#if USE_CUDA
ExecutionTimer::ExecutionTimer()
{
    running = false;
    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));
    EventTimer cuda_timer;
    std::cout << "cuda_timer" << std::endl;
}

ExecutionTimer::~ExecutionTimer()
{
    std::cout << "destroy s" << std::endl;
    CUDA_CHECK(cudaStreamDestroy(s));
}

void ExecutionTimer::start_cpu_run()
{
    assert(!running && "Timer already running");
    running = true;
    start_time = std::chrono::high_resolution_clock::now();
}

void ExecutionTimer::stop_cpu_run()
{
    assert(running && "Timer not running (cpu)");
    stop_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time - start_time).count();
    elapsed_times.push_back(elapsed_time);
    running = false;
}
#else
ExecutionTimer::ExecutionTimer()
{
    running = false;
}

ExecutionTimer::~ExecutionTimer()
{
}

void ExecutionTimer::start_cpu_run()
{
    assert(!running && "Timer already running");
    running = true;
    start_time = std::chrono::high_resolution_clock::now();
}

void ExecutionTimer::stop_cpu_run()
{
    assert(running && "Timer not running");
    stop_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time - start_time).count();
    elapsed_times.push_back(elapsed_time);
    running = false;
}
#endif

#if USE_CUDA
void ExecutionTimer::start_gpu_run()
{
    assert(!running && "Timer already running");
    running = true;
    cuda_timer.start(0);
}

void ExecutionTimer::stop_gpu_run()
{
    assert(running && "Timer not running (gpu)");
    cuda_timer.stop(0);
    elapsed_times.push_back(cuda_timer.elapsed());
    running = false;
}
#else
void ExecutionTimer::start_gpu_run()
{
    assert(false && "Not implemented yet");
    if (running)
    {
        std::cout << "ExecutionTimer: Warning: Timer already running. Starting a new run." << std::endl;
    }
    //
}

void ExecutionTimer::stop_gpu_run()
{
    assert(false && "Not implemented yet");
    if (!running)
    {
        std::cout << "ExecutionTimer: Warning: Timer not running. Stopping the timer has no effect." << std::endl;
    }
    //
}
#endif

void ExecutionTimer::reset_run()
{
    running = false;
}

void ExecutionTimer::reset_all()
{
    running = false;
    elapsed_times.clear();
}

std::vector<double> ExecutionTimer::get_runtimes()
{
    return elapsed_times;
}