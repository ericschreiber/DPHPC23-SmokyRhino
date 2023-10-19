#include "ExecutionTimer.hpp"

#include <cassert>
#include <iostream>

ExecutionTimer::ExecutionTimer()
{
    running = false;
}

ExecutionTimer::~ExecutionTimer()
{
}

void ExecutionTimer::start_gpu_run()
{
    assert(false && "Not implemented yet");
    if (running)
    {
        std::cout << "ExecutionTimer: Warning: Timer already running. Starting a new run." << std::endl;
    }
    // ...
}

void ExecutionTimer::start_cpu_run()
{
    assert(!running && "Timer already running");
    start_time = std::chrono::high_resolution_clock::now();
    running = true;
}

void ExecutionTimer::stop_gpu_run()
{
    assert(false && "Not implemented yet");
    if (!running)
    {
        std::cout << "ExecutionTimer: Warning: Timer not running. Stopping the timer has no effect." << std::endl;
    }
    // ...
}

void ExecutionTimer::stop_cpu_run()
{
    assert(running && "Timer not running");
    stop_time = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(stop_time - start_time).count();
    elapsed_times.push_back(elapsed_time);
    running = false;
}

void ExecutionTimer::reset_run()
{
    running = false;
}

void ExecutionTimer::reset_all()
{
    running = false;
    elapsed_times.clear();
}

std::vector<double> ExecutionTimer::get_runs()
{
    return elapsed_times;
}
