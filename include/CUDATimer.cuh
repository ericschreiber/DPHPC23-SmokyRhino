// File       : CUDATimer.cuh
// Created    : Sat May 01 2021 09:49:57 AM (+0200)
// Author     : Fabian Wermelinger
// Description: CUDA event timer
// Copyright 2021 ETH Zurich. All Rights Reserved.
#ifndef CUDATIMER_CUH_OBWAUHTQ
#define CUDATIMER_CUH_OBWAUHTQ

#include <cassert>

// timer
// http://stackoverflow.com/questions/6959213/timing-a-cuda-application-using-events
class EventTimer {
public:
  EventTimer() : started_(false), stopped_(false) {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~EventTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start(cudaStream_t s = 0) {
    cudaEventRecord(start_, s);
    started_ = true;
    stopped_ = false;
  }

  void stop(cudaStream_t s = 0) {
    assert(started_);
    cudaEventRecord(stop_, s);
    started_ = false;
    stopped_ = true;
  }

  float elapsed() {
    assert(stopped_);
    if (!stopped_)
      return 0;
    cudaEventSynchronize(stop_);
    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
  }

private:
  bool started_, stopped_;
  cudaEvent_t start_, stop_;
};

typedef EventTimer GPUtimer;

#endif /* CUDATIMER_CUH_OBWAUHTQ */
