#include <cuda_runtime.h>
#include "utils.h"
#include <algorithm>


__global__ void naivesampling(
    const int size,
    const double* const A,
    double* const C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        C[idx] = C[idx] * A[idx];
    }
}


void mysampling(
    const int size,
    const double* const A,
    double* const C)
{

  int blocks = std::min(1024, (size + 1023) / 1024);

  naivesampling<<<blocks, 1024>>>(size, A, C);
  CUDA_CHECK(cudaGetLastError());
}