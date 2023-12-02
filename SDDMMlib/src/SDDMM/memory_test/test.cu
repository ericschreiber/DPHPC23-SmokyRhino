#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

#include "memory_test/test.cuh"
#include "utils.h"

////
// this kernels computes the Hadamard Product of two dense matrices A and B
// A (m x n)
// B (m x n)
//
// the kernel expects the number of elements of the matrices as int (m*n) and
// pointers to the 2 matrices of type float in the following order A, B
// all pointers need to point to memory on the GPU
//
// the result is written to B
////

__global__ void memory_test(
    const float* __restrict__ const A,
    float* __restrict__ const B)
{
    int idx = threadIdx.x;

    for (int i = (idx << 2); i < 12288; i += (blockDim.x << 2))
    {
        B[i] = A[i] + 1;
        B[i + 1] = A[i + 1] + 1;
        B[i + 2] = A[i + 2] + 1;
        B[i + 3] = A[i + 3] + 1;
    }
}

void testing(
    const float* const A,
    float* const B)
{
    memory_test<<<1, 1024>>>(
        A,
        B);

    // CUDA_CHECK(cudaGetLastError());

    // synchronization not needed if implicit in the time measurement
    // CUDA_CHECK(cudaDeviceSynchronize());
}
