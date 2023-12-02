#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

#include "memory_test/test_shared.cuh"
#include "utils.h"

__global__ void memory_test_shared(
    const float* __restrict__ const A,
    float* __restrict__ const B)
{
    int idx = threadIdx.x;

    __shared__ float sA[12288];
    for (int i = (idx << 2); i < 12288; i += (blockDim.x << 2))
    {
        sA[i] = A[i];
        sA[i + 1] = A[i + 1];
        sA[i + 2] = A[i + 2];
        sA[i + 3] = A[i + 3];
    }

    for (int j = 0; j < 5; ++j)
    {
        __syncthreads();
        for (int i = (idx << 2); i < 12288; i += (blockDim.x << 2))
        {
            B[i] = sA[i] + 1;
            B[i + 1] = sA[i + 1] + 1;
            B[i + 2] = sA[i + 2] + 1;
            B[i + 3] = sA[i + 3] + 1;
        }
    }
}

void testing_shared(
    const float* const A,
    float* const B)
{
    memory_test_shared<<<1, 1024>>>(
        A,
        B);
    // CUDA_CHECK(cudaGetLastError());

    // synchronization not needed if implicit in the time measurement
    // CUDA_CHECK(cudaDeviceSynchronize());
}
