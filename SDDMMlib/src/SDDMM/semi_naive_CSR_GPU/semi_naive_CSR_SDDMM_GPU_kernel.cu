// Assumption: B is transposed in Memory <3

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "utils.h"

#define WARP_SIZE 32

__device__ float warp_wise_reduction(float sum)
{
    // This results in every 0th thread in a warp having the sum of all the warps
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);

    return sum;
}

__global__ void blocked_SDDMM_kernel(
    int m,
    int n,
    int k,
    float* d_A,
    float* d_B,
    float* d_C,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_result)
{
    __shared__ double warpSums_buffer[32];

    int thread_id = threadIdx.x;
    int warp_idx = thread_id / WARP_SIZE;

    // iterate over all rows assigned to a certain block
    for (int i = blockIdx.x; i < m; i += gridDim.x)
    {
        // iterate over all elements that need to be compputed in that row
        for (int j = d_rowPtr[i]; j < d_rowPtr[i + 1]; j++)
        {
            float my_sum = 0.0;
            // every thread does some multiplications jumping by the blockDimension and adds that to its local sum
            for (int l = thread_id; l < k; l += blockDim.x)
            {
                my_sum += d_A[i * k + l] * d_B[d_colIdx[j] * k + l];
            }
            // We reduce that local sum over all threads in a block
            float warp_sum = warp_wise_reduction(my_sum);

            // Each first thread in a warp now writes it's warp reduction result into a buffer
            if (thread_id % WARP_SIZE)
            {
                warpSums_buffer[warp_idx] = warp_sum;
            }

            // The first warp in the block now does another reduction on the elements in the buffer
            float block_sum = 0.0;
            if (warp_idx == 0)
            {
                float block_sum = warp_wise_reduction(warpSums_buffer[thread_id]);
            }

            // after we got the final sum, we now do the multiplication with the sample matrix and write the result
            if (thread_id == 0)
            {
                float result = block_sum * d_C[j];
                d_result[j] = result;
            }
        }
    }
}

void compute_blockwise(
    int m,
    int n,
    int k,
    float* d_A,
    float* d_B,
    float* d_C,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_result)
{
    int max_blocks = 2024;
    int num_blocks = min(max_blocks, m);

    blocked_SDDMM_kernel<<<num_blocks, 1024>>>(
        m,
        n,
        k,
        d_A,
        d_B,
        d_C,
        d_rowPtr,
        d_colIdx,
        d_result);
}