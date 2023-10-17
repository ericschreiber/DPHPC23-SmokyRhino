#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "utils.h"

__device__ float warp_wise_reduction(float sum)
{
    int warpsize = 32;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        // This results in every 0th thread in a warp having the sum of all the warps
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return sum;
}

__global__ void blocked_SDDMM_kernel(
    m,
    n,
    k,
    d_A,
    d_B,
    d_C,
    d_rowPtr,
    d_colIdx,
    d_result)
{
    int warpsize = 32;
    int warp_idx = threadIdx.x / warpsize;
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // iterate over all rows assigned to a certain block
    for (int i = blockIdx.x; i < m; i += gridDim.x)
    {
        // iterate over all elements that need to be compputed in that row
        for (int j = d_rowPtr[i]; j < d_rowPtr[i + 1]; j++)
            float my_sum = 0.0;
        // every thread does some multiplications jumping by the blockDimension and adds that to its local sum
        for (int l = threadIdx.x; l < n; l += blockDim.x)
        {
            my_sum += d_A[i][l] * d_B[d_colIdx[j]][l];
        }
        // We reduce that local sum over all threads in a block
        float warp_sum = warp_wise_reduction(my_sum);
        if (warpIdx == 0)
        {
            float block_sum = warp_wise_reduction(warp_sum);
        }
        // after we got the final sum, we now do the multiplication and write the result
        if (threadIdx.x == 0)
        {
            float result = block_sum * d_C[j];
            d_result[j] = result;
        }
    }
}

void compute_blockwise(
    int m,
    int n,
    int k,
    float *d_A,
    float *d_B,
    float *d_C,
    int *d_rowPtr,
    int *d_colIdx,
    float *d_result)
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

    cublasHandle_t handle;
    CUDA_CHECK(cublasCreate(&handle));
}
