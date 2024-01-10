// Assumption: B is transposed in Memory <3

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "cassert"
#include "utils.h"

namespace cg = cooperative_groups;

#define WARP_SIZE 32

__device__ float warp_wise_reduction(float sum)
{
    // This results in every 0th thread in a warp having the sum of all the warps
    // Try changing this around and measure if you have the time
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);

    return sum;
}
__device__ float two_warp_per_line_reduction(float sum)
{
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    return sum;
}

__device__ float four_warp_per_line_reduction(float sum)
{
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    return sum;
}

__device__ float eight_warp_per_line_reduction(float sum)
{
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    return sum;
}

__device__ float sixteen_warp_per_line_reduction(float sum)
{
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    return sum;
}

__global__ void one_warp_per_line_SDDMM_kernel(
    int m,
    int n,
    int k,
    float* d_A,
    float* d_B,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_result)
{
    int thread_id = threadIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x;
    int rows_per_block = block_size / WARP_SIZE;
    int warp_idx = thread_id / WARP_SIZE;

    // iterate over all rows assigned to a certain warp
    for (int i = blockIdx.x * rows_per_block + warp_idx; i < m; i += grid_size * rows_per_block)
    {
        // if (i == 2)
        // {
        //     printf("Hello");
        // }
        // printf("Hello from blockidx %d, i = %d\n", blockIdx.x, i);
        // iterate over all elements that need to be computed in that row
        for (int j = d_rowPtr[i]; j < d_rowPtr[i + 1]; j++)
        {
            float my_sum = 0.0;

            // every thread does some multiplications jumping by the Warp_size (since we have one warp per row) and adds that to its local sum
            for (int l = thread_id % WARP_SIZE; l < k; l += WARP_SIZE)
            {
                my_sum += d_A[i * k + l] * d_B[d_colIdx[j] * k + l];
            }

            // if (my_sum > 0)
            // {
            //     printf("hello from block %d, warp %d, thread %d, my sum is %f\n", blockIdx.x, warp_idx, thread_id, my_sum);
            // }
            // __syncthreads();
            // We reduce that local sum over all threads in a warp
            float warp_sum = warp_wise_reduction(my_sum);

            // if (warp_sum > 0)
            // {
            //     printf("hello from warp %d, thread %d, my sum is %f, j = %d, i = %d\n", warp_idx, thread_id, warp_sum, j, i);
            // }

            // __syncthreads();

            // Each first thread in a warp now writes it's warp reduction result into a buffer
            if (thread_id % WARP_SIZE == 0)
            {
                d_result[j] = warp_sum;
            }
        }
    }
}

__global__ void two_warps_per_line_SDDMM_kernel(
    int warps_per_line,
    int m,
    int n,
    int k,
    float* d_A,
    float* d_B,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_result)
{
    __shared__ double warpSums_buffer[32];

    int thread_id = threadIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x;
    int threads_per_line = WARP_SIZE * warps_per_line;
    int rows_per_block = block_size / WARP_SIZE / warps_per_line;
    int warp_idx = thread_id / WARP_SIZE;
    int thread_pos_in_warp = thread_id % WARP_SIZE;

    // iterate over all rows assigned to a certain block
    for (int i = blockIdx.x * rows_per_block + (warp_idx / warps_per_line); i < m; i += grid_size * rows_per_block)
    {
        // iterate over all elements that need to be computed in that row
        for (int j = d_rowPtr[i]; j < d_rowPtr[i + 1]; j++)
        {
            float my_sum = 0.0;
            // every thread does some multiplications jumping by the warps_per_line * WARP_SIZE and adds that to its local sum
            for (int l = thread_id % (WARP_SIZE * warps_per_line); l < k; l += warps_per_line * WARP_SIZE)
            {
                my_sum += d_A[i * k + l] * d_B[d_colIdx[j] * k + l];
            }

            // We reduce that local sum over all threads in a block
            float warp_sum = warp_wise_reduction(my_sum);

            // Each first thread in a warp now writes it's warp reduction result into a buffer
            if (thread_pos_in_warp == 0)
            {
                warpSums_buffer[warp_idx] = warp_sum;
            }

            __syncthreads();

            // The first warp in the block now does another reduction on the elements in the buffer
            float block_sum = 0.0;
            if (warp_idx % warps_per_line == 0)
            {
                block_sum = two_warp_per_line_reduction(warpSums_buffer[warp_idx + thread_id % warps_per_line]);
            }

            // after we got the final sum, we now do the multiplication with the sample matrix and write the result
            if (thread_pos_in_warp == 0 && warp_idx % warps_per_line == 0)
            {
                d_result[j] = block_sum;
            }
        }
    }
}

__global__ void four_warps_per_line_SDDMM_kernel(
    int warps_per_line,
    int m,
    int n,
    int k,
    float* d_A,
    float* d_B,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_result)
{
    __shared__ double warpSums_buffer[32];

    int thread_id = threadIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x;
    int threads_per_line = WARP_SIZE * warps_per_line;
    int rows_per_block = block_size / WARP_SIZE / warps_per_line;
    int warp_idx = thread_id / WARP_SIZE;
    int thread_pos_in_warp = thread_id % WARP_SIZE;

    // iterate over all rows assigned to a certain block
    for (int i = blockIdx.x * rows_per_block + (warp_idx / warps_per_line); i < m; i += grid_size * rows_per_block)
    {
        // iterate over all elements that need to be computed in that row
        for (int j = d_rowPtr[i]; j < d_rowPtr[i + 1]; j++)
        {
            float my_sum = 0.0;
            // every thread does some multiplications jumping by the warps_per_line * WARP_SIZE and adds that to its local sum
            for (int l = thread_id % (WARP_SIZE * warps_per_line); l < k; l += warps_per_line * WARP_SIZE)
            {
                my_sum += d_A[i * k + l] * d_B[d_colIdx[j] * k + l];
            }

            // We reduce that local sum over all threads in a block
            float warp_sum = warp_wise_reduction(my_sum);

            // Each first thread in a warp now writes it's warp reduction result into a buffer
            if (thread_pos_in_warp == 0)
            {
                warpSums_buffer[warp_idx] = warp_sum;
            }

            __syncthreads();

            // The first warp in the block now does another reduction on the elements in the buffer
            float block_sum = 0.0;
            if (warp_idx % warps_per_line == 0)
            {
                block_sum = four_warp_per_line_reduction(warpSums_buffer[warp_idx + thread_id % warps_per_line]);
            }

            // after we got the final sum, we now do the multiplication with the sample matrix and write the result
            if (thread_pos_in_warp == 0 && warp_idx % warps_per_line == 0)
            {
                d_result[j] = block_sum;
            }
        }
    }
}

__global__ void eight_warps_per_line_SDDMM_kernel(
    int warps_per_line,
    int m,
    int n,
    int k,
    float* d_A,
    float* d_B,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_result)
{
    __shared__ double warpSums_buffer[32];

    int thread_id = threadIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x;
    int threads_per_line = WARP_SIZE * warps_per_line;
    int rows_per_block = block_size / WARP_SIZE / warps_per_line;
    int warp_idx = thread_id / WARP_SIZE;
    int thread_pos_in_warp = thread_id % WARP_SIZE;

    // iterate over all rows assigned to a certain block
    for (int i = blockIdx.x * rows_per_block + (warp_idx / warps_per_line); i < m; i += grid_size * rows_per_block)
    {
        // iterate over all elements that need to be computed in that row
        for (int j = d_rowPtr[i]; j < d_rowPtr[i + 1]; j++)
        {
            float my_sum = 0.0;
            // every thread does some multiplications jumping by the warps_per_line * WARP_SIZE and adds that to its local sum
            for (int l = thread_id % (WARP_SIZE * warps_per_line); l < k; l += warps_per_line * WARP_SIZE)
            {
                my_sum += d_A[i * k + l] * d_B[d_colIdx[j] * k + l];
            }

            // We reduce that local sum over all threads in a block
            float warp_sum = warp_wise_reduction(my_sum);

            // Each first thread in a warp now writes it's warp reduction result into a buffer
            if (thread_pos_in_warp == 0)
            {
                warpSums_buffer[warp_idx] = warp_sum;
            }

            __syncthreads();

            // The first warp in the block now does another reduction on the elements in the buffer
            float block_sum = 0.0;
            if (warp_idx % warps_per_line == 0)
            {
                block_sum = eight_warp_per_line_reduction(warpSums_buffer[warp_idx + thread_id % warps_per_line]);
            }

            // after we got the final sum, we now do the multiplication with the sample matrix and write the result
            if (thread_pos_in_warp == 0 && warp_idx % warps_per_line == 0)
            {
                d_result[j] = block_sum;
            }
        }
    }
}

__global__ void sixteen_warps_per_line_SDDMM_kernel(
    int warps_per_line,
    int m,
    int n,
    int k,
    float* d_A,
    float* d_B,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_result)
{
    __shared__ double warpSums_buffer[32];

    int thread_id = threadIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x;
    int threads_per_line = WARP_SIZE * warps_per_line;
    int rows_per_block = block_size / WARP_SIZE / warps_per_line;
    int warp_idx = thread_id / WARP_SIZE;
    int thread_pos_in_warp = thread_id % WARP_SIZE;

    // iterate over all rows assigned to a certain block
    for (int i = blockIdx.x * rows_per_block + (warp_idx / warps_per_line); i < m; i += grid_size * rows_per_block)
    {
        // iterate over all elements that need to be computed in that row
        for (int j = d_rowPtr[i]; j < d_rowPtr[i + 1]; j++)
        {
            float my_sum = 0.0;
            // every thread does some multiplications jumping by the warps_per_line * WARP_SIZE and adds that to its local sum
            for (int l = thread_id % (WARP_SIZE * warps_per_line); l < k; l += warps_per_line * WARP_SIZE)
            {
                my_sum += d_A[i * k + l] * d_B[d_colIdx[j] * k + l];
            }

            // We reduce that local sum over all threads in a block
            float warp_sum = warp_wise_reduction(my_sum);

            // Each first thread in a warp now writes it's warp reduction result into a buffer
            if (thread_pos_in_warp == 0)
            {
                warpSums_buffer[warp_idx] = warp_sum;
            }

            __syncthreads();

            // The first warp in the block now does another reduction on the elements in the buffer
            float block_sum = 0.0;
            if (warp_idx % warps_per_line == 0)
            {
                block_sum = sixteen_warp_per_line_reduction(warpSums_buffer[warp_idx + thread_id % warps_per_line]);
            }

            // after we got the final sum, we now do the multiplication with the sample matrix and write the result
            if (thread_pos_in_warp == 0 && warp_idx % warps_per_line == 0)
            {
                d_result[j] = block_sum;
            }
        }
    }
}

__global__ void thirtyTwo_warps_per_line_SDDMM_kernel(
    int warps_per_line,
    int m,
    int n,
    int k,
    float* d_A,
    float* d_B,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_result)
{
    __shared__ double warpSums_buffer[32];

    int thread_id = threadIdx.x;
    int block_size = blockDim.x;
    int grid_size = gridDim.x;
    int rows_per_block = block_size / WARP_SIZE;
    int warp_idx = thread_id / WARP_SIZE;

    // iterate over all rows assigned to a certain block
    for (int i = blockIdx.x * rows_per_block; i < m; i += grid_size)
    {
        // iterate over all elements that need to be computed in that row
        for (int j = d_rowPtr[i]; j < d_rowPtr[i + 1]; j++)
        {
            float my_sum = 0.0;
            // every thread does some multiplications jumping by the warps_per_line * WARP_SIZE and adds that to its local sum
            for (int l = thread_id % (WARP_SIZE * warps_per_line); l < k; l += warps_per_line * WARP_SIZE)
            {
                my_sum += d_A[i * k + l] * d_B[d_colIdx[j] * k + l];
            }

            // We reduce that local sum over all threads in a block
            float warp_sum = warp_wise_reduction(my_sum);

            // Each first thread in a warp now writes it's warp reduction result into a buffer
            if (thread_id % WARP_SIZE == 0)
            {
                warpSums_buffer[warp_idx] = warp_sum;
            }
            __syncthreads();

            // The first warp in the block now does another reduction on the elements in the buffer
            float block_sum = 0.0;
            if ((warp_idx % warps_per_line) == 0)
            {
                block_sum = warp_wise_reduction(warpSums_buffer[thread_id]);
            }

            // after we got the final sum, we now do the multiplication with the sample matrix and write the result
            if (thread_id % warps_per_line == 0)
            {
                d_result[j] = block_sum;
            }
        }
    }
}

void compute_blockwise(
    int lines_per_block,
    int warps_per_line,
    int m,
    int n,
    int k,
    float* d_A,
    float* d_B,
    int* d_rowPtr,
    int* d_colIdx,
    float* d_result)
{
    // std::cout << lines_per_block << ", " << warps_per_line << ", lines per block: " << increase_block_size_factor << std::endl;
    // int warp_group_size = WARP_SIZE * warps_per_line;

    if (lines_per_block > 32)
    {
        // int num_blocks = min(160, m);
        // std::cout << "Hit in 1" << std::endl;
        one_warp_per_line_SDDMM_kernel<<<160, 1024>>>(
            m,
            n,
            k,
            d_A,
            d_B,
            d_rowPtr,
            d_colIdx,
            d_result);
    }
    else
    {
        if (warps_per_line == 2)
        {
            // std::cout << "Hit in 2" << std::endl;
            two_warps_per_line_SDDMM_kernel<<<160 * 16, 1024 / 16>>>(
                2,
                m,
                n,
                k,
                d_A,
                d_B,
                d_rowPtr,
                d_colIdx,
                d_result);
        }
        else if (warps_per_line > 2 && warps_per_line < 5)
        {
            // std::cout << "Hit in 4" << std::endl;
            four_warps_per_line_SDDMM_kernel<<<160 * 8, 1024 / 8>>>(
                4,
                m,
                n,
                k,
                d_A,
                d_B,
                d_rowPtr,
                d_colIdx,
                d_result);
        }
        else if (warps_per_line > 4 && warps_per_line < 9)
        {
            // std::cout << "Hit in 8" << std::endl;
            eight_warps_per_line_SDDMM_kernel<<<160 * 4, 1024 / 4>>>(
                8,
                m,
                n,
                k,
                d_A,
                d_B,
                d_rowPtr,
                d_colIdx,
                d_result);
        }
        else if (warps_per_line > 8 && warps_per_line < 17)
        {
            // std::cout << "Hit in 16" << std::endl;
            sixteen_warps_per_line_SDDMM_kernel<<<160 * 2, 1024 / 2>>>(
                16,
                m,
                n,
                k,
                d_A,
                d_B,
                d_rowPtr,
                d_colIdx,
                d_result);
        }
        else
        {
            // std::cout << "Hit in 32" << std::endl;
            thirtyTwo_warps_per_line_SDDMM_kernel<<<160, 1024>>>(
                32,
                m,
                n,
                k,
                d_A,
                d_B,
                d_rowPtr,
                d_colIdx,
                d_result);
        }
    }
    // CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
