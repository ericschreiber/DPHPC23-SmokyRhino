#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "coo_tiling_naive_gpu/coo_tiling_naive_gpu_SDDMM.cuh"
#include "utils.h"

// *********** IDEA ***********
//
// The idea is to go rowvise and use one block for each tile of a row vector. This row gets loaded into shared memory.
// In shared mem we keep the row and the calculated result.
// For row length tile n and m columns. We need to load n floats into shared mem. Additionally we need to save 32 floats for the
// intermediate result. So we need 32 + n floats of shared mem per block.
//
// We then use one warp for each column of the matrix B. We don't load the column into shared mem as we dont reuse it.
// Then we run warp reduce to calculate the dot product of the row and the column.
// We then write the result into the result matrix.
//
// ******* ASSUMPTIONS ********
//
// 1. For ease of use we assume that the coo matrix is stored in row major format. This means that the row indices are sorted
// and all columns of a row are stored next to each other. This assumption is used to load the row into shared mem.
// 2. We assume that matrixC_GPU_row_ptr is already in memory and computed. For the future we will have a CSR as input.
//
// ********* Problems **********
//
// 1. Since we don't tile over the columns if there are a lot more cols then threads or a lot less the efficiency will be bad.
// 2. matrixC_GPU_row_indices into shared mem
// 3. matrixA_GPU_values into shared mem
// 4. We may prefer CSR for this implementation
//
// *********** ***** ***********

// // For Cacheline optimisation we define some hyperparameters as constants
// // Number of sectors in a cacheline based on this "https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf"
// constexpr int NUM_SECTORS = 4;
// // Number of total threads in a warp
// constexpr int WARP_SIZE = 32;
// // Number of warps in a block
// constexpr int WARPS_PER_BLOCK = 32;
// // Number of threads in a block
// constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;

// // Max Blocks per SM (2* 1024 threads per SM)
// constexpr int MAX_BLOCKS_PER_SM = 2;
// // Number of SM
// constexpr int NUM_SM = 80;

// // Total amount of L1 cache 128 KB per SM
// constexpr int L1_CACHE_SIZE_BYTE_perSM = 128 * 1024;
// Max size of shared mem per SM
constexpr int MAX_SHARED_MEM_PER_SM = 96 * 1024;

// Biggest row tile size for a block (MAX_SHARED_MEM_PER_SM - Size of intermediate result) / Size of float
// constexpr int MAX_ROW_TILE_SIZE = (MAX_SHARED_MEM_PER_SM - 32 * 4) / 4;
constexpr int MAX_ROW_TILE_SIZE = (MAX_SHARED_MEM_PER_SM - 0) / 4;

__device__ float dot_product_for_tiled(
    const int k,
    const float* __restrict__ const matrixA_GPU_values_row,
    const float* __restrict__ const matrixB_transposed_GPU_values_col)
{
    // calculate SUM_i row[i] * col[i]
    float result = 0;
    for (int i = 0; i < k; i++)
    {
        result += matrixA_GPU_values_row[i] * matrixB_transposed_GPU_values_col[i];
    }
    return result;
}

__device__ float naive_coo_one_val_for_tiled(
    const int k,
    const float multiplier,
    const float* __restrict__ const matrixA_GPU_values_row,
    const float* __restrict__ const matrixB_transposed_GPU_values_col)
{
    // calculate mutiplier * SUM_i row[i] * col[i]
    return multiplier * dot_product_for_tiled(k, matrixA_GPU_values_row, matrixB_transposed_GPU_values_col);
}

// We encode the tiles as follows:
// Tile index = Row Number + n * Num Rows (As each tile is a block tile index = BlockIdx.x)
// So we reduce contention in the atomic add as all first tiles will be computed. Then all second tiles and so on.
// We can get the row as tile index % n
// We can get the column as tile index % n (As it is transposed)
// We can get the tile index as tile index / n

// ******* ASSUMPTIONS ********
//
// For ease of use we assume that the coo matrix is stored in row major format. This means that the row indices are sorted
// and all columns of a row are stored next to each other. This assumption is used to load the row into shared mem.
//
// *********** ***** ***********
__global__ void naive_coo_tiled_no_shared_mem(
    const int num_rows,
    const int num_cols,
    const int k,
    const int tile_size,
    const int last_tile_size,   // If this thread should compute a tile which is at the end, it may be smaller than the other tiles.
    const int last_tile_index,  // The index of the last tile of row 0
    const int numElementsCrowPtr,
    const int numElementsC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_ptr,  // CSR equivalent We save the index of the first column for each row
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values)
{
    int row_index = blockIdx.x % num_rows;
    int tile_index = blockIdx.x / num_rows;

    if (tile_index < last_tile_index)
    {
        int coo_start_index = matrixC_GPU_row_ptr[row_index];
        int coo_end_index = matrixC_GPU_row_ptr[row_index + 1];
        int row = matrixC_GPU_row_indices[coo_start_index];

        // Loop over all threads to check that we compute all columns.
        // Thread 0 calculates column 0, n, 2n, 3n, ...
        for (int col_runner = coo_start_index + threadIdx.x; col_runner < coo_end_index; col_runner += blockDim.x)
        {
            int col = matrixC_GPU_col_indices[col_runner];

            float multiplier = matrixC_GPU_values[col_runner];

            float result = naive_coo_one_val_for_tiled(tile_size, multiplier, matrixA_GPU_values + (row * k) + (tile_index * tile_size), matrixB_transposed_GPU_values + (col * k) + (tile_index * tile_size));
            // As we may have multiple threads from different blocks writing to the same location we need to use atomic add.
            atomicAdd(matrixResult_GPU_values + col_runner, result);
        }
    }
    else
    {
        int coo_start_index = matrixC_GPU_row_ptr[row_index];
        int coo_end_index = matrixC_GPU_row_ptr[row_index + 1];
        int row = matrixC_GPU_row_indices[coo_start_index];

        // Loop over all threads to check that we compute all columns.
        // Thread 0 calculates column 0, n, 2n, 3n, ...
        for (int col_runner = coo_start_index + threadIdx.x; col_runner < coo_end_index; col_runner += blockDim.x)
        {
            int col = matrixC_GPU_col_indices[col_runner];

            float multiplier = matrixC_GPU_values[col_runner];

            float result = naive_coo_one_val_for_tiled(last_tile_size, multiplier, matrixA_GPU_values + (row * k) + (tile_index * tile_size), matrixB_transposed_GPU_values + (col * k) + (tile_index * tile_size));
            // As we may have multiple threads from different blocks writing to the same location we need to use atomic add.
            atomicAdd(matrixResult_GPU_values + col_runner, result);
        }
    }
}

// ******* DOES NOT WORK FOR MULTIPLE BLOCKS ********
//
// For multiple blocks reuse the idea and put the result of each block into a row.
// Then spawn new blocks to add the rows together.
// Then spawn again new blocks to add the addition of all previous rows to all rows in the range of the block.
//
// **************************************************
// __global__ void computeRowPointerKernel(const int* __restrict__ const rowIndex, int* const rowPointer, const int numRows, const int cooRowIndex_size)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     // Count the columns for each row
//     if (tid < cooRowIndex_size)
//     {
//         atomicAdd(&rowPointer[rowIndex[tid]], 1);
//     }

//     // Make sure all threads in the block finish their work
//     __syncthreads();

//     // Calculate the sum over all previous indices
//     // We use an inverse hillis steele scan to calculate the prefix sum
//     // https://www.mimuw.edu.pl/~ps209291/kgkp/slides/scan.pdf

//     // Calculate the sum of the first half of the array
//     for (int stride = 1; stride < numRows + 1; stride *= 2)
//     {
//         int index = (2 * stride * tid) - 1;
//         if (index >= 0 && index < (numRows + 1))
//         {
//             rowPointer[index] += rowPointer[index - stride];
//         }
//         __syncthreads();
//     }

//     // Calculate the sum of the second half of the array
//     for (int stride = (numRows + 1) / 2; stride > 0; stride /= 2)
//     {
//         int index = (2 * stride * tid) - 1;
//         if (index + stride < (numRows + 1))
//         {
//             rowPointer[index + stride] += rowPointer[index];
//         }
//         __syncthreads();
//     }

//     // Make sure all threads in the block finish their work
//     __syncthreads();

//     // Shift the array to the right by one
//     for (int i = numRows; i > 0; i--)
//     {
//         rowPointer[i] = rowPointer[i - 1];
//     }
//     rowPointer[0] = 0;

//     // Make sure all threads in the block finish their work
//     __syncthreads();
// }

// Matrix sizes:
// A: m x k
// B: k x n
// Sampling: m x n
void compute_coo_tiling_naive_gpu(
    const int m,
    const int n,
    const int k,
    const int numElementsC,
    const int numElementsCrowPtr,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_row_ptr,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values)
{
    // **** Compute row pointer ****
    // **** DOES NOT WORK FOR MULTIPLE BLOCKS **** SEE ABOVE ****
    // // Launch CUDA kernel
    // int threadsPerBlock = 256;
    // int gridSize = (numElementsC + threadsPerBlock - 1) / threadsPerBlock;
    // computeRowPointerKernel<<<gridSize, threadsPerBlock>>>(matrixC_GPU_row_indices, matrixC_GPU_row_ptr, m, numElementsC);

    // **** Compute SDDMM  ****

    int threadsPerBlock = 1024;

    const int tile_size = std::min(MAX_ROW_TILE_SIZE, k);
    const int last_tile_size = k % MAX_ROW_TILE_SIZE;
    const int num_tiles = (k + tile_size - 1) / tile_size;
    const int last_tile_index = num_tiles - 1;

    // -1 because we don't need a block for the last indexed row. Thats just a pointer to the last element in the matrix C.
    int blocks = (numElementsCrowPtr - 1) * num_tiles;

    // std::cout << "k: " << k << std::endl;
    // std::cout << "MAX_ROW_TILE_SIZE: " << MAX_ROW_TILE_SIZE << std::endl;
    // std::cout << "tile_size: " << tile_size << std::endl;
    // std::cout << "last_tile_size: " << last_tile_size << std::endl;
    // std::cout << "num_tiles: " << num_tiles << std::endl;
    // std::cout << "last_tile_index: " << last_tile_index << std::endl;
    // std::cout << "numElementsCrowPtr: " << numElementsCrowPtr << std::endl;
    // std::cout << "blocks: " << blocks << std::endl;
    // std::cout << "starting kernel" << std::endl;

    naive_coo_tiled_no_shared_mem<<<blocks, threadsPerBlock>>>(
        m,
        n,
        k,
        tile_size,
        last_tile_size,
        last_tile_index,
        numElementsCrowPtr,
        numElementsC,
        matrixA_GPU_values,
        matrixB_transposed_GPU_values,
        matrixC_GPU_values,
        matrixC_GPU_row_ptr,
        matrixC_GPU_row_indices,
        matrixC_GPU_col_indices,
        matrixResult_GPU_values);
    // Aggregate the return value of the kernel
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}