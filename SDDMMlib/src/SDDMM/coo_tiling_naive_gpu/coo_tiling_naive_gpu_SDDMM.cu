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
// For ease of use we assume that the coo matrix is stored in row major format. This means that the row indices are sorted
// and all columns of a row are stored next to each other. This assumption is used to load the row into shared mem.
//
// ********* Problems **********
//
// 1. Since we don't tile over the columns if there are a lot more cols then threads or a lot less the efficiency will be bad.
// 2. matrixC_GPU_row_indices into shared mem
// 3. matrixA_GPU_values into shared mem
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
    const int tile_size,
    const int last_tile_size,   // If this thread should compute a tile which is at the end, it may be smaller than the other tiles.
    const int last_tile_index,  // The index of the last tile of row 0
    const int numElementsC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values)
{
    int row_index = blockIdx.x % num_rows;
    int col_start = threadIdx.x;
    int tile_index = blockIdx.x / num_rows;

    int own_tile_size = tile_size;
    if (tile_index >= last_tile_index)
    {
        own_tile_size = last_tile_size;
    }

    if (row_index < numElementsC)
    {
        int row_zero = matrixC_GPU_row_indices[row_index];

        // Loop over all threads to check that we compute all columns.
        // Thread 0 calculates column 0, n, 2n, 3n, ...
        for (int col_runner = col_start; col_runner < num_cols; col_runner += blockDim.x)
        {
            int index = row_index + col_runner;
            if (index >= numElementsC)
            {
                break;
            }
            int col = matrixC_GPU_col_indices[index];
            // Checking this should not occur a lot of overhead since it should be in shared (in the future)
            // TODO: matrixC_GPU_row_indices into shared mem
            if (row_zero != matrixC_GPU_row_indices[index])
            {
                printf("WE BREAKED!!! row_zero: %d, matrixC_GPU_row_indices[index]: %d\n", row_zero, matrixC_GPU_row_indices[index]);
                // We are in the next row so we don't calculate this in this thread and not even in this block.
                break;
            }
            float multiplier = matrixC_GPU_values[index];
            printf("row_index: %d, col_start: %d, col_runnder: %d, tile_index: %d, index: %d, multilier: %f\n", row_index, col_start, col_runner, tile_index, index, multiplier);
            float result = naive_coo_one_val_for_tiled(own_tile_size, multiplier, matrixA_GPU_values + row_zero + tile_index * tile_size, matrixB_transposed_GPU_values + col + tile_index * tile_size);

            // As we may have multiple threads from different blocks writing to the same location we need to use atomic add.
            atomicAdd(matrixResult_GPU_values + index, result);
        }
    }
}

// Matrix sizes:
// A: m x k
// B: k x n
// Sampling: m x n
void compute_coo_tiling_naive_gpu(
    const int m,
    const int n,
    const int k,
    const int numElementsC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values)
{
    std::cout << "Starting naive_coo_tiled_no_shared_mem" << std::endl;
    dim3 threadsPerBlock(1024);

    const int tile_size = std::min(MAX_ROW_TILE_SIZE, k);
    const int last_tile_size = k % MAX_ROW_TILE_SIZE;
    const int num_tiles = (k + tile_size - 1) / tile_size;
    const int last_tile_index = num_tiles - 1;

    int blocks = m * num_tiles;

    std::cout << "k: " << k << std::endl;
    std::cout << "MAX_ROW_TILE_SIZE: " << MAX_ROW_TILE_SIZE << std::endl;
    std::cout << "tile_size: " << tile_size << std::endl;
    std::cout << "last_tile_size: " << last_tile_size << std::endl;
    std::cout << "num_tiles: " << num_tiles << std::endl;
    std::cout << "last_tile_index: " << last_tile_index << std::endl;
    std::cout << "starting kernel" << std::endl;

    naive_coo_tiled_no_shared_mem<<<blocks, threadsPerBlock>>>(
        m,
        n,
        tile_size,
        last_tile_size,
        last_tile_index,
        numElementsC,
        matrixA_GPU_values,
        matrixB_transposed_GPU_values,
        matrixC_GPU_values,
        matrixC_GPU_row_indices,
        matrixC_GPU_col_indices,
        matrixResult_GPU_values);
    // Aggregate the return value of the kernel
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}