#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "coo_tiling_cached_gpu/coo_tiling_cached_gpu_SDDMM.cuh"
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
// *********** ***** ***********

// For Cacheline optimisation we define some hyperparameters as constants
// Number of sectors in a cacheline based on this "https://on-demand.gputechconf.com/gtc/2018/presentation/s81006-volta-architecture-and-performance-optimization.pdf"
constexpr int NUM_SECTORS = 4;
// Number of total threads in a warp
constexpr int WARP_SIZE = 32;
// Number of warps in a block
constexpr int WARPS_PER_BLOCK = 32;
// Number of threads in a block
constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE;

// Max Blocks per SM (2* 1024 threads per SM)
constexpr int MAX_BLOCKS_PER_SM = 2;
// Number of SM
constexpr int NUM_SM = 80;

// Total amount of L1 cache 128 KB per SM
constexpr int L1_CACHE_SIZE_BYTE_perSM = 128 * 1024;
// Max size of shared mem per SM
constexpr int MAX_SHARED_MEM_PER_SM = 96 * 1024;

// Biggest row tile size for a block (MAX_SHARED_MEM_PER_SM - Size of intermediate result) / Size of float
constexpr int MAX_ROW_TILE_SIZE = (MAX_SHARED_MEM_PER_SM - 32 * 4) / 4;

void compute_coo_tiling_cached_gpu(
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
}