#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "naive_coo_gpu/naive_coo_SDMM.cuh"
#include "utils.h"

#define THREADS_PER_BLOCK 2
#define SHARED_MEM_SIZE_BYTES 8  // this is the size of shared mem on both the A100 and V100 GPUs.
                                 // can force tiling (e.g. for testing) by setting this to something small.
#define SHARED_MEM_SIZE SHARED_MEM_SIZE_BYTES / sizeof(float)

__global__ void SM_SM_coo(
    const int T_ij,
    const int blocks,
    const int last_T_i,
    const int tiling_steps,
    const int last_T_j,
    const int k,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values,
    const int* nnz_indexing_array)
{
    // Define necessary parameters
    int block_index = blockIdx.x;
    // Set T_i for the block
    int T_i = T_ij;
    // This presumes that blocks are indexed at 0
    if (block_index == blocks - 1)
    {
        T_i = last_T_i;
    }

    // Declare pointer to start of A for the block
    const float* A_vals_row_start = matrixA_GPU_values + (block_index * T_ij * k);

    // Save A in Shared Memory for full reuse within a block
    extern __shared__ float A_in_row_panel[];
    if (threadIdx.x == 0)
    {
        // Iterate over the rows in block
        for (int i = 0; i < T_i * k; i++)
        {
            A_in_row_panel[i] = *(A_vals_row_start + i);
        }
    }

    // Main loop over the tiles in a row panel
    for (int tile = 0; tile < tiling_steps; tile++)
    {
        // For each tile declare pointer to beginning of B transposed
        const float* B_vals_row_start = matrixB_transposed_GPU_values + (tiling_steps * T_ij * k);
        // Set T_j for tile
        int T_j = T_ij;
        if (tile == tiling_steps - 1)
        {
            T_j = last_T_j;
        }

        // Save B in Shared Memory for reuse within a tile
        extern __shared__ float B_in_tile[];
        if (threadIdx.x == 0)
        {
            // Iterate over B transpose rows in a tile
            for (int i = 0; i < T_j * k; i++)
            {
                B_in_tile[i] = *(B_vals_row_start + i);
            }
        }
        __syncthreads();

        // Smart way of dividing work between all the threads:
        int c_index;
        for (int i = 0; i < T_i; i++)
        {
            c_index = nnz_indexing_array[(block_index * tiling_steps + tile) * T_ij + i];
            printf("block: %d, tile[%d], partial row: %d, nnz's: %d\n", block_index, tile, tile, c_index);
        }
    }
}

__global__ void precomputation(
    int* nnz_indexing_array,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    int tiling_steps,
    int numBlocks,
    int T_ij,
    const int numElementsC,
    const int m)
{
    // In this function is the sparse matrix indexed for tiling
    // Loop over the blocks
    int row;
    int col;
    int block;
    int tile;
    int row_in_block;

    // This loop counts the nbr of nnz in a given row in a tile
    for (int i = 0; i < numElementsC; i++)
    {
        row = matrixC_GPU_col_indices[i];
        col = matrixC_GPU_col_indices[i];
        block = row / numBlocks;
        tile = col / tiling_steps;
        row_in_block = row - block * T_ij;
        nnz_indexing_array[(block * tiling_steps + tile) * T_ij + row_in_block] += 1;
    }

    // m*tiling_steps is the len of nnz_indexing_array, this loop makes the count cummulative
    for (int i = 1; i < m * tiling_steps; i++)
    {
        nnz_indexing_array[i] += nnz_indexing_array[i - 1];
    }
}

void compute(
    const int m,
    const int n,
    const int k,  // number of rows of B
    const int numElementsC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values)
{
    // MY CODE
    // Tiling with T_i = T_j = T_ij based on size of shared memory
    int T_ij = (float)(SHARED_MEM_SIZE_BYTES) / (2 * k);
    // Each block calculates one row panel (and each tile in sequence)
    int blocks = ceil(m / T_ij);
    // The last block may get fewer rows if m isnt dividable by T_ij
    int last_T_i = m % T_ij;
    // Each block computes the tiles in a row-panel -> row_len/tile_width
    int tiling_steps = ceil(n / T_ij);
    // Last tile may get fewer columns from B if n isnt divicable by T_ij
    int last_T_j = n % T_ij;
    // This array gets filled up nnz index per row in each tile and block
    int* nnz_indexing_array;
    CUDA_CHECK(cudaMalloc((void**)&nnz_indexing_array, (m * tiling_steps) * sizeof(int)));

    precomputation<<<1, 1>>>(nnz_indexing_array, matrixC_GPU_row_indices, matrixC_GPU_col_indices, tiling_steps, blocks, T_ij, numElementsC, m);

    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    // Call main kernel
    SM_SM_coo<<<blocks, threadsPerBlock, SHARED_MEM_SIZE>>>(
        T_ij,
        blocks,
        last_T_i,
        tiling_steps,
        last_T_j,
        k,
        matrixA_GPU_values,
        matrixB_transposed_GPU_values,
        matrixC_GPU_values,
        matrixC_GPU_row_indices,
        matrixC_GPU_col_indices,
        matrixResult_GPU_values,
        nnz_indexing_array);
    // Aggregate the return value of the kernel
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // free the array prevBlocksWork on GPU
    CUDA_CHECK(cudaFree(nnz_indexing_array));
}