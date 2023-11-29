#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "coo_SM_SM_SDDMM_gpu/coo_SM_SM_SDDMM.cuh"
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
    if (block_index == blocks - 1 && last_T_i != 0)
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
        // Collecting nnz indices within this block:
        // This is the start index for the nnz_index array
        int nnz_index_in_block_start = (block_index * tiling_steps) * T_ij + tile * T_i;
        // End index
        int nnz_index_in_block_end = nnz_index_in_block_start + T_i;
        // This is the indices in the C_matrix with nnz:s
        int c_index;
        // Index for the nnz vector
        int nnz_index;
        // Nbr of nnz in previous partial row
        int prev;
        for (int i = 0; i < T_i; i++)
        {
            nnz_index = (block_index * tiling_steps) * T_ij + tile * T_i + i;
            if (nnz_index == 0)
            {
                prev = 0;
            }
            else
            {
                prev = nnz_indexing_array[nnz_index - 1];
            }
            c_index = nnz_indexing_array[nnz_index];

            if (threadIdx.x == 0)
            {
                printf("in block[%d] tile[%d], partial_row[%d] the c index are [%d,%d], nnz_index = %d, T_i: %d \n", block_index, tile, i, prev, c_index, nnz_index, T_i);
            }
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
    const int m,
    const int last_T_i)
{
    // In this function is the sparse matrix indexed for tiling
    // Loop over the blocks
    int row;
    int col;
    int block;
    int tile;
    int row_in_block;
    int index;
    int T_i;

    // This loop counts the nbr of nnz in a given row in a tile
    for (int i = 0; i < numElementsC; i++)
    {
        row = matrixC_GPU_row_indices[i];
        col = matrixC_GPU_col_indices[i];
        block = row / T_ij;
        // Fix edge case of matrix tiling
        if (block == numBlocks - 1)
        {
            T_i = last_T_i;
        }
        else
        {
            T_i = T_ij;
        }
        tile = col / T_ij;
        row_in_block = row - block * T_ij;
        // This is index for the partial row within a block and tile. This is ordered by block -> tile -> row in tile. So within a block the partial rows have adjecent index
        index = (block * tiling_steps) * T_ij + tile * T_i + row_in_block;
        nnz_indexing_array[index] += 1;
        if (index == 152)
        {
            printf(" i = 152 block: %d, tile: %d, row_in_block: %d, row %d, col %d \n", block, tile, row_in_block, row, col);
        }
    }

    // m*tiling_steps is the len of nnz_indexing_array, this loop makes the count cummulative, smaller end tile shouldnt be affected by this
    for (int i = 1; i < m * tiling_steps; i++)
    {
        nnz_indexing_array[i] += nnz_indexing_array[i - 1];
        printf("nnz_index_arr: %d i: %d \n", nnz_indexing_array[i], i);
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
    // int T_ij = (float)(SHARED_MEM_SIZE_BYTES) / (2 * k); for dev testing:
    int T_ij = 5;
    // Each block calculates one row panel (and each tile in sequence)
    // int blocks = ceil(m / T_ij); for dev testing:
    int blocks = 6;
    // The last block may get fewer rows if m isnt dividable by T_ij
    int last_T_i = m % T_ij;
    // Each block computes the tiles in a row-panel -> row_len/tile_width
    int tiling_steps = ceil(float(n) / float(T_ij));
    // Last tile may get fewer columns from B if n isnt divicable by T_ij
    int last_T_j = n % T_ij;
    // This array gets filled up nnz index per row in each tile and block
    int* nnz_indexing_array;
    CUDA_CHECK(cudaMalloc((void**)&nnz_indexing_array, (m * tiling_steps) * sizeof(int)));

    precomputation<<<1, 1>>>(nnz_indexing_array, matrixC_GPU_row_indices, matrixC_GPU_col_indices, tiling_steps, blocks, T_ij, numElementsC, m, last_T_i);

    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    printf("Tiling information: Blocks: %d, Tiles:%d, T_ij:%d, last tile width:%d, last block height: %d \n", blocks, tiling_steps, T_ij, last_T_j, last_T_i);

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