// here is what this GPU implementation does (and what it doesn't):
// - each thread block gets a row of the A matrix and its threads carry out all computations that involve this row. I do not do any load balancing here: if a row
//   has much more or less non-zero elements than another row this is not dealt with here.
// - the priority of this implementation is to KEEP AS MUCH AS POSSIBLE OF THE ROW THAT A BLOCK IS WORKING ON IN THE SHARED MEMORY OF THE BLOCK. if the full row
//   does not fit into shared mem then we resort to tiling.
//   - this explcit loading into the shared mem is necessary since otherwise the tiles of A would end up in L1 but data of B would also go there i.e. we could
//     not know if the data of A gets evicted from L1 but if we load it into shared mem (and nothing of B into shared mem) then we know that it will stay there.
// - the columns of B are just kept in GPU RAM and loaded from there when they are needed for the dot product i.e. we are not trying to keep them in some fast
//   memory. In case a thread has to work on more than one elem he will be assinged "consecutive" elements that correspond to a nonzero in C.
// - also, currently I am hardcoding this implementation to floats, so if we want to make it work with other datatypes we will need to change it in a few places.
//
// TILED TILES UPDATE: dont assing one col of B to a thread in a block but let all threads
// work on all blocks and then split the work between threads on the tile level. The working
// sets of the threads are not consecutive anymore but in regular intervals.

// TODO:
// make this use CSR (maybe I can use Erics cpp file for that?) and replace the precomputation by using the rowPtr array and the blockIdx.x

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "merged/merged.cuh"
#include "utils.h"

#define THREADS_PER_BLOCK 2                                                              // this should be 1024 (unless we are running tests, then we may want to set it to something small)
#define GPU_SHARED_MEM_SIZE_BYTES 136                                                    // size of shared mem on both the A100 and V100 GPUs = 49152 bytes
                                                                                         // can force tiling (e.g. for testing) by setting this to something small.
#define COMPUTATION_SHARED_MEM_BYTES (GPU_SHARED_MEM_SIZE_BYTES - (sizeof(float) << 5))  // reserve 32 floats for last reduction in tiled_dot_product_thread_subset
#define COMPUTATION_SHARED_MEM (COMPUTATION_SHARED_MEM_BYTES / sizeof(float))            // unit: floats

// helper function that abstracts away the indexing logic of computing a tiled dot product
// in the updated version, a thread does not compute the whole dot product of the tile but only a subset of it
__device__ float tiled_dot_product_thread_subset_m(
    const float* tile,
    const int tiling_step,
    const int normal_tile_size,
    const int curr_tile_size,  // curr_tile_size can be smaller than normal_tile_size if we are working on the last tile of a row
    const float* matrixB_transposed_GPU_values,
    const int B_col_index,  // indexes the col of B that we are computing the dot prod with in this method call
    const int k,            // number of rows of B
    float* reduction_space)
{
    const float* B_col_tile_beginning =                      // don't want entire column but only a tile of it
        (matrixB_transposed_GPU_values + B_col_index * k) +  // the thing in parens is ptr to start of col of B that we are computing the dot prod with
        tiling_step * normal_tile_size;
    float sum_of_chunks = 0;
    int numChunksInTile = ceilf((float)curr_tile_size / 4);  // ceil needed in case the current tile is smaller than 4
    for (int i = threadIdx.x; i < numChunksInTile - 1; i += blockDim.x)
    {
        i = i << 2;  // shared mem is indexed in bytes
        // compute the chunk of the dot product that this thread is responsible for
        const float* vector1_beginning = tile + i;
        const float* vector2_beginning = B_col_tile_beginning + i;
        sum_of_chunks += vector1_beginning[0] * vector2_beginning[0];
        sum_of_chunks += vector1_beginning[1] * vector2_beginning[1];
        sum_of_chunks += vector1_beginning[2] * vector2_beginning[2];
        sum_of_chunks += vector1_beginning[3] * vector2_beginning[3];
    }
    // let thread 0 take care of the last chunk (bc it might be smaller than 4)
    if (threadIdx.x == THREADS_PER_BLOCK - 1)
    {
        int loop_end = curr_tile_size & 3;  // this is less expensive than a modulo op
        for (int i = 0; i < loop_end; i++)
        {
            // cant unroll here because we only know last_chunk_size at runtime
            sum_of_chunks += (tile + (numChunksInTile - 1) * 4)[i] * (B_col_tile_beginning + (numChunksInTile - 1) * 4)[i];
        }
    }

    __syncthreads();  // all threads wait togehter here before we reduce their results

    // WARP-WIDE REDUCTION
    // i.e. reduce the sum_of_chunks varible (that each thread has) into one value per warp
    // this used to be a loop but we unrolled it bc why not (maybe the compiler can do some optimizations w/ it now)
    sum_of_chunks += __shfl_down_sync(0xffffffff, sum_of_chunks, 16);
    sum_of_chunks += __shfl_down_sync(0xffffffff, sum_of_chunks, 8);
    sum_of_chunks += __shfl_down_sync(0xffffffff, sum_of_chunks, 4);
    sum_of_chunks += __shfl_down_sync(0xffffffff, sum_of_chunks, 2);
    sum_of_chunks += __shfl_down_sync(0xffffffff, sum_of_chunks, 1);

    // COMMUNICATE RESULTS VIA SHARED MEMORY
    // 1: each "first thread" of a warp writes the sum into shared mem
    if ((threadIdx.x & 0x1f) == 0)  // 0x1f = 31 = 11111 in binary
    {
        reduction_space[threadIdx.x >> 5] = sum_of_chunks;
    }
    __syncthreads();

    // FINAL REDUCTION
    if (threadIdx.x < 32)  // only use the threads from the first warp for the final reduction
    {
        // 2: load the vals that we just wrote into shared mem into variables of the threads of the first warp
        float val = reduction_space[threadIdx.x];
        // 3: now warp reduce those vals
        val += __shfl_down_sync(0xffffffff, val, 16);  // the final reduction comprises a max of 32 values so hard coding 16 here is fine
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        // 4: now the final sum should sit int the first thread of the first warp (= thread 0) so it can return it
        if (threadIdx.x == 0)
        {
            return val;
        }
    }
}

// helper function that abstracts away the indexing logic of grabbing the factor from C/writing result into result matrix (both are sparse matrices)
__device__ void elem_compute_m(
    float* tile,                 // ptr to tile in shared mem
    const int tiling_step,       // index of the current tile (in the set of all tiles)
    const int normal_tile_size,  // normal_tile_size is the size of a tile that is not the last tile of a row
    const int curr_tile_size,    // curr_tile_size can be smaller than normal_tile_size if we are working on the last tile of a row
    const float* matrixB_transposed_GPU_values,
    const float* matrixC_GPU_values,
    float* matrixResult_GPU_values,
    const int* matrixC_GPU_col_indices,
    const int k,       // number of rows of B
    const int offset,  // offset into matrixC_GPU_values and matrixResult_GPU_values
    float* reduction_space)
{
    float dot_prod = tiled_dot_product_thread_subset_m(
        tile,
        tiling_step,
        normal_tile_size,
        curr_tile_size,
        matrixB_transposed_GPU_values,
        matrixC_GPU_col_indices[offset],  // col of B for dot product is the same col in which the nonzero of C sits
        k,
        reduction_space);
    if (threadIdx.x == 0)  // in tiled_dot_product_thread_subset only thread 0 returns something so only it needs to do the addition
    {
        atomicAdd(&matrixResult_GPU_values[offset], dot_prod * matrixC_GPU_values[offset]);
        // TODO: in merged version use dot_product_float4 from coo_opt_vectorization_SDDMM.cu instead of tiled_dot_product_thread_subset (which is not vectorized)
    }
}

// this is the kernel function.
// assumes matrixB_transposed_GPU_values is transposed.
__global__ void merged_m(
    const int num_rows,  // number of rows of A
    const int k,         // number of rows of B
    const int numElementsCrowPtr,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_row_ptr,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values,
    const int row_mem_size,
    const int num_tiles)
{
    // return if no non zeros in current row of C
    if (matrixC_GPU_row_ptr[blockIdx.x + 1] - matrixC_GPU_row_ptr[blockIdx.x] == 0)
    {
        return;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // This is the main part that I moved over from Erics implementation

    int tile_index = blockIdx.x / num_rows;  // this is called tiling_step in tiled_tiles.cu
    int row_index = blockIdx.x - (tile_index * num_rows);
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////    COMPUTE CURR TILE SIZE    ////////////////
    int curr_tile_size;
    if (tile_index == num_tiles - 1)  // if we are working on the last tile of the row
    {
        curr_tile_size = row_mem_size - ((num_tiles - 1) * COMPUTATION_SHARED_MEM);
    }
    else  // else is usually not good but here all threads of a block will take the same branch so we don't have this thread divergence issue
    {
        curr_tile_size = COMPUTATION_SHARED_MEM;
    }

    ////////////////    COPY TILE INTO SHARED MEM (NOW IN PARALLEL)    ////////////////
    __shared__ float tile[COMPUTATION_SHARED_MEM_BYTES];
    int last_chunk_size = curr_tile_size & 3;                // this is less expensive than a modulo op
    int numChunksInTile = ceilf((float)curr_tile_size / 4);  // ceil needed in case the current tile is smaller than 4
    const float* tile_start =
        matrixA_GPU_values + (row_index * k) +  // start of the row of A that we are working on
        (tile_index * COMPUTATION_SHARED_MEM);  // shift by the number of floats that are part of the previous tiles
    // do all but the last chunk
    for (int i = threadIdx.x; i < numChunksInTile - 1; i += blockDim.x)  // i steps through the chunks
    {
        i = i << 2;  // shared mem is indexed in bytes
        // unrolling (because compiler likes unrolling I guess)
        tile[i] = *(tile_start + i);
        tile[i + 1] = *(tile_start + i + 1);
        tile[i + 2] = *(tile_start + i + 2);
        tile[i + 3] = *(tile_start + i + 3);
    }
    // last chunk
    if (threadIdx.x == THREADS_PER_BLOCK - 1)
    {
        for (int i = 0; i < last_chunk_size; i++)  // i now steps through the elems in a chunk (different to the loop above) i.e. no need to multiply by 4
        {
            // cant unroll here because we only know last_chunk_size at runtime
            tile[(numChunksInTile - 1) * 4 + i] = *(tile_start + (numChunksInTile - 1) * 4 + i);
        }
    }

    __syncthreads();  // this is a barrier

    ////////////////    ACTUAL COMPUTATION    ////////////////
    // iterate over all elems OF THE ENTIRE ROW OF C (that this block is working on)
    __shared__ float reduction_space[sizeof(float) << 5];  // this is the shared mem that we use for the final reduction
    for (int elem_index = matrixC_GPU_row_ptr[row_index]; elem_index < matrixC_GPU_row_ptr[row_index + 1]; elem_index++)
    {
        elem_compute_m(
            tile,
            tile_index,
            COMPUTATION_SHARED_MEM,
            curr_tile_size,
            matrixB_transposed_GPU_values,
            matrixC_GPU_values,
            matrixResult_GPU_values,
            matrixC_GPU_col_indices,
            k,
            elem_index,
            reduction_space);
    }
}

// this is the function that is called from the outside and that launches the calls to the kernel function
void compute_m(
    const int m,
    const int k,  // number of rows of B
    const int numElementsCrowPtr,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_row_ptr,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values)
{
    ////////////////////////////////////////////////////////////////
    // This is also part of Erics code
    const int tile_size = COMPUTATION_SHARED_MEM;
    const int num_tiles = (k + tile_size - 1) / tile_size;
    int blocks = (numElementsCrowPtr - 1) * num_tiles;
    ////////////////////////////////////////////////////////////////

    // call main kernel
    merged_m<<<blocks, THREADS_PER_BLOCK>>>(
        m,
        k,
        numElementsCrowPtr,
        matrixA_GPU_values,
        matrixB_transposed_GPU_values,
        matrixC_GPU_values,
        matrixC_GPU_row_indices,
        matrixC_GPU_row_ptr,
        matrixC_GPU_col_indices,
        matrixResult_GPU_values,
        k,
        num_tiles);
    // Aggregate the return value of the kernel
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}