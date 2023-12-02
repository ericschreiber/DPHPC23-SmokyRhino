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
    const int tid,
    const int bdim,
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

    // setup float4
    const float4* tile_float4 = (float4*)tile;
    const float4* B_col_tile_beginning_float4 = (float4*)B_col_tile_beginning;

    float4 sum_of_chunks = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    int numChunksInTile = (curr_tile_size + 3) >> 2;
    for (int i = tid; i < numChunksInTile - 1; i += bdim)
    {
        // compute the chunk of the dot product that this thread is responsible for
        float4 vector1_beginning = tile_float4[i];
        float4 vector2_beginning = B_col_tile_beginning_float4[i];
        sum_of_chunks.x += vector1_beginning.x * vector2_beginning.x;
        sum_of_chunks.y += vector1_beginning.y * vector2_beginning.y;
        sum_of_chunks.z += vector1_beginning.z * vector2_beginning.z;
        sum_of_chunks.w += vector1_beginning.w * vector2_beginning.w;
    }
    // let thread 0 take care of the last chunk (bc it might be smaller than 4)
    if (tid == THREADS_PER_BLOCK - 1)
    {
        int loop_end = curr_tile_size & 3;  // this is less expensive than a modulo op
        for (int i = 0; i < loop_end; i++)
        {
            // cant unroll here because we only know last_chunk_size at runtime
            sum_of_chunks.x += (tile + ((numChunksInTile - 1) << 2))[i] * (B_col_tile_beginning + ((numChunksInTile - 1) << 2))[i];
        }
    }

    float sum_of_chunks_synced = sum_of_chunks.x + sum_of_chunks.y + sum_of_chunks.z + sum_of_chunks.w;
    __syncthreads();  // all threads wait togehter here before we reduce their results

    // WARP-WIDE REDUCTION
    // i.e. reduce the sum_of_chunks varible (that each thread has) into one value per warp
    // this used to be a loop but we unrolled it bc why not (maybe the compiler can do some optimizations w/ it now)
    sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 16);
    sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 8);
    sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 4);
    sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 2);
    sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 1);

    // COMMUNICATE RESULTS VIA SHARED MEMORY
    // 1: each "first thread" of a warp writes the sum into shared mem
    if ((tid & 0x1f) == 0)  // 0x1f = 31 = 11111 in binary
    {
        reduction_space[tid >> 5] = sum_of_chunks_synced;
    }
    __syncthreads();

    // FINAL REDUCTION
    if (tid < 32)  // only use the threads from the first warp for the final reduction
    {
        // 2: load the vals that we just wrote into shared mem into variables of the threads of the first warp
        float val = reduction_space[tid];
        // 3: now warp reduce those vals
        val += __shfl_down_sync(0xffffffff, val, 16);  // the final reduction comprises a max of 32 values so hard coding 16 here is fine
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        // 4: now the final sum should sit int the first thread of the first warp (= thread 0) so it can return it
        if (tid == 0)
        {
            return val;
        }
    }
}

// this is the kernel function.
// assumes matrixB_transposed_GPU_values is transposed.
__global__ void merged_m(
    const int num_rows,  // number of rows of A
    const int k,         // number of rows of B
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
    const int bid = blockIdx.x;
    if (matrixC_GPU_row_ptr[bid + 1] - matrixC_GPU_row_ptr[bid] == 0)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int bdim = blockDim.x;
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    // This is the main part that I moved over from Erics implementation

    int tile_index = bid / num_rows;  // this is called tiling_step in tiled_tiles.cu
    int row_index = bid - (tile_index * num_rows);
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    ////////////////    COMPUTE CURR TILE SIZE    ////////////////
    int curr_tile_size;
    if (tile_index == num_tiles - 1)  // if we are working on the last tile of the row
    {
        curr_tile_size = row_mem_size - (tile_index * COMPUTATION_SHARED_MEM);
    }
    else  // else is usually not good but here all threads of a block will take the same branch so we don't have this thread divergence issue
    {
        curr_tile_size = COMPUTATION_SHARED_MEM;
    }

    ////////////////    COPY TILE INTO SHARED MEM (NOW IN PARALLEL)    ////////////////
    __shared__ float tile[COMPUTATION_SHARED_MEM];
    int last_chunk_size = curr_tile_size & 3;  // this is less expensive than a modulo op
    int numChunksInTile = (curr_tile_size + 3) >> 2;
    const float* tile_start =
        matrixA_GPU_values + (row_index * k) +  // start of the row of A that we are working on
        (tile_index * COMPUTATION_SHARED_MEM);  // shift by the number of floats that are part of the previous tiles
    // do all but the last chunk
    int q;
    const float* start;
    for (int i = tid; i < numChunksInTile - 1; i += bdim)  // i steps through the chunks
    {
        q = i << 2;  // shared mem is indexed in bytes
        // can't be i = i << 2 because then the i in the for loop would change
        // unrolling (because compiler likes unrolling I guess)
        start = tile_start + q;  // kinda fishy with const in there
        tile[q] = *(start);
        tile[q + 1] = *(start + 1);
        tile[q + 2] = *(start + 2);
        tile[q + 3] = *(start + 3);
    }
    // last chunk
    if (tid == THREADS_PER_BLOCK - 1)
    {
        for (int i = (numChunksInTile - 1) << 2; i < last_chunk_size; i++)  // i now steps through the elems in a chunk (different to the loop above) i.e. no need to multiply by 4
        {
            // cant unroll here because we only know last_chunk_size at runtime
            tile[i] = *(tile_start + i);
        }
    }

    __syncthreads();  // this is a barrier

    ////////////////    ACTUAL COMPUTATION    ////////////////
    // iterate over all elems OF THE ENTIRE ROW OF C (that this block is working on)
    __shared__ float reduction_space[sizeof(float) << 5];  // this is the shared mem that we use for the final reduction
    for (int elem_index = matrixC_GPU_row_ptr[row_index]; elem_index < matrixC_GPU_row_ptr[row_index + 1]; elem_index++)
    {
        float dot_prod = tiled_dot_product_thread_subset_m(
            tid,
            bdim,
            tile,
            tile_index,
            COMPUTATION_SHARED_MEM,
            curr_tile_size,
            matrixB_transposed_GPU_values,
            matrixC_GPU_col_indices[elem_index],  // col of B for dot product is the same col in which the nonzero of C sits
            k,
            reduction_space);
        if (tid == 0)  // in tiled_dot_product_thread_subset only thread 0 returns something so only it needs to do the addition
        {
            atomicAdd(&matrixResult_GPU_values[elem_index], dot_prod * matrixC_GPU_values[elem_index]);
            // TODO: in merged version use dot_product_float4 from coo_opt_vectorization_SDDMM.cu instead of tiled_dot_product_thread_subset (which is not vectorized)
        }
    }
}

// this is the function that is called from the outside and that launches the calls to the kernel function
void compute_m(
    const int m,
    const int k,  // number of rows of B
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
    int blocks = m * num_tiles;
    ////////////////////////////////////////////////////////////////

    // call main kernel
    merged_m<<<blocks, THREADS_PER_BLOCK>>>(
        m,
        k,
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
    // CUDA_CHECK(cudaDeviceSynchronize()); not needed since the gpu timer does this for us
}