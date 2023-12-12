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

// Float4 does not work for some reason

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "merged/merged.cuh"
#include "utils.h"

#define THREADS_PER_BLOCK 1024                                                           // this should be 1024 (unless we are running tests, then we may want to set it to something small)
#define GPU_SHARED_MEM_SIZE_BYTES 49152                                                  // size of shared mem on both the A100 and V100 GPUs = 49152 bytes
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
    const int last_chunk_size,
    const float* matrixB_transposed_GPU_values,
    const int B_col_index,  // indexes the col of B that we are computing the dot prod with in this method call
    const int k,            // number of rows of B
    float* reduction_space,
    int numChunksInTile)
{
    const float* B_col_tile_beginning =                      // don't want entire column but only a tile of it
        (matrixB_transposed_GPU_values + B_col_index * k) +  // the thing in parens is ptr to start of col of B that we are computing the dot prod with
        tiling_step * normal_tile_size;

    // setup float4                                                  // Float4 does not work for some reason
    const float4* tile_float4 = reinterpret_cast<const float4*>(tile);
    const float4* B_col_tile_beginning_float4 = reinterpret_cast<const float4*>(B_col_tile_beginning);
    float4 sum_of_chunks = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // // setup float2
    // const float2* tile_float2 = reinterpret_cast<const float2*>(tile);
    // const float2* B_col_tile_beginning_float2 = reinterpret_cast<const float2*>(B_col_tile_beginning);
    // float2 sum_of_chunks = make_float2(0.0f, 0.0f);

    for (int i = tid; i < numChunksInTile; i += bdim)
    {
        // compute the chunk of the dot product that this thread is responsible for
        // Use float4 for the dot product
        const float4 vector1_beginning = tile_float4[i * 2];
        const float4 vector2_beginning = B_col_tile_beginning_float4[i * 2];
        sum_of_chunks.x += vector1_beginning.x * vector2_beginning.x;
        sum_of_chunks.y += vector1_beginning.y * vector2_beginning.y;
        sum_of_chunks.z += vector1_beginning.z * vector2_beginning.z;
        sum_of_chunks.w += vector1_beginning.w * vector2_beginning.w;
        // float2 vector1_beginning = tile_float2[i * 2];
        // float2 vector2_beginning = B_col_tile_beginning_float2[i * 2];
        // sum_of_chunks.x += vector1_beginning.x * vector2_beginning.x;
        // sum_of_chunks.y += vector1_beginning.y * vector2_beginning.y;
        // vector1_beginning = tile_float2[i * 2 + 1];
        // vector2_beginning = B_col_tile_beginning_float2[i * 2 + 1];
        // sum_of_chunks.x += vector1_beginning.x * vector2_beginning.x;
        // sum_of_chunks.y += vector1_beginning.y * vector2_beginning.y;
    }
    // FLOAT4 let last thread take care of the last chunk (bc it might be smaller than 4)
    if (tid == THREADS_PER_BLOCK - 1)
    {
        const int beginning = numChunksInTile << 2;
        for (int i = 0; i < last_chunk_size; i++)
        {
            // cant unroll here because we only know last_chunk_size at runtime
            sum_of_chunks.x += (tile + beginning)[i] * (B_col_tile_beginning + beginning)[i];
        }
    }
    float sum_of_chunks_synced = sum_of_chunks.x + sum_of_chunks.y + sum_of_chunks.z + sum_of_chunks.w;

    // // FLOAT2 let the last thread take care of the last chunk (bc it might be smaller than 2)
    // if (tid == THREADS_PER_BLOCK - 1)
    // {
    //     const int beginning = numChunksInTile << 2;
    //     for (int i = 0; i < last_chunk_size; i++)
    //     {
    //         // cant unroll here because we only know last_chunk_size at runtime
    //         sum_of_chunks.x += (tile + beginning)[i] * (B_col_tile_beginning + beginning)[i];
    //     }
    // }
    // float sum_of_chunks_synced = sum_of_chunks.x + sum_of_chunks.y;

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
        sum_of_chunks_synced = reduction_space[tid];
        // 3: now warp reduce those vals
        sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 16);  // the final reduction comprises a max of 32 values so hard coding 16 here is fine
        sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 8);
        sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 4);
        sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 2);
        sum_of_chunks_synced += __shfl_down_sync(0xffffffff, sum_of_chunks_synced, 1);
        // 4: now the final sum should sit int the first thread of the first warp (= thread 0) so it can return it
        if (tid == 0)
        {
            return sum_of_chunks_synced;
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
    const int* __restrict__ const matrixC_GPU_row_ptr,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values,
    const int row_mem_size,
    const int num_tiles)
{
    // return if no non zeros in current row of C
    const int bid = blockIdx.x;
    int tile_index = bid / num_rows;  // this is called tiling_step in tiled_tiles.cu
    int row_index = bid - (tile_index * num_rows);
    if (matrixC_GPU_row_ptr[row_index + 1] - matrixC_GPU_row_ptr[row_index] == 0)
    {
        return;
    }

    const int tid = threadIdx.x;
    const int bdim = blockDim.x;

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
    // __shared__ float tile[COMPUTATION_SHARED_MEM];  // This would be static shared mem allocation.
    extern __shared__ float tile[];  // Dynamic shared memory allocation. That way we can use all the shared memory that is available to us. THis would not be possible with static shared mem allocation.

    int last_chunk_size = curr_tile_size & 3;   // this is less expensive than a modulo op
    int numChunksInTile = curr_tile_size >> 2;  // only the number of chunks that are fully unrolled
    const float* tile_start =
        matrixA_GPU_values + (row_index * k) +  // start of the row of A that we are working on
        (tile_index * COMPUTATION_SHARED_MEM);  // shift by the number of floats that are part of the previous tiles

    // copy all but the last chunk
    // float4* tile_float4 = reinterpret_cast<float4*>(tile);
    float2* tile_float2 = reinterpret_cast<float2*>(tile);
    const float2* tile_start_float2 = reinterpret_cast<const float2*>(tile_start);
    // const float4* tile_float4_start = reinterpret_cast<const float4*>(tile_start);
    for (int i = tid; i < numChunksInTile; i += bdim)  // i steps through the chunks
    {
        // const float4 to_print = tile_float4_start[i];            // SOMEHOW FLOAT4 DOES NOT WORK
        // printf("value x if i = %d: %f\n", i, to_print.x);
        // printf("value y if i = %d: %f\n", i, to_print.y);
        // printf("value z if i = %d: %f\n", i, to_print.z);
        // printf("value w if i = %d: %f\n", i, to_print.w);
        // return;
        // tile_float4[i] = *((float4*)tile_start + i);
        // float4 to_print2 = tile_float4[i];
        // printf("to_print2 x: %f\n", to_print2.x);
        // printf("to_print2 y: %f\n", to_print2.y);
        // printf("to_print2 z: %f\n", to_print2.z);
        // printf("to_print2 w: %f\n", to_print2.w);
        float2 to_copy = tile_start_float2[i * 2];
        tile_float2[i * 2] = to_copy;
        to_copy = tile_start_float2[i * 2 + 1];
        tile_float2[i * 2 + 1] = to_copy;
    }
    // last chunk
    if (tid == THREADS_PER_BLOCK - 1)
    {
        for (int i = numChunksInTile << 2; i < curr_tile_size; i++)
        {
            // cant unroll here because we only know last_chunk_size at runtime
            tile[i] = tile_start[i];
        }
    }

    __syncthreads();  // this is a barrier

    ////////////////    ACTUAL COMPUTATION    ////////////////
    // iterate over all elems OF THE ENTIRE ROW OF C (that this block is working on)
    extern __shared__ float reduction_space[sizeof(float) << 5];  // this is the shared mem that we use for the final reduction
    for (int elem_index = matrixC_GPU_row_ptr[row_index]; elem_index < matrixC_GPU_row_ptr[row_index + 1]; elem_index++)
    {
        float dot_prod = tiled_dot_product_thread_subset_m(
            tid,
            bdim,
            tile,
            tile_index,
            COMPUTATION_SHARED_MEM,
            curr_tile_size,
            last_chunk_size,
            matrixB_transposed_GPU_values,
            matrixC_GPU_col_indices[elem_index],  // col of B for dot product is the same col in which the nonzero of C sits
            k,
            reduction_space,
            numChunksInTile);

        if (tid == 0)  // in tiled_dot_product_thread_subset only thread 0 returns something so only it needs to do the addition
        {
            atomicAdd(&matrixResult_GPU_values[elem_index], dot_prod * matrixC_GPU_values[elem_index]);
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

    // Allow the GPU to use all available shared memory
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-7-x
    CUDA_CHECK(cudaFuncSetAttribute(merged_m, cudaFuncAttributeMaxDynamicSharedMemorySize, GPU_SHARED_MEM_SIZE_BYTES));

    // call main kernel
    merged_m<<<blocks, THREADS_PER_BLOCK, GPU_SHARED_MEM_SIZE_BYTES>>>(
        m,
        k,
        matrixA_GPU_values,
        matrixB_transposed_GPU_values,
        matrixC_GPU_values,
        matrixC_GPU_row_ptr,
        matrixC_GPU_col_indices,
        matrixResult_GPU_values,
        k,
        num_tiles);
    // Aggregate the return value of the kernel
    CUDA_CHECK(cudaGetLastError());
    // CUDA_CHECK(cudaDeviceSynchronize()); not needed since the gpu timer does this for us
}