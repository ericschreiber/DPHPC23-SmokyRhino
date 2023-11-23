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

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "tiled_tiles/tiled_tiles.cuh"
#include "utils.h"

#define THREADS_PER_BLOCK 2
#define GPU_SHARED_MEM_SIZE_BYTES 8 + 32 * sizeof(float)                               // this is the size of shared mem on both the A100 and V100 GPUs.
                                                                                       // can force tiling (e.g. for testing) by setting this to something small.
#define COMPUTATION_SHARED_MEM_BYTES (GPU_SHARED_MEM_SIZE_BYTES - 32 * sizeof(float))  // reserve 32 floats for last reduction in tiled_dot_product_thread_subset
#define COMPUTATION_SHARED_MEM (COMPUTATION_SHARED_MEM_BYTES / sizeof(float))          // unit: floats

// std::count can't be used on the GPU
__device__ int count(const int* arr, int len, const int row_index)
{
    int count = 0;
    for (int i = 0; i < len; i++)
    {
        if (arr[i] == row_index)
        {
            count++;
        }
    }
    return count;
}

// helper function that abstracts away the indexing logic of computing a tiled dot product
// in the updated version, a thread does not compute the whole dot product of the tile but only a subset of it
__device__ float tiled_dot_product_thread_subset(
    const float* tile,
    const int tiling_step,
    const int normal_tile_size,
    const int curr_tile_size,  // curr_tile_size can be smaller than normal_tile_size if we are working on the last tile of a row
    const float* matrixB_transposed_GPU_values,
    const int B_col_index,  // indexes the col of B that we are computing the dot prod with in this method call
    const int k)            // number of rows of B
{
    const float* B_col_beginning = matrixB_transposed_GPU_values + B_col_index * k;        // ptr to start of col of B that we are computing the dot prod with
    const float* B_col_tile_beginning = B_col_beginning + tiling_step * normal_tile_size;  // don't want entire column but only a tile of it
    float sum_of_chunks = 0;
    int numChunksInTile = ceilf((float)curr_tile_size / 4);  // ceil needed in case the current tile is smaller than 4
    for (int i = threadIdx.x; i < numChunksInTile - 1; i += blockDim.x)
    {
        // compute the chunk of the dot product that this thread is responsible for
        const float* vector1_beginning = tile + i * 4;
        const float* vector2_beginning = B_col_tile_beginning + i * 4;
        sum_of_chunks += vector1_beginning[0] * vector2_beginning[0];
        sum_of_chunks += vector1_beginning[1] * vector2_beginning[1];
        sum_of_chunks += vector1_beginning[2] * vector2_beginning[2];
        sum_of_chunks += vector1_beginning[3] * vector2_beginning[3];
    }
    // let thread 0 take care of the last chunk (bc it might be smaller than 4)
    if (threadIdx.x == 0)
    {
        int last_chunk_size = curr_tile_size % 4;
        if (last_chunk_size != 0)
        {
            const float* vector1_beginning = tile + (numChunksInTile - 1) * 4;
            const float* vector2_beginning = B_col_tile_beginning + (numChunksInTile - 1) * 4;
            for (int i = 0; i < last_chunk_size; i++)
            {
                // cant unroll here because we only know last_chunk_size at runtime
                sum_of_chunks += vector1_beginning[i] * vector2_beginning[i];
            }
        }
    }

    __syncthreads();  // all threads wait togehter here before we reduce their results

    // warp-wide reduction
    unsigned mask = __activemask();
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        sum_of_chunks += __shfl_down_sync(mask, sum_of_chunks, offset);
    }

    // reduce the result of the warp-wide reduction to a single value
    extern __shared__ float reduction_space[32 * sizeof(float)];  // in cuda we allocate shared mem in bytes
    if (threadIdx.x % warpSize == 0)
    {
        reduction_space[threadIdx.x / warpSize] = sum_of_chunks;
    }

    __syncthreads();

    // thread 0 sweeps over the values and returns the sum.
    // I guess this could also be done in a tree like fashion but log_2(32) is 5 (and 5 vs 32 steps shouldnt make a big difference I think).
    if (threadIdx.x == 0)
    {
        float fresh_sum = 0;
        for (int i = 0; i < warpSize; i++)
        {
            fresh_sum += reduction_space[i];
        }
        return fresh_sum;
    }
}

// helper function that abstracts away the indexing logic of grabbing the factor from C/writing result into result matrix (both are sparse matrices)
__device__ void elem_compute(
    float* tile,                 // ptr to tile in shared mem
    const int tiling_step,       // index of the current tile (in the set of all tiles)
    const int normal_tile_size,  // normal_tile_size is the size of a tile that is not the last tile of a row
    const int curr_tile_size,    // curr_tile_size can be smaller than normal_tile_size if we are working on the last tile of a row
    const float* matrixB_transposed_GPU_values,
    const float* matrixC_GPU_values,
    float* matrixResult_GPU_values,
    const int* matrixC_GPU_col_indices,
    const int k,       // number of rows of B
    const int offset)  // offset into matrixC_GPU_values and matrixResult_GPU_values
{
    float dot_prod = tiled_dot_product_thread_subset(
        tile,
        tiling_step,
        normal_tile_size,
        curr_tile_size,
        matrixB_transposed_GPU_values,
        matrixC_GPU_col_indices[offset],  // col of B for dot product is the same col in which the nonzero of C sits
        k);
    if (threadIdx.x == 0)
    {
        // no need for atomic add since only thread 0 is writing back (all the partial sums from the other thread have already been reduced)
        matrixResult_GPU_values[offset] += dot_prod * matrixC_GPU_values[offset];
    }
}

// this is the kernel function.
// assumes matrixB_transposed_GPU_values is transposed.
__global__ void tiled_tiles(
    const int k,  // number of rows of B
    const int numElementsC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values,
    const int* prevBlocksWorkAll,
    const int* tiles_sizes,
    const int tiling_steps)
{
    ////////////////    SETUP NECESSARY VARS    ////////////////
    int row_index = blockIdx.x;                                            // holds bc we have set up one block per row so n-th block will take on n-th row of A
    const float* A_vals_row_start = matrixA_GPU_values + (row_index * k);  // pointer to beginning of row of A that this thread block is working on.
    int prevBlocksWork = prevBlocksWorkAll[blockIdx.x];
    int nnzs = prevBlocksWorkAll[row_index + 1] - prevBlocksWorkAll[row_index];  // number of nnzs in this row of C (= amount of work for thread block)

    ////////////////    MAIN LOOP    ////////////////
    for (int tiling_step = 0; tiling_step < tiling_steps; tiling_step++)
    {
        ////////////////    COMPUTE SIZE OF CURR TILE    ////////////////
        int curr_tile_size = tiles_sizes[tiling_step];

        ////////////////    THREAD 0: COPY TILE INTO SHARED MEM    ////////////////
        // TODO: this can very likely also be parallelized over the threads in the block
        // decalare a ptr to a shared mem region (this needs to be done so that threads other than thread 0 can access the tile later on)
        extern __shared__ float tile[COMPUTATION_SHARED_MEM_BYTES];
        if (threadIdx.x == 0)
        {
            // copy the tile into shared mem (I think this copying happens float by float (bc of pointer arithmetic) but maybe also byte by byte (?))
            for (int i = 0; i < curr_tile_size; i++)
            {
                // second summand (in parentheses) = offset of the tile that we're working on in this iteration of outer loop
                tile[i] = *(A_vals_row_start + (tiling_step * COMPUTATION_SHARED_MEM) + i);
            }
        }

        __syncthreads();  // this is a barrier

        ////////////////    ACTUAL COMPUTATION    ////////////////
        for (int elem_index = 0; elem_index < nnzs; elem_index++)  // iterate over all elems OF THE ENTIRE ROW OF C (that this block is working on)
        {
            int offset = prevBlocksWork + elem_index;
            elem_compute(
                tile,
                tiling_step,
                COMPUTATION_SHARED_MEM,
                curr_tile_size,
                matrixB_transposed_GPU_values,
                matrixC_GPU_values,
                matrixResult_GPU_values,
                matrixC_GPU_col_indices,
                k,
                offset);
        }
    }
}

// precompute stuff that the main kernels need with 1 thread (doing it inside of the main kernel would make them all do the same precomputations)
__global__ void precomputation(
    const int numElementsC,
    const int* __restrict__ const matrixC_GPU_row_indices,
    int* prevBlocksWork,
    int numBlocks,
    int* tiles_sizes,
    int tiling_steps,
    int row_mem_size)
{
    // populate prevBlocksWork
    for (int i = 0; i < numBlocks; i++)
    {
        int last = 0;
        if (i != 0)
            last = prevBlocksWork[i];
        int counter = count(matrixC_GPU_row_indices, numElementsC, i);
        prevBlocksWork[i + 1] += last + counter;
    }

    // populate tiles_sizes
    for (int i = 0; i < tiling_steps; i++)
    {
        tiles_sizes[i] = COMPUTATION_SHARED_MEM;
        // last tile might be smaller than the regular tile size
        if (i == tiling_steps - 1 && row_mem_size % COMPUTATION_SHARED_MEM != 0)
        {
            tiles_sizes[i] = row_mem_size % COMPUTATION_SHARED_MEM;
        }
    }
}

// this is the function that is called from the outside and that launches the calls to the kernel function
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
    int blocks = m;  // one block per row of A
    // allocate array that will be populated by the precomputation kernel
    int* prevBlocksWork;
    int row_mem_size = k * sizeof(float);                                           // size of a row of A (= non-sparse) in mem
    int tiling_steps = ceil(row_mem_size / (float)(COMPUTATION_SHARED_MEM_BYTES));  // #pieces that we need to chop row of A into (bc it might not fit into shared mem)
    int* tiles_sizes;
    CUDA_CHECK(cudaMalloc((void**)&prevBlocksWork, (blocks + 1) * sizeof(int)));  // + 1 needed for the computation (for last block) of nnzs in the main kernel
    CUDA_CHECK(cudaMalloc((void**)&tiles_sizes, tiling_steps * sizeof(int)));
    // run the precomputation kernel
    precomputation<<<1, 1>>>(numElementsC, matrixC_GPU_row_indices, prevBlocksWork, blocks, tiles_sizes, tiling_steps, row_mem_size);

    dim3 threadsPerBlock(THREADS_PER_BLOCK);

    // call main kernel
    // TODO: currently I am spawning dynamic shared mem, maybe non dynamic shared mem is better?
    tiled_tiles<<<blocks, threadsPerBlock, GPU_SHARED_MEM_SIZE_BYTES>>>(
        k,
        numElementsC,
        matrixA_GPU_values,
        matrixB_transposed_GPU_values,
        matrixC_GPU_values,
        matrixC_GPU_row_indices,
        matrixC_GPU_col_indices,
        matrixResult_GPU_values,
        prevBlocksWork,
        tiles_sizes,
        tiling_steps);
    // Aggregate the return value of the kernel
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // free the array prevBlocksWork on GPU
    CUDA_CHECK(cudaFree(prevBlocksWork));
    CUDA_CHECK(cudaFree(tiles_sizes));
}