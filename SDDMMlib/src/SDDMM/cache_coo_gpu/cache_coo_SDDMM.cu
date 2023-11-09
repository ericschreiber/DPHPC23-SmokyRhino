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

#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "cache_coo_gpu/cache_coo_SDMM.cuh"
#include "utils.h"

// TODO: use #defines for as much as possible (e.g. shared mem size, block size, etc.)

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

// helper function that abstracts away the indexing logic of computing a dot product
__device__ float dot_product(
    const float* vector1_beginning,
    const float* vector2_beginning,
    const int size)
{
    float result = 0;
    for (int i = 0; i < size; i++)
    {
        result += vector1_beginning[i] * vector2_beginning[i];
    }
    return result;
}

// helper function that abstracts away the indexing logic of computing a tiled dot product
__device__ float tiled_dot_product(
    const float* tile,
    const int tiling_step,
    const int normal_tile_size,
    const int curr_tile_size,  // curr_tile_size can be smaller than normal_tile_size if we are working on the last tile of a row
    const float* matrixB_transposed_GPU_values,
    const int B_col_index,  // indexes the col of B that we are computing the dot prod with in this method call
    const int k)            // number of rows of B
{
    const float* A_row_tile_beginning = tile;                                              // tile is ptr to start of shared mem (which we have filled in kernel)
    const float* B_col_beginning = matrixB_transposed_GPU_values + B_col_index * k;        // ptr to start of col of B that we are computing the dot prod with
    const float* B_col_tile_beginning = B_col_beginning + tiling_step * normal_tile_size;  // don't want entire column but only a tile of it
    return dot_product(A_row_tile_beginning, B_col_tile_beginning, curr_tile_size);
}

// helper function that abstracts away the indexing logic of grabbing the factor from C/writing result into result matrix (both are sparse matrices)
__device__ void elem_compute(
    float* tile,                 // ptr to tile in shared mem
    const int tiling_step,       // index of the current tile (in the set of all tiles)
    const int elem_index,        // index of the current elem that thread is working on (in the set of all elems that thread has been assigned)
    const int normal_tile_size,  // normal_tile_size is the size of a tile that is not the last tile of a row
    const int curr_tile_size,    // curr_tile_size can be smaller than normal_tile_size if we are working on the last tile of a row
    const float* matrixB_transposed_GPU_values,
    const float* matrixC_GPU_values,
    float* matrixResult_GPU_values,
    const int* matrixC_GPU_col_indices,
    const int k,  // number of rows of B
    const int elems_base_offset)
{
    offset += elems_base_offset + elem_index;           // offset into matrixC_GPU_values and matrixResult_GPU_values
    int B_col_index = matrixC_GPU_col_indices[offset];  // we will need to use the col of B (for dot product) which is the same col in which the nonzero of C sits

    float dot_prod = tiled_dot_product(
        tile,
        tiling_step,
        normal_tile_size,
        curr_tile_size,
        matrixB_transposed_GPU_values,
        B_col_index,
        k);

    float sparse_matrix_factor = matrixC_GPU_values[offset];
    matrixResult_GPU_values[offset] += sparse_matrix_factor * dot_prod  // += because we are doing tiling
}

// this is the kernel function.
// assumes matrixB_transposed_GPU_values is transposed.
__global__ void cache_coo(
    const int k,
    const int numElementsC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values)
{
    ////////////////    SETUP NECESSARY VARS    ////////////////
    const int shared_mem_size_bytes = 49152;                                // in bytes (this is the size of shared mem on both the A100 and V100 GPUs)
    int shared_mem_size = shared_mem_size_bytes / sizeof(float);            // shared mem size in number of floats
    int numThreadsPerBlock = blockDim.x;                                    // this holds since our thread blocks are 1D
    int row_index = blockIdx.x;                                             // holds bc we have set up one block per row so n-th block will take on n-th row of A
    int row_mem_size = k * sizeof(float);                                   // size of a row of A (= non-sparse) in mem
    int row_offset = row_index * row_mem_size;                              // offset that (if added to base_ptr of A) points to curr_row of A
    int A_vals_row_start = matrixA_GPU_values + row_offset;                 // pointer to beginning of row of A that this thread block is working on
    int nnzs = count(matrixC_GPU_row_indices, numElementsC, row_index);     // number of nnzs in this row of C (= amount of work for thread block)
    int tiling_steps = ceil((float)row_mem_size / (float)shared_mem_size);  // # pieces that we need to chop the row of A into (bc it might not fit into shared mem)

    ////////////////    MAIN LOOP    ////////////////
    for (int tiling_step = 0; tiling_step < tiling_steps; tiling_step++)
    {
        ////////////////    COMPUTE SIZE OF CURR TILE    ////////////////
        int curr_tile_size = shared_mem_size;
        // if row mem size is not divisible by shared mem size then the last tile will be smaller than shared mem size
        if (tiling_step == tiling_steps - 1 && row_mem_size % shared_mem_size != 0)
        {
            curr_tile_size = row_mem_size % shared_mem_size;
        }

        ////////////////    THREAD 0: COPY TILE INTO SHARED MEM    ////////////////
        // TODO: this can very likely also be parallelized over the threads in the block
        // decalare a ptr to a shared mem region (this needs to be done so that threads other than thread 0 can access the tile later on)
        extern __shared__ float tile[];
        if (threadIdx.x == 0)
        {
            tile = new float[curr_tile_size];  // allocate space for the tile in shared mem
            // copy the tile into shared mem (I think this copying happens float by float (bc of pointer arithmetic) but maybe also byte by byte (?))
            for (int i = 0; i < curr_tile_size; i++)
            {
                int tile_offset = tiling_step * shared_mem_size;  // need to offset indexing by the tile that we're working on in this iteration of outer loop
                tile[i] = matrixA_GPU_values[A_vals_row_start + tile_offset + i];
            }
        }
        __syncthreads();  // this is a barrier

        ////////////////    PREPARE THE ACTUAL COMPUTATION    ////////////////
        // compute num_nnzs_per_thread (which will be needed by the elem_compute helper function)
        int num_nnzs_per_thread[numThreadsPerBlock];
        for (int i = 0; i < numThreadsPerBlock; i++)
        {
            num_nnzs_per_thread[i] = nnzs / numThreadsPerBlock;
            // if numThreadsPerBlock does not divide nnzs some thread will have to do one more elem
            if (i < nnzs % numThreadsPerBlock)
            {
                num_nnzs_per_thread[i]++;
            }
        }

        // compute the base offset into matrixC_GPU_values and matrixResult_GPU_values (i.e. how many vals are being worked on by threads before this one)
        int elems_base_offset = 0;
        for (int i = 0; i < threadIdx.x; i++)
        {
            elems_base_offset += num_nnzs_per_thread[i];
        }

        ////////////////    ACTUAL COMPUTATION    ////////////////
        for (int elem_index = 0; elem_index < num_nnzs_per_thread[threadIdx.x]; elem_index++)  // iterate over all elems that this thread has been assigned
        {
            elem_compute(
                &tile,
                tiling_step,
                elem_index,
                shared_mem_size,
                curr_tile_size,
                matrixB_transposed_GPU_values,
                matrixC_GPU_values,
                matrixResult_GPU_values,
                matrixC_GPU_col_indices,
                k,
                elems_base_offset);
        }
    }
}

// this is the function that is called from the outside and that launches the calls to the kernel function
void compute(
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
    dim3 threadsPerBlock(1024);
    int blocks = m;  // one block per row of A

    // call the kernel
    cache_coo<<<blocks, threadsPerBlock>>>(
        k,
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