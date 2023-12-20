#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <random>

#include "coo_opt_loop_unrolled_gpu/coo_opt_loop_unrolled_SDDMM.cuh"
#include "utils.h"

// *********** IDEA ***********
//
// Loop unrolling and vectorization for the naive coo implementation
// As shown below we take the float4 optimization.
//
// ******* Testing Results *****
//
// Simple results on 5000x5000 matrices have shown the following results: (mean of 10 runs)
// - default (naive_coo): 0.429 s
// - loop unrolling (dot_product_4): 0.409 s
// - vectorization (dot_product_float2): 0.222 s
// - vectorization (dot_product_float4): 0.144 s
//
// *********** ***** ***********

// Calculate the dotproduct but loading 4 values at a time
__device__ float dot_product_4(
    const int k,
    const float* __restrict__ const matrixA_GPU_values_row,
    const float* __restrict__ const matrixB_transposed_GPU_values_col)
{
    // calculate SUM_i row[i] * col[i]
    float result = 0;
    // Start at k and go to 0 to allow for better code for small rows too
    for (int i = k - 1; i >= 3; i -= 4)
    {
        result += matrixA_GPU_values_row[i] * matrixB_transposed_GPU_values_col[i];
        result += matrixA_GPU_values_row[i - 1] * matrixB_transposed_GPU_values_col[i - 1];
        result += matrixA_GPU_values_row[i - 2] * matrixB_transposed_GPU_values_col[i - 2];
        result += matrixA_GPU_values_row[i - 3] * matrixB_transposed_GPU_values_col[i - 3];
    }

    // Add the rest if k is not divisible by 4
    for (int i = 0; i < k % 4; i++)
    {
        result += matrixA_GPU_values_row[i] * matrixB_transposed_GPU_values_col[i];
    }
    return result;
}

// Assumes matrixB_transposed_GPU_values is transposed
__global__ void naive_coo_loop_unrolled(
    const int k,
    const int numElementsC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < numElementsC)
    {
        int row = matrixC_GPU_row_indices[index];
        int col = matrixC_GPU_col_indices[index];
        float multiplier = matrixC_GPU_values[index];

        // calculate matrixResult_GPU_values[index][col] = naive_coo_one_val(multiplier, matrixA_GPU_values[row][:], matrixB_GPU_values[:][col])
        matrixResult_GPU_values[index] = multiplier * dot_product_4(k, matrixA_GPU_values + (row * k), matrixB_transposed_GPU_values + (col * k));
    }
}

void compute_coo_opt_loop_unrolled(
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
    // Iterate of matrixC_GPU_row_indices, matrixC_GPU_col_indices, matrixC_GPU_values (as they share indices)
    // Each thread will calculate one value of the result matrix

    dim3 threadsPerBlock(1024);
    int blocks = (numElementsC + threadsPerBlock.x - 1) / threadsPerBlock.x;

    naive_coo_loop_unrolled<<<blocks, threadsPerBlock>>>(
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