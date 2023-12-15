#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <random>

#include "coo_opt_vectorization_gpu/coo_opt_vectorization_SDDMM.cuh"
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

__device__ float dot_product_float4(
    const int k,
    const float* __restrict__ const matrixA_GPU_values_row,
    const float* __restrict__ const matrixB_transposed_GPU_values_col)
{
    // Check if the pointers are aligned to float4
    int offsetA = 4 - (reinterpret_cast<uintptr_t>(matrixA_GPU_values_row) % 16) / 4;

    float result = 0;

    // Since both points may not be alignable we align one of them and use float4 only for one
    for (int i = 0; i < offsetA && i < k; i++)
    {
        result += matrixA_GPU_values_row[i] * matrixB_transposed_GPU_values_col[i];
    }

    const float4* matrixA_GPU_values_row_float4 = reinterpret_cast<const float4*>(matrixA_GPU_values_row + offsetA);

    // calculate SUM_i row[i] * col[i]
    // Start at k and go to 0 to allow for better code for small rows too
    for (int i = 0; i < k - offsetA - 3; i += 4)
    {
        float4 a = matrixA_GPU_values_row_float4[i / 4];
        int local_offset = i + offsetA;
        result += a.x * matrixB_transposed_GPU_values_col[local_offset];
        result += a.y * matrixB_transposed_GPU_values_col[local_offset + 1];
        result += a.z * matrixB_transposed_GPU_values_col[local_offset + 2];
        result += a.w * matrixB_transposed_GPU_values_col[local_offset + 3];
    }

    // Add the rest if k is not divisible by 4
    for (int i = max(k - offsetA - 3, offsetA); i < k; i++)
    {
        result += matrixA_GPU_values_row[i] * matrixB_transposed_GPU_values_col[i];
    }

    return result;
}

// TODO: Check for alignment
// __device__ float dot_product_float2(
//     const int k,
//     const float* __restrict__ const matrixA_GPU_values_row,
//     const float* __restrict__ const matrixB_transposed_GPU_values_col)
// {
//     const float2* matrixA_GPU_values_row_float2 = reinterpret_cast<const float2*>(matrixA_GPU_values_row);
//     const float2* matrixB_transposed_GPU_values_col_float2 = reinterpret_cast<const float2*>(matrixB_transposed_GPU_values_col);

//     float2 result = make_float2(0, 0);
//     for (int i = k - 1; i >= 1; i -= 2)
//     {
//         float2 a = matrixA_GPU_values_row_float2[i / 2];
//         float2 b = matrixB_transposed_GPU_values_col_float2[i / 2];
//         result.x += a.x * b.x;
//         result.y += a.y * b.y;
//     }

//     if (k % 2 == 1)
//     {
//         result.x += matrixA_GPU_values_row[0] * matrixB_transposed_GPU_values_col[0];
//     }
//     return result.x + result.y;
// }

// Assumes matrixB_transposed_GPU_values is transposed
__global__ void naive_coo_vectorized(
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
        matrixResult_GPU_values[index] = multiplier * dot_product_float4(k, matrixA_GPU_values + (row * k), matrixB_transposed_GPU_values + (col * k));
    }
}

void compute_coo_opt_vectorization(
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

    naive_coo_vectorized<<<blocks, threadsPerBlock>>>(
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