#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "naive_coo_gpu/naive_coo_SDMM.cuh"
#include "utils.h"

__device__ float dot_product(
    const int k,
    const float* __restrict__ const matrixA_GPU_values_row,
    const float* __restrict__ const matrixB_transposed_GPU_values_col)
{
    // calculate SUM_i row[i] * col[i]
    float result = 0;
    for (int i = 0; i < k; i++)
    {
        result += matrixA_GPU_values_row[i] * matrixB_transposed_GPU_values_col[i];
    }
    return result;
}

__device__ float naive_coo_one_val(
    const int k,
    const float multiplier,
    const float* __restrict__ const matrixA_GPU_values_row,
    const float* __restrict__ const matrixB_transposed_GPU_values_col)
{
    // calculate mutiplier * SUM_i row[i] * col[i]
    return multiplier * dot_product(k, matrixA_GPU_values_row, matrixB_transposed_GPU_values_col);
}

// Assumes matrixB_transposed_GPU_values is transposed
__global__ void naive_coo(
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
        matrixResult_GPU_values[index] = naive_coo_one_val(k, multiplier, matrixA_GPU_values + (row * k), matrixB_transposed_GPU_values + (col * k));
    }
}

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
    // Iterate of matrixC_GPU_row_indices, matrixC_GPU_col_indices, matrixC_GPU_values (as they share indices)
    // Each thread will calculate one value of the result matrix

    dim3 threadsPerBlock(1024);
    int blocks = std::min(1024, (numElementsC + 1023) / 1024);

    // std::cout << "n: " << n << std::endl;
    // std::cout << "m: " << m << std::endl;
    // std::cout << "k: " << k << std::endl;
    // std::cout << "numElementsC: " << numElementsC << std::endl;
    // std::cout << "blocks: " << blocks << std::endl;
    // std::cout << "threadsPerBlock: " << threadsPerBlock.x << std::endl;

    naive_coo<<<blocks, threadsPerBlock>>>(
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