#include <cuda_runtime.h>

#include <iostream>

#include "naive_coo_gpu/naive_coo_SDMM.cuh"
#include "naive_csr_via_coo_gpu/naive_csr_via_coo_SDDMM.cuh"
#include "utils.h"

__global__ void CSR_to_COO_kernel(
    const int numElementsC,
    const int rowPtrSizeC,
    const int* __restrict__ const matrixC_GPU_row_indices,
    int* __restrict__ const matrixC_GPU_row_indices_complete)
{
    // Iterate over the old rowPtr and fill it with the correct values
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rowPtrSizeC)
    {
        int row = matrixC_GPU_row_indices[index];
        int rowStart = matrixC_GPU_row_indices[index];
        int rowEnd = matrixC_GPU_row_indices[index + 1];
        for (int i = rowStart; i < rowEnd; i++)
        {
            matrixC_GPU_row_indices_complete[i] = row;
        }
    }
}

// Assumes matrixB is transposed
void compute_naive_csr_via_coo(
    const int m,
    const int n,
    const int k,
    const int numElementsC,
    const int rowPtrSizeC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values)
{
    CUDA_CHECK(cudaGetLastError());
    // Make a rowPtr for all values on the GPU to get COO format and then call the naive_coo_SDMM_kernel
    int* matrixC_GPU_row_indices_complete;
    CUDA_CHECK(cudaMalloc((void**)&matrixC_GPU_row_indices_complete, sizeof(int) * numElementsC));

    dim3 threadsPerBlock(1024);
    dim3 Blocks((rowPtrSizeC + 1024 - 1) / 1024);

    CSR_to_COO_kernel<<<Blocks, threadsPerBlock>>>(
        numElementsC,
        rowPtrSizeC,
        matrixC_GPU_row_indices,
        matrixC_GPU_row_indices_complete);

    CUDA_CHECK(cudaGetLastError());

    // Call naive_coo_SDMM_kernel
    compute_naive_coo(
        m,
        n,
        k,
        numElementsC,
        matrixA_GPU_values,
        matrixB_transposed_GPU_values,
        matrixC_GPU_values,
        matrixC_GPU_row_indices_complete,
        matrixC_GPU_col_indices,
        matrixResult_GPU_values);

    CUDA_CHECK(cudaFree(matrixC_GPU_row_indices_complete));
    matrixC_GPU_row_indices_complete = nullptr;
    return;
}