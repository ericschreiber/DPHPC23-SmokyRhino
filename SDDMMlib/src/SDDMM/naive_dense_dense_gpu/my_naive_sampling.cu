#include <cuda_runtime.h>

#include <algorithm>
#include <iostream>

#include "utils.h"

////
// this kernels computes the Hadamard Product of two dense matrices A and B
// A (m x n)
// B (m x n)
//
// the kernel expects the number of elements of the matrices as int (m*n) and
// pointers to the 2 matrices of type float in the following order A, B
// all pointers need to point to memory on the GPU
//
// the result is written to B
////

__global__ void naivesampling(
    const int size,
    const float *const A,
    float *const B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size)
    {
        B[idx] = B[idx] * A[idx];
    }
}

void my_naive_sampling(
    const int size,
    const float *const A,
    float *const B)
{
    // every block can have up to 1024 threads
    int blocks = std::min(1024, (size + 1023) / 1024);
    std::cout << "blocks: " << blocks << std::endl;
    std::cout << "size: " << size << std::endl;
    std::cout << "Pointer A: " << A << std::endl;
    std::cout << "Pointer B: " << B << std::endl;
    std::cout << "Pointer to size: " << &size << std::endl;

    // Copy A to Host
    float *A_host = new float[size];
    CUDA_CHECK(cudaMemcpy(A_host, A, size * sizeof(float), cudaMemcpyDeviceToHost));
    // Print A
    std::cout << "A: ";
    for (int i = 0; i < size; i++)
    {
        std::cout << A_host[i] << " ";
    }
    std::cout << std::endl;

    // Copy B to Host
    float *B_host = new float[size];
    CUDA_CHECK(cudaMemcpy(B_host, B, size * sizeof(float), cudaMemcpyDeviceToHost));
    // Print B
    std::cout << "B: ";
    for (int i = 0; i < size; i++)
    {
        std::cout << B_host[i] << " ";
    }
    std::cout << std::endl;

    // naivesampling<<<blocks, 1024>>>(
    //     size,
    //     A,
    //     B);
    CUDA_CHECK(cudaGetLastError());

    // synchronization not needed if implicit in the time measurement
    CUDA_CHECK(cudaDeviceSynchronize());
}