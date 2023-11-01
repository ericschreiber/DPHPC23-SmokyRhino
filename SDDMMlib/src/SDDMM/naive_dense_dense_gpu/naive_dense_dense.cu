#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "utils.h"

void my_naive_sampling(
    int,
    const float *,
    float *);

////
// this function computes the SDDMM of 3 dense matrices using cublas Sgemm and a custom sampling kernel
// A (m x k)
// B (k x n)
// C (m x n)
// D (m x n)
//
// the function expects the sizes m, n, k as int and pointers to the 4 matrices of type float
//                                                         in the following order A, B, C, D
// all pointers need to point to memory on the GPU
// the matrix C is needed as a buffer - it does not contribute data to the SDDMM
//
// the result is written to D
////

void compute(
    int m,
    int n,
    int k,
    float *d_A,
    float *d_B,
    float *d_C,
    float *d_D)
{
    CUDA_CHECK(cudaGetLastError());
    // alpha and beta are needed for Sgemm
    float alpha = 1;
    float beta = 0;

    // to run cublas functions we need to first create a handle
    cublasHandle_t handle;
    CUDA_CHECK(cublasCreate(&handle));

    // // Load d_A to host and print it
    // float *h_A = new float[m * k];
    // CUDA_CHECK(cudaMemcpy(h_A, d_A, m * k * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "A = " << std::endl;
    // for (int i = 0; i < m * k; i++)
    // {
    //     std::cout << h_A[i] << " ";
    //     if ((i + 1) % k == 0)
    //     {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    // // Load d_B to host and print it
    // float *h_B = new float[k * n];
    // CUDA_CHECK(cudaMemcpy(h_B, d_B, k * n * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "B = " << std::endl;
    // for (int i = 0; i < k * n; i++)
    // {
    //     std::cout << h_B[i] << " ";
    //     if ((i + 1) % n == 0)
    //     {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    // Sgemm calculates A*B and the result is stored in C
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        m,
        n,
        k,
        &alpha,
        d_A,
        m,
        d_B,
        k,
        &beta,
        d_D,
        m);

    // // Copy Matric C to Host and print it
    // float *h_C = new float[m * n];
    // CUDA_CHECK(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "C = " << std::endl;
    // for (int i = 0; i < m * n; i++)
    // {
    //     std::cout << h_C[i] << " ";
    //     if ((i + 1) % n == 0)
    //     {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    // // copy matrix D to host and print it
    // float *h_D = new float[m * n];
    // CUDA_CHECK(cudaMemcpy(h_D, d_D, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "D = " << std::endl;
    // for (int i = 0; i < m * n; i++)
    // {
    //     std::cout << h_D[i] << " ";
    //     if ((i + 1) % n == 0)
    //     {
    //         std::cout << std::endl;
    //     }
    // }
    // std::cout << std::endl;

    // custom kernel to calculate the Hadamard product between C and D
    // the result is stored in D
    my_naive_sampling(
        m * n,
        d_C,
        d_D);

    // // Copy Matric D to Host and print it
    // h_D = new float[m * n];
    // CUDA_CHECK(cudaMemcpy(h_D, d_D, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    // std::cout << "D_final = " << std::endl;
    // for (int i = 0; i < m * n; i++)
    // {
    //     std::cout << h_D[i] << " ";
    //     if ((i + 1) % n == 0)
    //     {
    //         std::cout << std::endl;
    //     }
    // }
}