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
    // alpha and beta are needed for Sgemm
    float alpha = 1;
    float beta = 0;

    // to run cublas functions we need to first create a handle
    cublasHandle_t handle;
    CUDA_CHECK(cublasCreate(&handle));

    // Sgemm calculates A*B and the result is stored in C
    cublasSgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        n,
        k,
        &alpha,
        d_A,
        m,
        d_B,
        k,
        &beta,
        d_C,
        m);

    // custom kernel to calculate the Hadamard product between C and D
    // the result is stored in D
    my_naive_sampling(
        m * n,
        d_C,
        d_D);
}