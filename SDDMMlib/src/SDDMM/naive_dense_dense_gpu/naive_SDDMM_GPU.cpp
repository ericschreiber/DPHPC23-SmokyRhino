// naive_SDDMM_GPU.cpp
#include "naive_dense_dense_gpu/naive_SDDMM_GPU.hpp"

#include <iostream>

#include "naive_dense_dense_gpu/naive_dense_dense.cuh"
#include "utils.h"

template <typename T>
void naive_SDDMM_GPU<T>::SDDMM_DENSE(
    const DenseMatrix<T>& matrixA_HOST,
    const DenseMatrix<T>& matrixB_transpose_HOST,
    const DenseMatrix<T>& matrixC_HOST,
    DenseMatrix<T>& matrixResult_dense_HOST) const
{
    // get sizes of matrixA and matrixB {A=mxk; B=kxn; B_transpose=nxk}
    int m = matrixA_HOST.getNumRows();
    int n = matrixA_HOST.getNumCols();
    int k = matrixB_transpose_HOST.getNumCols();

    // allocate memory for the matrices on the GPU
    float* matrixA_GPU;
    float* matrixB_transpose_GPU;
    float* matrixC_GPU;
    float* matrixResult_GPU;
    CUDA_CHECK(
        cudaMalloc(
            &matrixA_GPU,
            m * k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixB_transpose_GPU,
            n * k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixC_GPU,
            m * n * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixResult_GPU,
            m * n * sizeof(float)));

    // copy matrices to the GPU
    CUDA_CHECK(
        cudaMemcpy(
            matrixA_GPU,
            matrixA_HOST,
            m * k * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(
            matrixB_transpose_GPU,
            matrixB_transpose_HOST,
            n * k * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(
            matrixC_GPU,
            matrixC_HOST,
            m * n * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(
            matrixResult_GPU,
            matrixResult_dense_HOST,
            m * n * sizeof(float),
            cudaMemcpyHostToDevice));

    // call compute in naive_dense_dense.cu
    compute(
        m,
        n,
        k,
        matrixA_GPU,
        matrixB_transpose_GPU,
        matrixC_GPU,
        matrixResult_GPU);

    // copy result from the GPU
    CUDA_CHECK(
        cudaMemcpy(
            matrixResult_dense_HOST,
            matrixResult_GPU,
            m * n * sizeof(float),
            cudaMemcpyDeviceToHost));

    // free memory on the device
    CUDA_CHECK(
        cudaFree(
            matrixA_GPU));
    CUDA_CHECK(
        cudaFree(
            matrixB_transpose_GPU));
    CUDA_CHECK(
        cudaFree(
            matrixC_GPU));
    CUDA_CHECK(
        cudaFree(
            matrixResult_GPU));

    std::cout << "naive_SDDMM was executed :)" << std::endl;
    return;
}

template <typename T>
void naive_SDDMM_GPU<T>::SDDMM(
    const DenseMatrix<T>& matrixA_HOST,
    const DenseMatrix<T>& matrixB_HOST,
    const SparseMatrix<T>& matrixC_HOST,
    SparseMatrix<T>& matrixResult_HOST) const
{
    // change matrixB_transpose and matrixResult to a dense matrix
    DenseMatrix<float> matrixC_dense_HOST(matrixC_HOST);
    DenseMatrix<float> matrixResult_dense_HOST(matrixResult_HOST);

    // transpose matrixB to B^t
    DenseMatrix<float> matrixB_transpose_HOST();
    matrixB_transpose_HOST = matrixB_HOST.transpose();

    // call naive_SDDMM_GPU to compute the SDDMM
    SDDMM_DENSE(
        matrixA_HOST,
        matrixB_transpose_HOST,
        matrixC_dense_HOST,
        matrixResult_dense_HOST);

    // change matrixResult to a sparse matrix
    CSRMatrix<float> matrixResult_finished_HOST(matrixResult_dense_HOST);
    matrixResult_HOST = matrixResult_finished_HOST;

    std::cout
        << "naive_SDDMM was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class naive_SDDMM_GPU<float>;
template class naive_SDDMM_GPU<double>;
template class naive_SDDMM_GPU<int>;