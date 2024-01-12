// naive_SDDMM_GPU.cpp
#include "memory_test/test_hub.hpp"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "memory_test/test.cuh"
#include "memory_test/test_shared.cuh"
#include "utils.h"

void test_hub_GPU<float>::SDDMM_DENSE(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_transpose_HOST,
    const DenseMatrix<float>& matrixC_HOST,
    DenseMatrix<float>& matrixResult_dense_HOST) const
{
    // start the profiler
    // CUDA_CHECK(cudaProfilerStart());

    // get sizes of matrixA and matrixB {A=mxk; B=kxn; B_transpose=nxk}
    int m = matrixA_HOST.getNumRows();
    int k = matrixA_HOST.getNumCols();
    int n = matrixB_transpose_HOST.getNumRows();

    // check the dimensions of the matrices
    assert(matrixB_transpose_HOST.getNumCols() == k && "Error: matrixB_transpose has incompatible dimensions");
    assert(matrixA_HOST.getNumCols() == matrixB_transpose_HOST.getNumRows() && "Error: matrixA and matrixB_transpose have incompatible dimensions");
    assert(matrixC_HOST.getNumRows() == m && "Error: matrixC has incompatible dimensions m");
    assert(matrixC_HOST.getNumCols() == n && "Error: matrixC has incompatible dimensions n");
    assert(matrixResult_dense_HOST.getNumRows() == m && "Error: matrixResult has incompatible dimensions m");
    assert(matrixResult_dense_HOST.getNumCols() == n && "Error: matrixResult has incompatible dimensions n");

    // allocate memory for the matrices on the GPU
    float* matrixA_GPU;
    float* matrixB_GPU;
    float* matrixC_GPU;
    float* matrixResult_GPU;
    CUDA_CHECK(
        cudaMalloc(
            &matrixA_GPU,
            m * k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixB_GPU,
            n * k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixC_GPU,
            m * k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixResult_GPU,
            m * k * sizeof(float)));

    // copy matrices to the GPU
    CUDA_CHECK(
        cudaMemcpy(
            matrixA_GPU,
            matrixA_HOST.getValues(),
            m * k * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(
            matrixB_GPU,
            matrixB_transpose_HOST.getValues(),
            n * k * sizeof(float),
            cudaMemcpyHostToDevice));

    for (int i = 0; i < 10; i++)
    {
        // start the timer
        this->start_run();

        testing_shared(
            matrixB_GPU,
            matrixC_GPU);

        // stop the timer
        this->stop_run();

        // start the timer
        this->start_run();

        testing(
            matrixA_GPU,
            matrixResult_GPU);

        // stop the timer
        this->stop_run();
    }

    // free memory on the device and destroy the handle
    CUDA_CHECK(
        cudaFree(
            matrixA_GPU));
    CUDA_CHECK(
        cudaFree(
            matrixB_GPU));
    CUDA_CHECK(
        cudaFree(
            matrixC_GPU));
    CUDA_CHECK(
        cudaFree(
            matrixResult_GPU));

    // stop the profiler
    // CUDA_CHECK(cudaProfilerStop());

    return;
}

void test_hub_GPU<float>::SDDMM_CSR(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const CSRMatrix<float>& matrixC_HOST,
    CSRMatrix<float>& matrixResult_HOST) const
{
    // change matrixB_transpose and matrixResult to a dense matrix
    const DenseMatrix<float> matrixC_dense_HOST = DenseMatrix<float>(matrixC_HOST);
    DenseMatrix<float> matrixResult_dense_HOST = DenseMatrix<float>(matrixResult_HOST);

    // transpose matrixB to B^t
    DenseMatrix<float> matrixB_transpose_HOST = DenseMatrix<float>(matrixB_HOST);
    matrixB_transpose_HOST.transpose();

    // call naive_SDDMM_GPU to compute the SDDMM
    SDDMM_DENSE(
        matrixA_HOST,
        matrixB_transpose_HOST,
        matrixC_dense_HOST,
        matrixResult_dense_HOST);

    // change matrixResult to a sparse matrix
    CSRMatrix<float> matrixResult_finished_HOST(
        matrixResult_dense_HOST);

    // set the values of matrixResult_HOST to the values of matrixResult_finished_HOST
    matrixResult_HOST.setValues(
        matrixResult_finished_HOST.getValues());
    matrixResult_HOST.setColIndices(
        matrixResult_finished_HOST.getColIndices());
    matrixResult_HOST.setRowArray(
        matrixResult_finished_HOST.getRowArray());

    return;
}

void test_hub_GPU<float>::SDDMM(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const SparseMatrix<float>& matrixC_HOST,
    SparseMatrix<float>& matrixResult_HOST,
    const int num_iterations) const
{
    const CSRMatrix<float>* csrMatrixC = dynamic_cast<const CSRMatrix<float>*>(&matrixC_HOST);
    CSRMatrix<float>* csrMatrixResult = dynamic_cast<CSRMatrix<float>*>(&matrixResult_HOST);
    if (csrMatrixC == nullptr || csrMatrixResult == nullptr)
    {
        throw std::invalid_argument("Error: convert Sparse to CSR before using this function");
    }
    else
    {
        SDDMM_CSR(
            matrixA_HOST,
            matrixB_HOST,
            *csrMatrixC,
            *csrMatrixResult);
    }
}

template <typename T>
void test_hub_GPU<T>::SDDMM(
    const DenseMatrix<T>& matrixA_HOST,
    const DenseMatrix<T>& matrixB_HOST,
    const SparseMatrix<T>& matrixC_HOST,
    SparseMatrix<T>& matrixResult_HOST,
    const int num_iterations) const
{
    assert(false && "Error: test_hub_GPU::SDDMM() only accepts float as input. Other types are not supported");
}

void test_hub_GPU<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: test_hub_GPU::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_gpu_run();
}

void test_hub_GPU<float>::stop_run() const
{
    this->_timer->stop_gpu_run();
}

// Explicit template instantiation
// template class test_hub_GPU<float>;
template class test_hub_GPU<double>;
template class test_hub_GPU<int>;