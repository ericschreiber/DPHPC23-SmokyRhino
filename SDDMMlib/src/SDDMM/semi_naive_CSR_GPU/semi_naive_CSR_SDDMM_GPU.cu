// semi_naive_CSR_SDDMM_GPU.cpp

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <semi_naive_CSR_GPU/semi_naive_CSR_SDDMM_GPU.cuh>
#include <semi_naive_CSR_GPU/semi_naive_CSR_SDDMM_GPU_kernel.cuh>
#include <type_traits>
#include <typeinfo>

#include "utils.h"

void semi_naive_CSR_SDDMM_GPU<float>::SDDMM_CSR(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const CSRMatrix<float>& matrixC_HOST,
    CSRMatrix<float>& matrixResult_sparse_HOST,
    const int num_iterations) const
{
    // start the profiler
    // CUDA_CHECK(cudaProfilerStart());
    // transpose matrixB to B^t
    DenseMatrix<float> matrixB_transpose_HOST = DenseMatrix<float>(matrixB_HOST);
    matrixB_transpose_HOST.transpose();

    // get sizes of matrixA and matrixB {A=mxk; B=kxn; B_transpose=nxk}
    int m = matrixA_HOST.getNumRows();
    int k = matrixA_HOST.getNumCols();
    int n = matrixB_transpose_HOST.getNumRows();
    int nnz = matrixC_HOST.getNumValues();

    // check the dimensions of the matrices
    assert(matrixB_transpose_HOST.getNumCols() == k && "Error: matrixB_transpose has incompatible dimensions");
    assert(matrixC_HOST.getNumRows() == m && "Error: matrixC has incompatible dimensions m");
    assert(matrixC_HOST.getNumCols() == n && "Error: matrixC has incompatible dimensions n");
    assert(matrixResult_sparse_HOST.getNumRows() == m && "Error: matrixResult has incompatible dimensions m");
    assert(matrixResult_sparse_HOST.getNumCols() == n && "Error: matrixResult has incompatible dimensions n");

    // allocate memory for the matrices on the GPU
    float* matrixA_GPU;
    float* matrixB_transpose_GPU;
    float* matrixResult_GPU;
    int* col_idx_GPU;
    int* row_ptr_GPU;
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
            &matrixResult_GPU,
            nnz * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &col_idx_GPU,
            nnz * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &row_ptr_GPU,
            (m + 1) * sizeof(int)));

    // copy matrices to the GPU
    CUDA_CHECK(
        cudaMemcpy(
            matrixA_GPU,
            matrixA_HOST.getValues(),
            m * k * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(
            matrixB_transpose_GPU,
            matrixB_transpose_HOST.getValues(),
            n * k * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(
            col_idx_GPU,
            (matrixC_HOST.getColIndices()).data(),
            nnz * sizeof(int),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(
            row_ptr_GPU,
            (matrixC_HOST.getRowArray()).data(),
            (m + 1) * sizeof(int),
            cudaMemcpyHostToDevice));

    for (int i = 0; i < num_iterations; i++)
    {
        // start the timer
        this->start_run();

        // Call the kernel to execute the acutal SDDMM
        compute_blockwise(
            m,
            n,
            k,
            matrixA_GPU,
            matrixB_transpose_GPU,
            row_ptr_GPU,
            col_idx_GPU,
            matrixResult_GPU);

        // stop the timer
        this->stop_run();
    }

    // copy result from the GPU to the CPU
    float* return_values = new float[nnz];
    CUDA_CHECK(
        cudaMemcpy(
            return_values,
            matrixResult_GPU,
            nnz * sizeof(float),
            cudaMemcpyDeviceToHost));

    // Convert pointer to std::vector
    std::vector<float> result_vector(return_values, return_values + nnz);

    // set the result matrix
    matrixResult_sparse_HOST.setValues(result_vector);
    matrixResult_sparse_HOST.setColIndices(matrixC_HOST.getColIndices());
    matrixResult_sparse_HOST.setRowArray(matrixC_HOST.getRowArray());

    // free memory on the device and destroy the handle
    CUDA_CHECK(
        cudaFree(
            matrixA_GPU));
    CUDA_CHECK(
        cudaFree(
            matrixB_transpose_GPU));
    CUDA_CHECK(
        cudaFree(
            matrixResult_GPU));
    CUDA_CHECK(
        cudaFree(
            col_idx_GPU));
    CUDA_CHECK(
        cudaFree(
            row_ptr_GPU));

    // stop the profiler
    // CUDA_CHECK(cudaProfilerStop());

    return;
}

void semi_naive_CSR_SDDMM_GPU<float>::SDDMM(
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
            *csrMatrixResult,
            num_iterations);
    }
}

template <typename T>
void semi_naive_CSR_SDDMM_GPU<T>::SDDMM(
    const DenseMatrix<T>& matrixA_HOST,
    const DenseMatrix<T>& matrixB_HOST,
    const SparseMatrix<T>& matrixC_HOST,
    SparseMatrix<T>& matrixResult_HOST,
    const int num_iterations) const
{
    assert(false && "Error: semi_naive_CSR_SDDMM_GPU::SDDMM() only accepts float as input. Other types are not supported");
}

void semi_naive_CSR_SDDMM_GPU<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: semi_naive_CSR_SDDMM_GPU::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_gpu_run();
}

void semi_naive_CSR_SDDMM_GPU<float>::stop_run() const
{
    this->_timer->stop_gpu_run();
}

// Explicit template instantiation
// template class semi_naive_CSR_SDDMM_GPU<float>;
template class semi_naive_CSR_SDDMM_GPU<double>;
template class semi_naive_CSR_SDDMM_GPU<int>;
