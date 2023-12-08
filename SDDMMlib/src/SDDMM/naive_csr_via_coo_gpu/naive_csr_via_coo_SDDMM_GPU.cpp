// naive_csr_SDDMM_GPU.cpp
#include "naive_csr_via_coo_gpu/naive_csr_via_coo_SDDMM_GPU.hpp"

#include <cassert>
#include <iostream>

#include "naive_csr_via_coo_gpu/naive_csr_via_coo_SDDMM.cuh"
#include "utils.h"

void naive_csr_via_coo_SDDMM_GPU<float>::SDDMM(
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
        std::cout << "Starting computation of naive_csr_via_coo_SDDMM_GPU" << std::endl;
        SDDMM_CSR(
            matrixA_HOST,
            matrixB_HOST,
            *csrMatrixC,
            *csrMatrixResult,
            num_iterations);
    }

    csrMatrixC = nullptr;
    csrMatrixResult = nullptr;

    return;
}

void naive_csr_via_coo_SDDMM_GPU<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: naive_csr_SDDMM_GPU::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_cpu_run();
}

void naive_csr_via_coo_SDDMM_GPU<float>::stop_run() const
{
    this->_timer->stop_cpu_run();
}

// void convert_csr_to_coo(const std::vector<int>& rowPtr_csr, std::vector<int>& rowIndices_coo)
// {
//     for (int row = 0; row < rowPtr_csr.size(); ++row)
//     {
//         int start = rowPtr_csr[row];
//         int end = rowPtr_csr[row + 1];

//         for (int i = start; i < end; ++i)
//         {
//             rowIndices_coo.push_back(row);  // Row index
//         }
//     }
// }

void naive_csr_via_coo_SDDMM_GPU<float>::SDDMM_CSR(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const CSRMatrix<float>& matrixC_HOST,
    CSRMatrix<float>& matrixResult_HOST,
    const int num_iterations) const
{
    // Get all the sizes (A=mxk; B=kxn; C=mxn; Result=mxn)
    int m = matrixA_HOST.getNumRows();
    int k = matrixA_HOST.getNumCols();
    int n = matrixB_HOST.getNumCols();
    int numElementsC = matrixC_HOST.getValues().size();
    int rowPtrSizeC = matrixC_HOST.getRowArray().size();

    // check the dimensions of the matrices s.t. we can multiply them
    assert(matrixB_HOST.getNumRows() == k && "Error: matrixB has incompatible dimensions");
    assert(matrixC_HOST.getNumRows() == m && "Error: matrixC has incompatible dimensions m");
    assert(matrixC_HOST.getNumCols() == n && "Error: matrixC has incompatible dimensions n");
    assert(matrixResult_HOST.getNumRows() == m && "Error: matrixResult has incompatible dimensions m");
    assert(matrixResult_HOST.getNumCols() == n && "Error: matrixResult has incompatible dimensions n");

    DenseMatrix<float> matrixBTranspose_HOST = DenseMatrix<float>(matrixB_HOST);
    matrixBTranspose_HOST.transpose();

    // Convert matrixC to COO format for that change the rowPtr to rowIndices
    // std::vector<int> matrixC_row_indices = std::vector<int>(numElementsC);
    // convert_csr_to_coo(matrixC_HOST.getRowArray(), matrixC_row_indices);

    // allocate memory for the matrices on the GPU
    float* matrixA_GPU_values;
    float* matrixB_transpose_GPU_values;
    float* matrixC_GPU_values;
    int* matrixC_GPU_row_indices;
    // int* matrixC_GPU_row_indices_complete;
    int* matrixC_GPU_col_indices;
    float* matrixResult_GPU_values;

    CUDA_CHECK(cudaMalloc(&matrixA_GPU_values, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixB_transpose_GPU_values, n * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixC_GPU_values, numElementsC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixC_GPU_row_indices, rowPtrSizeC * sizeof(float)));
    // CUDA_CHECK(cudaMalloc(&matrixC_GPU_row_indices_complete, (numElementsC + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&matrixC_GPU_col_indices, numElementsC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixResult_GPU_values, numElementsC * sizeof(float)));

    // copy matrices to the GPU
    CUDA_CHECK(cudaMemcpy(matrixA_GPU_values, matrixA_HOST.getValues(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixB_transpose_GPU_values, matrixBTranspose_HOST.getValues(), n * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixC_GPU_values, (matrixC_HOST.getValues()).data(), numElementsC * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixC_GPU_row_indices, (matrixC_HOST.getRowArray()).data(), rowPtrSizeC * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(matrixC_GPU_row_indices_complete, matrixC_row_indices.data(), numElementsC * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixC_GPU_col_indices, (matrixC_HOST.getColIndices()).data(), numElementsC * sizeof(float), cudaMemcpyHostToDevice));

    for (int i = 0; i < num_iterations; i++)
    {
        this->start_run();
        compute_naive_csr_via_coo(
            m,
            n,
            k,
            numElementsC,
            rowPtrSizeC,
            matrixA_GPU_values,
            matrixB_transpose_GPU_values,
            matrixC_GPU_values,
            matrixC_GPU_row_indices,
            // matrixC_GPU_row_indices_complete,
            matrixC_GPU_col_indices,
            matrixResult_GPU_values);
        this->stop_run();
    }

    // copy matrixResult_GPU to matrixResult
    float* matrixResult_HOST_values = new float[numElementsC];
    CUDA_CHECK(cudaMemcpy(matrixResult_HOST_values, matrixResult_GPU_values, numElementsC * sizeof(float), cudaMemcpyDeviceToHost));
    matrixResult_HOST.setValues(std::vector<float>(matrixResult_HOST_values, matrixResult_HOST_values + numElementsC));
    delete[] matrixResult_HOST_values;
    matrixResult_HOST_values = nullptr;

    // We actually keep the same row and col indices
    matrixResult_HOST.setColIndices(matrixC_HOST.getColIndices());
    matrixResult_HOST.setRowArray(matrixC_HOST.getRowArray());

    // free memory
    CUDA_CHECK(cudaFree(matrixA_GPU_values));
    CUDA_CHECK(cudaFree(matrixB_transpose_GPU_values));
    CUDA_CHECK(cudaFree(matrixC_GPU_values));
    CUDA_CHECK(cudaFree(matrixC_GPU_row_indices));
    // CUDA_CHECK(cudaFree(matrixC_GPU_row_indices_complete));
    CUDA_CHECK(cudaFree(matrixC_GPU_col_indices));
    CUDA_CHECK(cudaFree(matrixResult_GPU_values));

    matrixA_GPU_values = nullptr;
    matrixB_transpose_GPU_values = nullptr;
    matrixC_GPU_values = nullptr;
    matrixC_GPU_row_indices = nullptr;
    // matrixC_GPU_row_indices_complete = nullptr;
    matrixC_GPU_col_indices = nullptr;
    matrixResult_GPU_values = nullptr;

    return;
}