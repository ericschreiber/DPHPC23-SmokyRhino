#include "cusparse_baseline/cusparse_baseline.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <CSRMatrix.hpp>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "utils.h"

// TODO: remove the transposition of matrixB_HOST

void cusparse_baseline<float>::SDDMM_COO(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const COOMatrix<float>& matrixC_HOST,
    COOMatrix<float>& matrixResult_HOST,
    const int num_iterations) const
{
    for (int i = 0; i < num_iterations; i++)
    {
        // Get all the sizes (A=mxk; B=kxn; C=mxn; Result=mxn)
        int m = matrixA_HOST.getNumRows();
        int k = matrixA_HOST.getNumCols();
        int n = matrixB_HOST.getNumCols();
        int numElementsC = matrixC_HOST.getValues().size();

        // check the dimensions of the matrices s.t. we can multiply them
        assert(matrixB_HOST.getNumRows() == k && "Error: matrixB has incompatible dimensions");
        assert(matrixC_HOST.getNumRows() == m && "Error: matrixC has incompatible dimensions m");
        assert(matrixC_HOST.getNumCols() == n && "Error: matrixC has incompatible dimensions n");
        assert(matrixResult_HOST.getNumRows() == m && "Error: matrixResult has incompatible dimensions m");
        assert(matrixResult_HOST.getNumCols() == n && "Error: matrixResult has incompatible dimensions n");

        DenseMatrix<float> matrixBTranspose_HOST = DenseMatrix<float>(matrixB_HOST);
        matrixBTranspose_HOST.transpose();

        // allocate memory for the matrices on the GPU
        float* matrixA_GPU_values;
        float* matrixB_transpose_GPU_values;
        float* matrixC_GPU_values;
        int* matrixC_GPU_row_indices;
        int* matrixC_GPU_col_indices;
        float* matrixResult_GPU_values;
        int* matrixResult_GPU_row_indices;
        int* matrixResult_GPU_col_indices;
        int* prevBlocksWork;
        int* tiles_sizes;

        CUDA_CHECK(cudaMalloc(&matrixA_GPU_values, m * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&matrixB_transpose_GPU_values, n * k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&matrixC_GPU_values, numElementsC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&matrixC_GPU_row_indices, numElementsC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&matrixC_GPU_col_indices, numElementsC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&matrixResult_GPU_values, numElementsC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&matrixResult_GPU_row_indices, numElementsC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&matrixResult_GPU_col_indices, numElementsC * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&prevBlocksWork, (m + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&tiles_sizes, ceil(k * sizeof(float) / (float)(49152)) * sizeof(int)));

        // copy matrices to the GPU
        CUDA_CHECK(cudaMemcpy(matrixA_GPU_values, matrixA_HOST.getValues(), m * k * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(matrixB_transpose_GPU_values, matrixBTranspose_HOST.getValues(), n * k * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(matrixC_GPU_values, (matrixC_HOST.getValues()).data(), numElementsC * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(matrixC_GPU_row_indices, (matrixC_HOST.getRowArray()).data(), numElementsC * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(matrixC_GPU_col_indices, (matrixC_HOST.getColIndices()).data(), numElementsC * sizeof(float), cudaMemcpyHostToDevice));

        // zero out the result matrix
        CUDA_CHECK(cudaMemset(matrixResult_GPU_values, 0.0, numElementsC * sizeof(float)));

        /*
        cusparseStatus t
        cusparseCreateDnMat (
            cusparseDnMatDescr_†* dnMatDescr,
            size_t rows,
            size_t cols,
            int64_t ld,
            void* values,
            cudaDataType type,
            cusparseOrder_t order
        )
        */

        // convert matrixC_HOST to CSR format
        CSRMatrix<float> matrixC_HOST_CSR = CSRMatrix<float>(matrixC_HOST);
        std::vector<int> colIndicesCopy = matrixC_HOST_CSR.getColIndices();
        std::vector<int> rowArrayCopy = matrixC_HOST_CSR.getRowArray();
        std::vector<float> valuesCopy = matrixC_HOST_CSR.getValues();

        cusparseDnMatDescr_t matrixA_desc;
        cusparseDnMatDescr_t matrixB_desc;
        cusparseCreateDnMat(&matrixA_desc, m, k, k, matrixA_GPU_values, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseCreateDnMat(&matrixB_desc, n, k, k, matrixB_transpose_GPU_values, CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseSpMatDescr_t matrixC_desc;
        cusparseCreateCsr(
            &matrixC_desc,
            m,
            n,
            valuesCopy.size(),
            colIndicesCopy.data(),
            rowArrayCopy.data(),
            valuesCopy.data(),
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO,
            CUDA_R_32F);

        /*
        cusparseStatus_t
        cusparseSDDMM(
            cusparseHandle_t handle,
            cusparseOperation_t opA,
            cusparseOperation_t opB,
            const void* alpha,
            cusparseConstDnMatDescr_t matA,
            cusparseConstDnMatDescr_t matB,
            const void* beta,
            cusparseSpMatDescr_t matC,
            cudaDataType computeType,
            cusparseSDDMMAlg_t alg,
            void* externalBuffer
        )
        */

        cusparseHandle_t handle;
        cusparseStatus_t status = cusparseCreate(&handle);
        float alpha_value = 1.0f;
        float beta_value = 1.0f;
        void* externalBuffer = 0;
        cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
        cusparseOperation_t opB = CUSPARSE_OPERATION_TRANSPOSE;
        cusparseSDDMMAlg_t alg = CUSPARSE_SDDMM_ALG_DEFAULT;

        printf("1. WE ARE HERE NOW \n");

        // get the size of the external buffer
        size_t bufferSize;
        cusparseSDDMM_bufferSize(
            handle,
            opA,
            opB,
            &alpha_value,
            matrixA_desc,
            matrixB_desc,
            &beta_value,
            matrixC_desc,
            CUDA_R_32F,
            alg,
            &bufferSize);
        cudaMalloc((void**)&externalBuffer, bufferSize);

        printf("bufferSize: %d \n", bufferSize);

        printf("2. WE ARE HERE NOW \n");

        this->start_run();

        cusparseSDDMM(
            handle,
            opA,
            opB,
            &alpha_value,
            matrixA_desc,
            matrixB_desc,
            &beta_value,
            matrixC_desc,
            CUDA_R_32F,
            alg,
            externalBuffer);

        this->stop_run();

        printf("3. WE ARE HERE NOW \n");

        // get values of matricC_desc and write them into the return_matrix
        int64_t rows, cols, nnzC;
        cusparseSpMatGetSize(matrixC_desc, &rows, &cols, &nnzC);
        void* matrixC_HOST_values_void;
        cusparseSpMatGetValues(matrixC_desc, &matrixC_HOST_values_void);
        float* matrixC_HOST_values = static_cast<float*>(matrixC_HOST_values_void);
        matrixResult_HOST.setValues(std::vector<float>(matrixC_HOST_values, matrixC_HOST_values + nnzC));
        delete[] matrixC_HOST_values;
        matrixC_HOST_values = nullptr;

        // We actually keep the same row and col indices
        matrixResult_HOST.setColIndices(matrixC_HOST.getColIndices());
        matrixResult_HOST.setRowArray(matrixC_HOST.getRowArray());

        // free memory
        CUDA_CHECK(cudaFree(matrixA_GPU_values));
        CUDA_CHECK(cudaFree(matrixB_transpose_GPU_values));
        CUDA_CHECK(cudaFree(matrixC_GPU_values));
        CUDA_CHECK(cudaFree(matrixC_GPU_row_indices));
        CUDA_CHECK(cudaFree(matrixC_GPU_col_indices));
        CUDA_CHECK(cudaFree(matrixResult_GPU_values));
        CUDA_CHECK(cudaFree(matrixResult_GPU_row_indices));
        CUDA_CHECK(cudaFree(matrixResult_GPU_col_indices));
        CUDA_CHECK(cudaFree(prevBlocksWork));
        CUDA_CHECK(cudaFree(tiles_sizes));

        matrixA_GPU_values = nullptr;
        matrixB_transpose_GPU_values = nullptr;
        matrixC_GPU_values = nullptr;
        matrixC_GPU_row_indices = nullptr;
        matrixC_GPU_col_indices = nullptr;
        matrixResult_GPU_values = nullptr;
        matrixResult_GPU_row_indices = nullptr;
        matrixResult_GPU_col_indices = nullptr;
        prevBlocksWork = nullptr;
        tiles_sizes = nullptr;

        cusparseDestroyDnMat(matrixA_desc);
        cusparseDestroyDnMat(matrixB_desc);
        cusparseDestroySpMat(matrixC_desc);
        cusparseDestroy(handle);
        cudaFree(externalBuffer);
    }
    return;
}

void cusparse_baseline<float>::SDDMM(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const SparseMatrix<float>& matrixC_HOST,
    SparseMatrix<float>& matrixResult_HOST,
    const int num_iterations) const
{
    const COOMatrix<float>* cooMatrixC = dynamic_cast<const COOMatrix<float>*>(&matrixC_HOST);
    COOMatrix<float>* cooMatrixResult = dynamic_cast<COOMatrix<float>*>(&matrixResult_HOST);
    if (cooMatrixC == nullptr || cooMatrixResult == nullptr)
    {
        throw std::invalid_argument("Error: convert Sparse to COO before using this function");
    }
    else
    {
        SDDMM_COO(
            matrixA_HOST,
            matrixB_HOST,
            *cooMatrixC,
            *cooMatrixResult,
            num_iterations);
    }

    cooMatrixC = nullptr;
    cooMatrixResult = nullptr;

    return;
}

void cusparse_baseline<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: cusparse_baseline::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_cpu_run();
}

void cusparse_baseline<float>::stop_run() const
{
    this->_timer->stop_cpu_run();
}

// Explicit template instantiation
// template class cusparse_baseline<float>;
template class cusparse_baseline<double>;
template class cusparse_baseline<int>;