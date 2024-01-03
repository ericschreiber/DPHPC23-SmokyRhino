#include "cusparse_baseline/cusparse_baseline.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>

#include <CSRMatrix.hpp>
#include <cmath>
#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "semi_naive_CSR_GPU/semi_naive_CSR_SDDMM_GPU.cuh"
#include "utils.h"

void cusparse_baseline<float>::SDDMM_CSR(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const CSRMatrix<float>& matrixC_HOST_CSR,
    CSRMatrix<float>& matrixResult_HOST,
    const int num_iterations) const
{
    // Print Matrix A
    std::cout << "Matrix A: " << std::endl;
    for (int i = 0; i < matrixA_HOST.getNumRows(); ++i)
    {
        for (int j = 0; j < matrixA_HOST.getNumCols(); ++j)
        {
            std::cout << matrixA_HOST.getValues()[i * matrixA_HOST.getNumCols() + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    // Print Matrix B
    std::cout << "Matrix B: " << std::endl;
    for (int i = 0; i < matrixB_HOST.getNumRows(); ++i)
    {
        for (int j = 0; j < matrixB_HOST.getNumCols(); ++j)
        {
            std::cout << matrixB_HOST.getValues()[i * matrixB_HOST.getNumCols() + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    CSRMatrix<float> test_input = CSRMatrix<float>(matrixC_HOST_CSR.getNumRows(), matrixC_HOST_CSR.getNumCols());
    ExecutionTimer timer = ExecutionTimer();
    semi_naive_CSR_SDDMM_GPU<float>* class_to_run = new semi_naive_CSR_SDDMM_GPU<float>(&timer);

    matrixC_HOST_CSR.SDDMM(
        matrixA_HOST,
        matrixB_HOST,
        test_input,
        1,
        std::bind(
            &semi_naive_CSR_SDDMM_GPU<float>::SDDMM,
            class_to_run,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3,
            std::placeholders::_4,
            std::placeholders::_5));
    delete class_to_run;
    class_to_run = nullptr;

    // Print Test Input
    std::cout << "Semi Naive Test Input colIndices: ";
    for (int i = 0; i < test_input.getColIndices().size(); ++i)
    {
        std::cout << test_input.getColIndices().at(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Semi Naive Test Input rowArray: ";
    for (int i = 0; i < test_input.getRowArray().size(); ++i)
    {
        std::cout << test_input.getRowArray().at(i) << " ";
    }
    std::cout << std::endl;
    std::cout << "Semi Naive Test Input values: ";
    for (int i = 0; i < test_input.getValues().size(); ++i)
    {
        std::cout << test_input.getValues().at(i) << " ";
    }
    std::cout << std::endl;

    if (num_iterations == 2)
    {
        CSRMatrix<float> matrixC_HOST_CSR_new = CSRMatrix<float>(matrixC_HOST_CSR);
        // // Add +1 for each col index
        // std::vector<int> colIndices = matrixC_HOST_CSR_new.getColIndices();
        // for (int i = 0; i < colIndices.size(); ++i)
        // {
        //     colIndices[i] += 1;
        // }
        // matrixC_HOST_CSR_new.setColIndices(colIndices);
        // // Move the row array by 1
        // std::vector<int> rowArray = matrixC_HOST_CSR_new.getRowArray();
        // int start = rowArray[0];
        // for (int i = 0; i < rowArray.size(); ++i)
        // {
        //     if (rowArray[i] != start)
        //     {
        //         int temp = rowArray[i];
        //         rowArray[i] = start;
        //         start = temp;
        //     }
        // }
        // matrixC_HOST_CSR_new.setRowArray(rowArray);

        CSRMatrix<float> test_input_cusparse = CSRMatrix<float>(matrixC_HOST_CSR.getNumRows(), matrixC_HOST_CSR.getNumCols());
        ExecutionTimer timer2 = ExecutionTimer();
        cusparse_baseline<float>* class_to_run_2 = new cusparse_baseline<float>(&timer2);

        matrixC_HOST_CSR_new.SDDMM(
            matrixA_HOST,
            matrixB_HOST,
            test_input_cusparse,
            1,
            std::bind(
                &cusparse_baseline<float>::SDDMM,
                class_to_run_2,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3,
                std::placeholders::_4,
                std::placeholders::_5));
        delete class_to_run_2;
        class_to_run_2 = nullptr;

        // Print test_input_cusparse
        std::cout << "Test Input cusparse colIndices: ";
        for (int i = 0; i < test_input_cusparse.getColIndices().size(); ++i)
        {
            std::cout << test_input_cusparse.getColIndices().at(i) << " ";
        }
        std::cout << std::endl;
        std::cout << "Test Input cusparse rowArray: ";
        for (int i = 0; i < test_input_cusparse.getRowArray().size(); ++i)
        {
            std::cout << test_input_cusparse.getRowArray().at(i) << " ";
        }
        std::cout << std::endl;
        std::cout << "Test Input cusparse values: ";
        for (int i = 0; i < test_input_cusparse.getValues().size(); ++i)
        {
            std::cout << test_input_cusparse.getValues().at(i) << " ";
        }
        std::cout << std::endl;
    }
    // Get all the sizes (A=mxk; B=kxn; C=mxn; Result=mxn)
    int m = matrixA_HOST.getNumRows();
    int k = matrixA_HOST.getNumCols();
    int n = matrixB_HOST.getNumCols();
    int numElementsC = matrixC_HOST_CSR.getValues().size();

    // check the dimensions of the matrices s.t. we can multiply them
    assert(matrixB_HOST.getNumRows() == k && "Error: matrixB has incompatible dimensions");
    assert(matrixC_HOST_CSR.getNumRows() == m && "Error: matrixC has incompatible dimensions m");
    assert(matrixC_HOST_CSR.getNumCols() == n && "Error: matrixC has incompatible dimensions n");
    assert(matrixResult_HOST.getNumRows() == m && "Error: matrixResult has incompatible dimensions m");
    assert(matrixResult_HOST.getNumCols() == n && "Error: matrixResult has incompatible dimensions n");

    // allocate memory for the matrices on the GPU
    float* matrixA_GPU_values;
    float* matrixB_GPU_values;
    int* matrixResult_GPU_row_indices;
    int* matrixResult_GPU_col_indices;
    int* prevBlocksWork;
    int* tiles_sizes;

    CUDA_CHECK(cudaMalloc(&matrixA_GPU_values, m * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixB_GPU_values, n * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixResult_GPU_row_indices, numElementsC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixResult_GPU_col_indices, numElementsC * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&prevBlocksWork, (m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&tiles_sizes, ceil(k * sizeof(float) / (float)(49152)) * sizeof(int)));

    // copy matrices to the GPU
    CUDA_CHECK(cudaMemcpy(matrixA_GPU_values, matrixA_HOST.getValues(), m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixB_GPU_values, matrixB_HOST.getValues(), n * k * sizeof(float), cudaMemcpyHostToDevice));

    int colIndicesSize = matrixC_HOST_CSR.getColIndices().size();
    int rowArraySize = matrixC_HOST_CSR.getRowArray().size();
    int valuesSize = matrixC_HOST_CSR.getValues().size();
    // alloc GPU memory for the copies
    int* colIndicesCopy_GPU;
    int* rowArrayCopy_GPU;
    float* valuesCopy_GPU;
    CUDA_CHECK(cudaMalloc(&colIndicesCopy_GPU, colIndicesSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&rowArrayCopy_GPU, rowArraySize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&valuesCopy_GPU, valuesSize * sizeof(float)));
    // copy the copies to the GPU
    CUDA_CHECK(cudaMemcpy(colIndicesCopy_GPU, (matrixC_HOST_CSR.getColIndices()).data(), colIndicesSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rowArrayCopy_GPU, (matrixC_HOST_CSR.getRowArray()).data(), rowArraySize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(valuesCopy_GPU, (matrixC_HOST_CSR.getValues()).data(), valuesSize * sizeof(float), cudaMemcpyHostToDevice));

    // create the cuda sparse matrix descriptors
    cusparseDnMatDescr_t matrixA_desc;
    cusparseDnMatDescr_t matrixB_desc;
    cusparseCreateDnMat(&matrixA_desc, m, k, k, matrixA_GPU_values, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseCreateDnMat(&matrixB_desc, k, n, n, matrixB_GPU_values, CUDA_R_32F, CUSPARSE_ORDER_ROW);
    cusparseSpMatDescr_t matrixC_desc;
    cusparseCreateCsr(
        &matrixC_desc,
        m,
        n,
        valuesSize,
        rowArrayCopy_GPU,
        colIndicesCopy_GPU,
        valuesCopy_GPU,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F);

    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        std::cerr << "CUSPARSE Library initialization failed" << std::endl;
    }
    float alpha_value = 1.0f;
    float beta_value = 0.0f;
    void* externalBuffer = 0;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSDDMMAlg_t alg = CUSPARSE_SDDMM_ALG_DEFAULT;

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
    cudaMalloc(&externalBuffer, bufferSize);

    for (int i = 0; i < num_iterations; i++)
    {
        // // Preprocessing not needed and not timed so not included
        // cusparseSDDMM_preprocess(
        //     handle,
        //     opA,
        //     opB,
        //     &alpha_value,
        //     matrixA_desc,
        //     matrixB_desc,
        //     &beta_value,
        //     matrixC_desc,
        //     CUDA_R_32F,
        //     alg,
        //     externalBuffer);

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
    }

    float* matrixResult_HOST_values = new float[numElementsC];
    CUDA_CHECK(cudaMemcpy(matrixResult_HOST_values, valuesCopy_GPU, numElementsC * sizeof(float), cudaMemcpyDeviceToHost));
    matrixResult_HOST.setValues(std::vector<float>(matrixResult_HOST_values, matrixResult_HOST_values + numElementsC));
    delete[] matrixResult_HOST_values;
    matrixResult_HOST_values = nullptr;

    // We actually keep the same row and col indices
    matrixResult_HOST.setColIndices(matrixC_HOST_CSR.getColIndices());
    matrixResult_HOST.setRowArray(matrixC_HOST_CSR.getRowArray());

    // free memory
    CUDA_CHECK(cudaFree(matrixA_GPU_values));
    CUDA_CHECK(cudaFree(matrixB_GPU_values));
    CUDA_CHECK(cudaFree(matrixResult_GPU_row_indices));
    CUDA_CHECK(cudaFree(matrixResult_GPU_col_indices));
    CUDA_CHECK(cudaFree(prevBlocksWork));
    CUDA_CHECK(cudaFree(tiles_sizes));

    matrixA_GPU_values = nullptr;
    matrixB_GPU_values = nullptr;
    matrixResult_GPU_row_indices = nullptr;
    matrixResult_GPU_col_indices = nullptr;
    prevBlocksWork = nullptr;
    tiles_sizes = nullptr;

    cusparseDestroyDnMat(matrixA_desc);
    cusparseDestroyDnMat(matrixB_desc);
    cusparseDestroySpMat(matrixC_desc);
    cusparseDestroy(handle);
    cudaFree(externalBuffer);

    return;
}

void cusparse_baseline<float>::SDDMM(
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

    return;
}

void cusparse_baseline<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: cusparse_baseline::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_gpu_run();
}

void cusparse_baseline<float>::stop_run() const
{
    this->_timer->stop_gpu_run();
}

// Explicit template instantiation
template class cusparse_baseline<double>;
template class cusparse_baseline<int>;