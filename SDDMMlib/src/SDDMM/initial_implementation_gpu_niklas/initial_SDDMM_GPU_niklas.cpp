// initial_SDDMM_GPU_niklas.cpp
#include "initial_implementation_gpu_niklas/initial_SDDMM_GPU_niklas.hpp"

#include <iostream>

#include "initial_implementation_gpu_niklas/initial_SDDMM_solver.cuh"

template <typename T>
void initial_SDDMM_GPU_niklas<T>::SDDMM(
    const DenseMatrix<T> &matrixA_HOST,
    const DenseMatrix<T> &matrixB_HOST,
    const SparseMatrix<T> &matrixC_HOST,
    SparseMatrix<T> &matrixResult_HOST) const
{
    /*
    // get values colIndices and rowPtr from matrixC
    std::vector<T> values_matrixC_HOST = matrixC_HOST.getValues();
    std::vector<int> colIndices_matrixC_HOST = matrixC_HOST.getColIndices();
    std::vector<int> rowPtr_matrixC_HOST = matrixC_HOST.getRowPtr();

    // copy colIndices and rowPtr from matrixC_HOST to matrixResult_HOST
    matrixResult_HOST.set_colIndices(colIndices_matrixC_HOST);  // function not implemented yet
    matrixResult_HOST.set_rowPtr(rowPtr_matrixC_HOST);          // function not implemented yet

    // transpose matrixB_HOST to matrixB_transpose_HOST

    // get sizes of matrixA and matrixB {A=mxk; B=kxn; B_transpose=nxk}
    int m = matrixA_HOST.getNumRows();
    int n = matrixA_HOST.getNumCols();
    int k = matrixB_transpose_HOST.getNumCols();

    // allocate memory for the matrices on the GPU
    float *matrixA_GPU;
    float *matrixB_transpose_GPU;
    float *values_matrixC_GPU;
    float *values_matrixResult_GPU;
    float *colIndices_matrixResult_GPU;
    float *rowPtr_matrixResult_GPU;
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
            &values_matrixC_GPU,
            values_matrixC_HOST.size() * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &values_matrixResult_GPU,
            values_matrixC_HOST.size() * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &colIndices_matrixResult_GPU,
            colIndices_matrixC_HOST.size() * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &rowPtr_matrixResult_GPU,
            rowPtr_matrixC_HOST.size() * sizeof(float)));

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
        cudaMalloc(
            values_matrixC_GPU,
            values_matrixC_HOST,
            values_matrixC_HOST.size() * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMalloc(
            values_matrixResult_GPU,
            values_matrixC_HOST,
            values_matrixC_HOST.size() * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMalloc(
            colIndices_matrixResult_GPU,
            colIndices_matrixC_HOST,
            colIndices_matrixC_HOST * sizeof(float),
            cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMalloc(
            rowPtr_matrixResult_GPU,
            rowPtr_matrixC_HOST,
            rowPtr_matrixC_HOST * sizeof(float),
            cudaMemcpyHostToDevice));

    // call compute in .cu
    compute(
        (rowPtr_matrixC_HOST.size() - 1),  // this will give the number of blocks to use
                                           // this is by no means optimized
                                           // maybe use a different metric according to
                                           // threads per SM
        m,
        n,
        k,
        matrixA_GPU,
        matrixB_GPU,
        values_matrixC_GPU,
        values_matrixResult_GPU,
        colIndices_matrixResult_GPU,
        rowPtr_matrixResult_GPU);

    // copy result from the GPU
    CUDA_CHECK(
        cudaMemcpy(
            values_matrixC_HOST,
            values_matrixResult_GPU,
            values_matrixC_HOST.size() * sizeof(float),
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
            values_matrixC_GPU));
    CUDA_CHECK(
        cudaFree(
            values_matrixResult_GPU));
    CUDA_CHECK(
        cudaFree(
            colIndices_matrixResult_GPU));
    CUDA_CHECK(
        cudaFree(
            rowPtr_matrixResult_GPU));
    */
    std::cout << "naive_SDDMM from niklas was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class initial_SDDMM_GPU_niklas<float>;
template class initial_SDDMM_GPU_niklas<double>;
template class initial_SDDMM_GPU_niklas<int>;