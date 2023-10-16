// naive_SDDMM_GPU.cpp
#include "naive_dense_dense_gpu/naive_SDDMM_GPU.hpp"

#include <iostream>

template <typename T>
void naive_SDDMM_GPU<T>::SDDMM_DENSE(
    const DenseMatrix<T>& matrixA_HOST,
    const DenseMatrix<T>& matrixB_transpose_HOST,
    const DenseMatrix<T>& matrixC_HOST,
    DenseMatrix<T>& matrixResult_HOST) const
{
    // input matrices are on CPU - need to be copied
    /*
    // get sizes of matrixA and matrixB {A=mxk; B=kxn; B_transpose=nxk}
    int m = matrixA.getNumRows();
    int n = matrixA.getNumCols();
    int k = matrixB.getNumCols();

    // allocate memory for the matrices on the GPU
    float *matrixA_GPU;
    float *matrixB_transpose_GPU;
    float *matrixC_GPU;
    float *matrixResult_GPU;
    CUDA_CHECK(cudaMalloc(&matrixA_GPU, m*k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixB_transpose_GPU, n*k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixC_GPU, m*n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixResult_GPU, m*n * sizeof(float)));

    // copy matrices to the GPU
    CUDA_CHECK(cudaMemcpy(matrixA_GPU, matrixA_HOST m*k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixB_transpose_GPU, matrixB_transpose, n*k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixC_GPU, matrixC_HOST m*n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixResult_GPU, matrixResult, m*n * sizeof(float), cudaMemcpyHostToDevice));

    // call compute in naive_dense_dense.cu
    compute(m, n, k, matrixA_GPU, matriB_transpose_GPU, matrixC_GPU, matrixResult_GPU)

    // copy result from the GPU
    CUDA_CHECK(cudaMemcpy(matrixResult, matrixResult_GPU, m*n * sizeof(float), cudaMemcpyDeviceToHost));

    // free memory on the device
    CUDA_CHECK(cudaFree(matrixA_GPU));
    CUDA_CHECK(cudaFree(matrixB_transpose_GPU));
    CUDA_CHECK(cudaFree(matrixC_GPU));
    CUDA_CHECK(cudaFree(matrixResult_GPU));
    */
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
    /*
    // transpose matrixB to B^t
    matrixB_transpose =

    // change matrixB and matrixResult to a dense matrix



    // call naive_SDDMM_GPU to compute the SDDMM
    matrixA.SDDMM_DENSE(matrixB_HOST matrixC_HOST matrixResult, std::bind(&naive_SDDMM_GPU<float>::SDDMM_DENSE, naive_SDDMM_GPU<float>(),
    std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

    // change matrixResult to a sparse matrix

    */
    std::cout << "naive_SDDMM was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class naive_SDDMM_GPU<float>;
template class naive_SDDMM_GPU<double>;
template class naive_SDDMM_GPU<int>;