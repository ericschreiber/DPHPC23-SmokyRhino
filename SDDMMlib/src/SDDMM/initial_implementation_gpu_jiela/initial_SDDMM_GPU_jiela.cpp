// initial_SDDMM_GPU_jiela.cpp
#include "initial_implementation_gpu_jiela/initial_SDDMM_GPU_jiela.hpp"

#include <iostream>

#include "DenseMatrix.hpp"

template <typename T>
void initial_SDDMM_GPU_jiela<T>::SDDMM(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    const SparseMatrix<T>& z,
    SparseMatrix<T>& result) const
{
    // This uses a blocked version, where each row of the matrix is handled by one block

    // input matrices are on CPU - need to be copied
    // for the sparse matrices, we need to copy not only the values, but also the access vectors!
    // However since the Result and z will have the same access vectors we only need to copy those once.
    /*
    // get sizes of matrixA and matrixB {A=mxk; B=kxn; B_transpose=nxk}
    int m = x.getNumRows();
    int n = y.getNumCols();
    int k = x.getNumCols();
    int nnz = z.getNumValues();

    // allocate memory for the matrices on the GPU
    float *matrixA_GPU;
    float *matrixB_transpose_GPU;
    float *sparse_C_values_GPU;
    int *sparse_rowPtr_GPU;
    int *sparse_colIdx_GPU
    float *matrixResult_GPU;
    CUDA_CHECK(cudaMalloc(&matrixA_GPU, m*k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&matrixB_transpose_GPU, n*k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sparse_C_values_GPU, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sparse_rowPtr_GPU, (m +1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&sparse_colIdx_GPU, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&matrixResult_GPU, nnz * sizeof(float)));

    // copy matrices to the GPU
    CUDA_CHECK(cudaMemcpy(matrixA_GPU, matrixA, m*k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixB_transpose_GPU, matrixB_transpose, n*k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sparse_C_values_GPU, z.getValues(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(sparse_rowPtr, z.getRowPtr(), (m+1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(CcudaMemcpy(sparse_colIdx_GPU, z.getColIndices(), nnz * sizeof(int), cudamemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrixResult_GPU, matrixResult, nnz * sizeof(float), cudaMemcpyHostToDevice));

    // call compute_blockwise in initial_SDDMM_GPU_kernel_jiela.cu
    compute_blockwise(m, n, k, matrixA_GPU, matriB_transpose_GPU, sparse_C_values_GPU, sparse_rowPtr, sparse_colIdx_GPU, matrixResult_GPU)

    // copy result from the GPU
    CUDA_CHECK(cudaMemcpy(matrixResult, matrixResult_GPU, nnz * sizeof(float), cudaMemcpyDeviceToHost));
    */

    std::cout << "naive_SDDMM from jiela was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class initial_SDDMM_GPU_jiela<float>;
template class initial_SDDMM_GPU_jiela<double>;
template class initial_SDDMM_GPU_jiela<int>;
;