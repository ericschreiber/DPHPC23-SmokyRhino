// initial_SDDMM_GPU_niklas.cpp
#include "initial_implementation_gpu_niklas/initial_SDDMM_GPU_niklas.hpp"

#include <iostream>

template <typename T>
void initial_SDDMM_GPU_niklas<T>::SDDMM(
    const DenseMatrix<T>& matrixA_HOST,
    const DenseMatrix<T>& matrixB_HOST,
    const SparseMatrix<T>& matrixC_HOST,
    SparseMatrix<T>& matrixResult_HOST) const
{
    /*
    // get values colIndices and rowPtr from matrixC
    std::vector<T> values_matrixC = matrixC.getValues();
    std::vector<int> colIndices_matrixC = matrixC.getColIndices();
    std::vector<int> rowPtr_matrixC = matrixC.getRowPtr();


    */
    std::cout << "naive_SDDMM from niklas was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class initial_SDDMM_GPU_niklas<float>;
template class initial_SDDMM_GPU_niklas<double>;
template class initial_SDDMM_GPU_niklas<int>;