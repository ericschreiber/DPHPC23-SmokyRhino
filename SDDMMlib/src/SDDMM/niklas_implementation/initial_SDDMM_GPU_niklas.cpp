// initial_SDDMM_GPU_niklas.cpp
#include <iostream>

#include "naive_SDDMM.hpp"

template <typename T>
void naive_SDDMM<T>::SDDMM(
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
template class naive_SDDMM<float>;
template class naive_SDDMM<double>;
template class naive_SDDMM<int>;