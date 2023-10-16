// initial_SDDMM_GPU_niklas.cpp
#include "naive_SDDMM.hpp"
#include <iostream>

template <typename T>
void naive_SDDMM<T>::SDDMM(const DenseMatrix<T>& matrixA,
                           const DenseMatrix<T>& matrixB,
                           const SparseMatrix<T>& matrixC,
                           SparseMatrix<T>& matrixResult) const
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