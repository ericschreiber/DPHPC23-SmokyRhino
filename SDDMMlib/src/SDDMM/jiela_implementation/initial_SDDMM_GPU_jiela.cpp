// initial_SDDMM_GPU_jiela.cpp
#include <iostream>

#include "naive_SDDMM.hpp"

template <typename T>
void naive_SDDMM<T>::SDDMM(
    const DenseMatrix<T>& matrixA_HOST,
    const DenseMatrix<T>& matrixB_HOST,
    const SparseMatrix<T>& matrixC_HOST,
    SparseMatrix<T>& matrixResult_HOST) const
{
    // please implement
    std::cout << "naive_SDDMM from jiela was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class naive_SDDMM<float>;
template class naive_SDDMM<double>;
template class naive_SDDMM<int>;