// initial_SDDMM_GPU_jiela.cpp
#include "naive_SDDMM.hpp"
#include <iostream>

template <typename T>
void naive_SDDMM<T>::SDDMM(const DenseMatrix<T>& matrixA, const DenseMatrix<T>& matrixB, const SparseMatrix<T>& matrixC, SparseMatrix<T>& matrixResult) const {
    // please implement
    std::cout << "naive_SDDMM from jiela was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class naive_SDDMM<float>;
template class naive_SDDMM<double>;
template class naive_SDDMM<int>;