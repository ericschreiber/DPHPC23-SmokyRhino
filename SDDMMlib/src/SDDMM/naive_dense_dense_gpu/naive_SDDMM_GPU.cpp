// naive_SDDMM.cpp
#include "naive_dense_dense_gpu/naive_SDDMM_GPU.hpp"
#include <iostream>

template <typename T>
void naive_SDDMM_GPU<T>::SDDMM_DENSE(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const DenseMatrix<T>& z, DenseMatrix<T>& result) const {
    // input matrices are on CPU - need to be copied
    // please implement
    std::cout << "naive_SDDMM was executed :)" << std::endl;
    return;
}

template <typename T>
void naive_SDDMM_GPU<T>::SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) const {
    // please implement
    std::cout << "naive_SDDMM was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class naive_SDDMM_GPU<float>;
template class naive_SDDMM_GPU<double>;
template class naive_SDDMM_GPU<int>;