// naive_SDDMM.cpp
#include "naive_SDDMM.hpp"

template <typename T>
void naive_SDDMM<T>::SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) const {
    // please implement
    return;
}

// Explicit template instantiation
template class naive_SDDMM<float>;
template class naive_SDDMM<double>;
template class naive_SDDMM<int>;