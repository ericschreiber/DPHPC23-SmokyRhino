// naive_CPU_SDDMM.cpp
#include "naive_CPU_SDDMM.hpp"

#include <iostream>

template <typename T>
void naive_CPU_SDDMM<T>::SDDMM(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    const SparseMatrix<T>& z,
    SparseMatrix<T>& result) const
{
    // please implement

    return;
}

// Explicit template instantiation
template class naive_CPU_SDDMM<float>;
template class naive_CPU_SDDMM<double>;
template class naive_CPU_SDDMM<int>;