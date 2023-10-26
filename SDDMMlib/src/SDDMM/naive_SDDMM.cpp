// naive_SDDMM.cpp
#include "naive_SDDMM.hpp"

#include <iostream>

template <typename T>
void naive_SDDMM<T>::SDDMM(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    const SparseMatrix<T>& z,
    SparseMatrix<T>& result) const
{
    // Check if CSRMatrix
    const CSRMatrix<T>* csrMatrix = dynamic_cast<const CSRMatrix<T>*>(&z);
    CSRMatrix<T>* csrResult = dynamic_cast<CSRMatrix<T>*>(&result);
    if (csrMatrix == nullptr || csrResult == nullptr)
    {
        throw std::invalid_argument("Error: naive_SDDMM::SDDMM() only accepts CSRMatrix<T> as input. Other formats are not supported yet");
    }
    else
    {
        naive_SDDMM_CSR(x, y, *csrMatrix, *csrResult);
    }

    return;
}

template <typename T>
void naive_SDDMM<T>::naive_SDDMM_CSR(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    const CSRMatrix<T>& z,
    CSRMatrix<T>& result) const
{
    // please implement
    std::cout << "naive_SDDMM was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class naive_SDDMM<float>;
template class naive_SDDMM<double>;
template class naive_SDDMM<int>;