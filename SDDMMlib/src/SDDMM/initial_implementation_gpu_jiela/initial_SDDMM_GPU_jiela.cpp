// initial_SDDMM_GPU_jiela.cpp
#include "initial_implementation_gpu_jiela/initial_SDDMM_GPU_jiela.hpp"

#include <iostream>

template <typename T>
void initial_SDDMM_GPU_jiela<T>::SDDMM(
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
template class initial_SDDMM_GPU_jiela<float>;
template class initial_SDDMM_GPU_jiela<double>;
template class initial_SDDMM_GPU_jiela<int>;