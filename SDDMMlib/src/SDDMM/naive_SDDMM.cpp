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
    // please implement
    this->start_run();
    std::cout << "naive_SDDMM was executed :)" << std::endl;
    this->stop_run();
    return;
}

template <typename T>
void naive_SDDMM<T>::start_run() const
{
    this->_timer->start_cpu_run();  // OR _timer.start_gpu_run();
}

template <typename T>
void naive_SDDMM<T>::stop_run() const
{
    this->_timer->stop_cpu_run();  // OR _timer.stop_gpu_run();
}

// Explicit template instantiation
template class naive_SDDMM<float>;
template class naive_SDDMM<double>;
template class naive_SDDMM<int>;