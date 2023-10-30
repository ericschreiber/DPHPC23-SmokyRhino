// naive_SDDMM.cpp
#include "naive_SDDMM.hpp"

#include <iostream>

// Only the float type of the class is valid all other types will throw an error
void naive_SDDMM<float>::SDDMM(
    const DenseMatrix<float>& x,
    const DenseMatrix<float>& y,
    const SparseMatrix<float>& z,
    SparseMatrix<float>& result) const
{
    // Check if CSRMatrix
    const CSRMatrix<float>* csrMatrix = dynamic_cast<const CSRMatrix<float>*>(&z);
    CSRMatrix<float>* csrResult = dynamic_cast<CSRMatrix<float>*>(&result);
    if (csrMatrix == nullptr || csrResult == nullptr)
    {
        throw std::invalid_argument("Error: naive_SDDMM::SDDMM() only accepts CSRMatrix<float> as input. Other formats are not supported yet");
    }
    else
    {
        naive_SDDMM_CSR(x, y, *csrMatrix, *csrResult);
    }

    return;
}

template <typename T>
void naive_SDDMM<T>::SDDMM(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    const SparseMatrix<T>& z,
    SparseMatrix<T>& result) const
{
    assert(false && "Error: naive_SDDMM::SDDMM() only accepts float as input. Other types are not supported yet");
}

void naive_SDDMM<float>::naive_SDDMM_CSR(
    const DenseMatrix<float>& x,
    const DenseMatrix<float>& y,
    const CSRMatrix<float>& z,
    CSRMatrix<float>& result) const
{
    // please implement
    this->start_run();
    std::cout << "naive_SDDMM was executed :)" << std::endl;
    this->stop_run();
    return;
}

void naive_SDDMM<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: naive_SDDMM::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_cpu_run();  // OR _timer.start_gpu_run();
}

void naive_SDDMM<float>::stop_run() const
{
    this->_timer->stop_cpu_run();  // OR _timer.stop_gpu_run();
}

// Explicit template instantiation
// template class naive_SDDMM<float>;
template class naive_SDDMM<double>;
template class naive_SDDMM<int>;