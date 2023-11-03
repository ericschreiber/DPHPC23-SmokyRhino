// naive_coo_SDDMM_GPU.cpp
#include "naive_coo_gpu/naive_coo_SDDMM_GPU.hpp"

#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "utils.h"

void naive_coo_SDDMM_GPU<float>::SDDMM_COO(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const COOMatrix<float>& matrixC_HOST,
    COOMatrix<float>& matrixResult_HOST) const
{
    std::cout << "SDDMM_COO will run but must be implemented" << std::endl;
    return;
}

void naive_coo_SDDMM_GPU<float>::SDDMM(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const SparseMatrix<float>& matrixC_HOST,
    SparseMatrix<float>& matrixResult_HOST) const
{
    const COOMatrix<float>* cooMatrixC = dynamic_cast<const COOMatrix<float>*>(&matrixC_HOST);
    COOMatrix<float>* cooMatrixResult = dynamic_cast<COOMatrix<float>*>(&matrixResult_HOST);
    if (cooMatrixC == nullptr || cooMatrixResult == nullptr)
    {
        throw std::invalid_argument("Error: convert Sparse to COO before using this function");
    }
    else
    {
        SDDMM_COO(
            matrixA_HOST,
            matrixB_HOST,
            *cooMatrixC,
            *cooMatrixResult);
    }

    cooMatrixC = nullptr;
    cooMatrixResult = nullptr;

    return;
}

void naive_coo_SDDMM_GPU<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: naive_coo_SDDMM_GPU::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_cpu_run();
}

void naive_coo_SDDMM_GPU<float>::stop_run() const
{
    this->_timer->stop_cpu_run();
}

// Explicit template instantiation
// template class naive_coo_SDDMM_GPU<float>;
template class naive_coo_SDDMM_GPU<double>;
template class naive_coo_SDDMM_GPU<int>;