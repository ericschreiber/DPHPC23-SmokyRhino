// naive_CPU_SDDMM.cpp
#include "naive_CPU_SDDMM.hpp"

#include <iostream>
#include <stdexcept>

#include "CSRMatrix.hpp"

template <typename T>
void naive_CPU_SDDMM<T>::SDDMM(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    const SparseMatrix<T>& z,
    SparseMatrix<T>& result) const
{
    // Dense-Dense multiplication

    if (x.getNumCols() != y.getNumCols())
    {
        throw std::length_error("Error: The dimensions of your matrices x and y don't match");
    }

    DenseMatrix<double> XY(x.getNumRows(), y.getNumRows());
    int XY_element = 0;

    for (int i = 0; i < x.getNumRows(); i++)
    {
        for (int j = 0; j < y.getNumRows(); j++)
        {
            XY_element = 0;
            for (int k = 0; k < y.getNumCols(); k++)
            {
                XY_element += x.at(i, k) * y.at(j, k);
            }
            XY.setValue(i, j, XY_element);
        }
    }

    if (XY.getNumCols() != z.getNumRows())
    {
        throw std::length_error("Error: The dimensions of your matrices XY and z don't match");
    }

    DenseMatrix<double> XYZ(XY.getNumRows(), z.getNumCols());
    int XYZ_element = 0;
    // Dense-Sparse Multiplication

    for (int i = 0; i < XY.getNumRows(); i++)
    {
        for (int j = 0; j < z.getNumCols(); j++)
        {
            XYZ_element = 0;
            for (int k = 0; k < z.getNumRows(); k++)
            {
                XYZ_element += XY.at(i, k) * z.at(k, j);
            }
            std::cout << XYZ_element << " ";
            XYZ.setValue(i, j, XYZ_element);
        }
        std::cout << std::endl;
    }

    return;
}

// Explicit template instantiation
template class naive_CPU_SDDMM<float>;
template class naive_CPU_SDDMM<double>;
template class naive_CPU_SDDMM<int>;