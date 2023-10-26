// naive_CPU_SDDMM.cpp
#include "naive_CPU_SDDMM.hpp"

#include <iostream>
#include <stdexcept>
#include <vector>

#include "CSRMatrix.hpp"

template <typename T>
void naive_CPU_SDDMM<T>::SDDMM(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    const SparseMatrix<T>& z,
    SparseMatrix<T>& result) const
{
    // SDDMM Sampled implementation for CSR matrix:

    std::vector<int> RowPtr = z.getRowPtr();
    std::vector<int> ColIndices = z.getColIndices();
    std::vector<T> values = z.getValues();
    std::vector<int> calcRowPtr = {0};
    std::vector<int> calcColIndices;
    std::vector<T> calcValues;
    T XY_element, h;
    int start_el, end_el, col, row = 0;

    // Iterating over RowPointer
    for (int row_i = 0; row_i < RowPtr.size() - 1; row_i++)
    {
        start_el = RowPtr[row_i];
        end_el = RowPtr[row_i + 1];

        // Iterating over values and their column in a row
        for (int i = start_el; i < end_el; i++)
        {
            XY_element = 0;
            col = ColIndices[i];
            // Iterating over K
            for (int k = 0; k < x.getNumCols(); k++)
            {
                XY_element += x.at(row, k) * y.at(col, k);
            }
            // Hadamard product
            h = values[i] * XY_element;
            if (h != 0)
            {
                calcValues.push_back(h);
                calcColIndices.push_back(col);
            }
        }
        calcRowPtr.push_back(calcValues.size());
        row += 1;
    }

    result.setValues(calcValues);
    result.setColIndices(calcColIndices);
    result.setRowPtr(calcRowPtr);

    return;
}

// Explicit template instantiation
template class naive_CPU_SDDMM<float>;
template class naive_CPU_SDDMM<double>;
template class naive_CPU_SDDMM<int>;
