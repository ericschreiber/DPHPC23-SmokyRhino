// naive_sequential_full_SDDMM.cpp
#include "naive_sequential_full_SDDMM_HOST/naive_sequential_full_SDDMM_HOST.hpp"

#include <iostream>

#include "DenseMatrix.hpp"

template <typename T>
void naive_sequential_full_SDDMM_HOST<T>::SDDMM(
    const DenseMatrix<T>& x,
    const DenseMatrix<T>& y,
    const SparseMatrix<T>& z,
    SparseMatrix<T>& result) const
{
    // This is a very dumb implementation, because it samples only AFTER the
    // matrix x matrix multiplication

    int m = x.getNumRows();
    int n = x.getNumCols();
    int k = y.getNumCols();

    auto xy = DenseMatrix<T>(m, n);

    // I assume a size check has been done for now, but we might want to make that
    // one explicitly Please note that the paper uses A[M][K] and B[N][K]. I.e. B
    // already seems to be transposed! I am making the same assumption
    // I also assume we are taking a CRS matrix

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int l = 0; l < k; l++)
            {
                auto mul = x.at(i, l) * y.at(j, l);
                auto curr_xy_ij = xy.at(i, j);
                xy.setValue(i, j, mul + curr_xy_ij);
            }
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = z.getRowPtr()[i]; j < z.getRowPtr()[i + 1]; j++)
        {
            std::cout << "entered inner loop" << std::endl;
            // std::cout << j << std::endl;
            auto temp = xy.at(i, z.getColIndices()[j]);
            std::cout << "assigned the temp" << std::endl;
            // std::cout << z.getValues() << std::endl;
            std::cout << "values" << std::endl;
            for (int höck = 0; höck < z.getNumValues(); höck++)
            {
                std::cout << z.getValues()[höck] << std::endl;
            }
            std::cout << "num Values, num Rows, num cols" << std::endl;
            std::cout << z.getNumValues() << std::endl;
            // std::cout << j << std::endl;
            std::cout << z.getNumRows() << std::endl;
            std::cout << z.getNumCols() << std::endl;

            result.getValues()[j] = temp * z.getValues()[j];
            std::cout << "did a thing" << std::endl;
        }
    }

    std::cout << "naive_sequential_sampled_SDDMM was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class naive_sequential_full_SDDMM_HOST<float>;
template class naive_sequential_full_SDDMM_HOST<double>;
template class naive_sequential_full_SDDMM_HOST<int>;