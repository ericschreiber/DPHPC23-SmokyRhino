// naive_sequential_sampled_SDDMM.cpp
#include "naive_sequential_sampled_SDDMM_HOST/naive_sequential_sampled_SDDMM_HOST.hpp"

#include <iostream>

template <typename T>
void naive_sequential_sampled_SDDMM_HOST<T>::SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) const
{
    // This is literally just a straightforward implementation of Algorithm 2 in the SDMM HPC Paper

    int m = x.getNumRows();
    int n = x.getNumCols();
    int k = y.getNumCols();

    // I assume a size check has been done for now, but we might want to make that one explicitly
    // Please note that the paper uses A[M][K] and B[N][K]. I.e. B already seems to be transposed!
    // I am making the same assumption
    // I also assume we are taking a CRS matrix

    for (int i = 0; i < m; i++)
    {
        for (int j = z.getRowPtr()[i]; j < z.getRowPtr()[i + 1]; j++)
        {
            for (int l = 0; l < k; l++)
            {
                result.getValues()[j] += x.at(i, l) * y.at(z.getColIndices()[j], l);
            }
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = z.getRowPtr()[i]; j < z.getRowPtr()[i + 1]; j++)
        {
            result.getValues()[j] *= z.getValues()[j];
        }
    }

    std::cout << "naive_sequential_sampled_SDDMM was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class naive_sequential_sampled_SDDMM_HOST<float>;
template class naive_sequential_sampled_SDDMM_HOST<double>;
template class naive_sequential_sampled_SDDMM_HOST<int>;