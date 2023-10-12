// naive_SDDMM.cpp
#include "naive_sequential_full_SDDMM.hpp"
#include "DenseMatrix.hpp"
#include <iostream>

template <typename T>
void naive_sequential_full_SDDMM<T>::SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) const {

    // This is a very dumb implementation, because it samples only AFTER the matrix x matrix multiplication

    int m = x.getNumRows();
    int n = x.getNumCols();
    int k = y.getNumCols();

    auto xy = DenseMatrix<T>(m, n); 

    // I assume a size check has been done for now, but we might want to make that one explicitly
    // Please note that the paper uses A[M][K] and B[N][K]. I.e. B already seems to be transposed!
        // I am making the same assumption
    // I also assume we are taking a CRS matrix

    for(int i = 0; i < m; i++){
        for(int j = 0; j < n; j++){
            for(int l = 0; l < k; l++){
                xy[i][j] += x[i][l] * y[j][l];
            }
        }
    }

    for(int i = 0; i < m; i ++){
        for(int j = z.getRowPtr[i]; j < z.getRowPtr[i+1]; j++){
            result.getValues[j] *= z.getValues[j];
        }
    }

    std::cout << "naive_sequential_sampled_SDDMM was executed :)" << std::endl;
    return;
}

// Explicit template instantiation
template class naive_sequential_full_SDDMM<float>;
template class naive_sequential_full_SDDMM<double>;
template class naive_sequential_full_SDDMM<int>;