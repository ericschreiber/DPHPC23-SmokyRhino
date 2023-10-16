// SDDMMlib.hpp
#ifndef SDDMMLIB_HPP
#define SDDMMLIB_HPP

#include "DenseMatrix.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class SDDMMlib
{
    public:
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST) const = 0;
};

#endif  // SDDMMLIB_HPP