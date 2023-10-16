// SDDMMlib.hpp
#ifndef SDDMMLIB_HPP
#define SDDMMLIB_HPP

#include "DenseMatrix.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class SDDMMlib
{
   public:
    virtual void SDDMM(const DenseMatrix<T>& matrixA, const DenseMatrix<T>& matrixB, const SparseMatrix<T>& matrixC,
                       SparseMatrix<T>& matrixResult) const = 0;
};

#endif  // SDDMMLIB_HPP