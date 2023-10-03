// SDDMMlib.hpp
#ifndef SDDMMLIB_HPP
#define SDDMMLIB_HPP

#include "MatrixLib/SparseMatrix.hpp"
#include "MatrixLib/DenseMatrix.hpp"

template <typename T>
class SDDMMlib {
    public:
        static void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) = 0;
};

#endif // SDDMMLIB_HPP