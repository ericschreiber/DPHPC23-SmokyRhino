// SDDMMlib.hpp
#ifndef SDDMMLIB_HPP
#define SDDMMLIB_HPP

#include "SparseMatrix.hpp"
#include "DenseMatrix.hpp"

template <typename T>
class SDDMMlib {
    public:
        virtual void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) const = 0;

};

#endif // SDDMMLIB_HPP