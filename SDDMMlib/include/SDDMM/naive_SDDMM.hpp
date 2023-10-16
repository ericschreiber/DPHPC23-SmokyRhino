// naive_SDDMM.hpp
#ifndef NAIVE_SDDMM_HPP
#define NAIVE_SDDMM_HPP

#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"
#include "DenseMatrix.hpp"

template <typename T>
class naive_SDDMM : public SDDMMlib<T> {
    public:
        virtual void SDDMM(const DenseMatrix<T>& matrixA, const DenseMatrix<T>& matrixB, const SparseMatrix<T>& matrixC, SparseMatrix<T>& matrixResult) const override;
};

#endif // NAIVE_SDDMM_HPP