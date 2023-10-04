// naive_SDDMM.hpp
#ifndef NAIVE_SDDMM_HPP
#define NAIVE_SDDMM_HPP

#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"
#include "DenseMatrix.hpp"

template <typename T>
class naive_SDDMM : public SDDMMlib<T> {
    public:
        virtual void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) const override;
};

#endif // NAIVE_SDDMM_HPP