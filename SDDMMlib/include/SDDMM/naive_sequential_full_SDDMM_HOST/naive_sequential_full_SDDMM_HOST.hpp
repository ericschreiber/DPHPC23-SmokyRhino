// naive_sequential_full_SDDMM.hpp
#ifndef NAIVE_SEQUENTIAL_FULL_SDDMM_HPP
#define NAIVE_SEQUENTIAL_FULL_SDDMM_HPP

#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_sequential_full_SDDMM_HOST : public SDDMMlib<T>
{
    public:
        virtual void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) const override;
};

#endif  // NAIVE_SEQUENTIAL_FULL_SDDMM_HPP