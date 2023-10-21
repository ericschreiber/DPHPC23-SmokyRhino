// naive__CPU_SDDMM.hpp
#ifndef NAIVE_CPU_SDDMM_HPP
#define NAIVE_CPU_SDDMM_HPP

#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_CPU_SDDMM : public SDDMMlib<T>
{
    public:
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            const SparseMatrix<T>& z,
            SparseMatrix<T>& result) const override;
};

#endif  // NAIVE_CPU_SDDMM_HPP