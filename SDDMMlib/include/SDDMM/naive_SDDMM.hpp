// naive_SDDMM.hpp
#ifndef NAIVE_SDDMM_HPP
#define NAIVE_SDDMM_HPP

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_SDDMM : public SDDMMlib<T>
{
    public:
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            const SparseMatrix<T>& z,
            SparseMatrix<T>& result) const override;

    private:
        void naive_SDDMM_CSR(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            const CSRMatrix<T>& z,
            CSRMatrix<T>& result) const;
};

#endif  // NAIVE_SDDMM_HPP