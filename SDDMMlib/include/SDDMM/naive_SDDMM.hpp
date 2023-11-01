// naive_SDDMM.hpp
#ifndef NAIVE_SDDMM_HPP
#define NAIVE_SDDMM_HPP

#include <cassert>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

// Implement naive SDDMM for all types but return an error if the input is not of type double
template <typename T>
class naive_SDDMM : public SDDMMlib<T>
{
    public:
        naive_SDDMM() { assert(false && "Error: naive_SDDMM<T>::naive_SDDMM() is only implemented for float."); }
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            const SparseMatrix<T>& z,
            SparseMatrix<T>& result) const override;
};

// Implement naive SDDMM for all types but return an error if the input is not of type float
template <>
class naive_SDDMM<float> : public SDDMMlib<float>
{
    public:
        virtual void SDDMM(
            const DenseMatrix<float>& x,
            const DenseMatrix<float>& y,
            const SparseMatrix<float>& z,
            SparseMatrix<float>& result) const override;

    private:
        void naive_SDDMM_CSR(
            const DenseMatrix<float>& x,
            const DenseMatrix<float>& y,
            const CSRMatrix<float>& z,
            CSRMatrix<float>& result) const;
};

#endif  // NAIVE_SDDMM_HPP