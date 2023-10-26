// naive_SDDMM_GPU.hpp
#ifndef NAIVE_SDDMM_GPU_HPP
#define NAIVE_SDDMM_GPU_HPP

#include <cassert>
#include <type_traits>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_SDDMM_GPU : public SDDMMlib<T>
{
    public:
        naive_SDDMM_GPU() { assert(false && "Error: naive_SDDMM_GPU<T>::naive_SDDMM_GPU() is only implemented for float."); }
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST) const override;
};

template <>
class naive_SDDMM_GPU<float> : public SDDMMlib<float>
{
    public:
        virtual void SDDMM(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const SparseMatrix<float>& matrixC_HOST,
            SparseMatrix<float>& matrixResult_HOST) const override;

    private:
        void SDDMM_CSR(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const CSRMatrix<float>& matrixC_HOST,
            CSRMatrix<float>& matrixResult_HOST) const;

        void SDDMM_DENSE(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_transpose_HOST,
            const DenseMatrix<float>& matrixC_HOST,
            DenseMatrix<float>& matrixResult_HOST) const;
};

#endif  // NAIVE_SDDMM_GPU_HPP