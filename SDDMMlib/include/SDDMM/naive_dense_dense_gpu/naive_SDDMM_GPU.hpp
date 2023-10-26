// naive_SDDMM_GPU.hpp
#ifndef NAIVE_SDDMM_GPU_HPP
#define NAIVE_SDDMM_GPU_HPP

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_SDDMM_GPU : public SDDMMlib<T>
{
    public:
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST) const override;

    private:
        virtual void SDDMM_CSR(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const CSRMatrix<T>& matrixC_HOST,
            CSRMatrix<T>& matrixResult_HOST) const;

        void SDDMM_DENSE(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_transpose_HOST,
            const DenseMatrix<T>& matrixC_HOST,
            DenseMatrix<T>& matrixResult_HOST) const;
};

#endif  // NAIVE_SDDMM_GPU_HPP