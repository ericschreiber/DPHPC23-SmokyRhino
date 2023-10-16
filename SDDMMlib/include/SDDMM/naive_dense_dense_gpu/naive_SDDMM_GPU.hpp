// naive_SDDMM_GPU.hpp
#ifndef NAIVE_SDDMM_GPU_HPP
#define NAIVE_SDDMM_GPU_HPP

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
        void SDDMM_DENSE(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_transpose_HOST,
            const DenseMatrix<T>& matrixC_HOST,
            DenseMatrix<T>& matrixResult_HOST) const;
};

#endif  // NAIVE_SDDMM_GPU_HPP