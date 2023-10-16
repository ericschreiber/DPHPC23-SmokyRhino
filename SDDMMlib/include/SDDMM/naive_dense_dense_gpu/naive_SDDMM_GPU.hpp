// naive_SDDMM_GPU.hpp
#ifndef NAIVE_SDDMM_GPU_HPP
#define NAIVE_SDDMM_GPU_HPP

#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"
#include "DenseMatrix.hpp"

template <typename T>
class naive_SDDMM_GPU : public SDDMMlib<T> {
    public:
        virtual void SDDMM(const DenseMatrix<T>& matrixA, const DenseMatrix<T>& matrixB, const SparseMatrix<T>& matrixC, SparseMatrix<T>& matrixResult) const override;
    private:
        void SDDMM_DENSE(const DenseMatrix<T>& matrixA, const DenseMatrix<T>& matrixB_transpose, const DenseMatrix<T>& matrixC, DenseMatrix<T>& matrixResult) const;
};

#endif // NAIVE_SDDMM_GPU_HPP