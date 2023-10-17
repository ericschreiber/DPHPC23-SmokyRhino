// initial_SDDMM_GPU_jiela.hpp
#ifndef INITIAL_SDDMM_GPU_JIELA_HPP
#define INITIAL_SDDMM_GPU_JIELA_HPP

#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"
#include "DenseMatrix.hpp"

template <typename T>
class initial_SDDMM_GPU_jiela : public SDDMMlib<T> {
    public:
        virtual void SDDMM(const DenseMatrix<T>& matrixA, const DenseMatrix<T>& matrixB, const SparseMatrix<T>& matrixC, SparseMatrix<T>& matrixResult) const override;
    private:
        void SDDMM_DENSE(const DenseMatrix<T>& matrixA, const DenseMatrix<T>& matrixB_transpose, const DenseMatrix<T>& matrixC, DenseMatrix<T>& matrixResult) const;
};

#endif // INITIAL_SDDMM_GPU_JIELA_HPP