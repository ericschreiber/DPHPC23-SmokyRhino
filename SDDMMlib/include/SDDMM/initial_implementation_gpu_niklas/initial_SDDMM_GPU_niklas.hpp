// initial_SDDMM_GPU_niklas.hpp
#ifndef INITIAL_SDDMM_GPU_NIKLAS_HPP
#define INITIAL_SDDMM_GPU_NIKLAS_HPP

#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class initial_SDDMM_GPU_niklas : public SDDMMlib<T>
{
    public:
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST) const override;
};

#endif  // INITIAL_SDDMM_GPU_NIKLAS_HPP