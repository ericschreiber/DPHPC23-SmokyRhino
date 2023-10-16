// initial_SDDMM_GPU_jiela.hpp
#ifndef INITIAL_SDDMM_GPU_JIELA_HPP
#define INITIAL_SDDMM_GPU_JIELA_HPP

#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class initial_SDDMM_GPU_jiela : public SDDMMlib<T>
{
    public:
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST) const override;
};

#endif  // INITIAL_SDDMM_GPU_JIELA_HPP