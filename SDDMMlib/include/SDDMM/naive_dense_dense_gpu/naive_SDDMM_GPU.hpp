// naive_SDDMM_GPU.hpp
#ifndef NAIVE_SDDMM_GPU_HPP
#define NAIVE_SDDMM_GPU_HPP

#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"
#include "DenseMatrix.hpp"

template <typename T>
class naive_SDDMM_GPU : public SDDMMlib<T> {
    public:
        virtual void SDDMM(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const SparseMatrix<T>& z, SparseMatrix<T>& result) const override;
    private:
        void SDDMM_DENSE(const DenseMatrix<T>& x, const DenseMatrix<T>& y, const DenseMatrix<T>& z, DenseMatrix<T>& result) const;
};

#endif // NAIVE_SDDMM_GPU_HPP