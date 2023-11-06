// naive_coo_SDDMM_GPU.hpp
#ifndef NAIVE_COO_SDDMM_GPU_HPP
#define NAIVE_COO_SDDMM_GPU_HPP

#include <cassert>
#include <type_traits>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_coo_SDDMM_GPU : public SDDMMlib<T>
{
    public:
        naive_coo_SDDMM_GPU() { assert(false && "Error: naive_coo_SDDMM_GPU<T>::naive_coo_SDDMM_GPU() is only implemented for float."); }
        naive_coo_SDDMM_GPU(ExecutionTimer* timer) { assert(false && "Error: naive_coo_SDDMM_GPU<T>::naive_coo_SDDMM_GPU() is only implemented for float."); }
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST) const override {}
        virtual void start_run() const override {}  // Would need to be implemented but we don't need it because the class can never be constructed except for float
        virtual void stop_run() const override {}
};

template <>
class naive_coo_SDDMM_GPU<float> : public SDDMMlib<float>
{
    public:
        naive_coo_SDDMM_GPU() {}
        naive_coo_SDDMM_GPU(ExecutionTimer* timer) { this->_timer = timer; }
        virtual void SDDMM(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const SparseMatrix<float>& matrixC_HOST,
            SparseMatrix<float>& matrixResult_HOST) const override;

    private:
        void SDDMM_COO(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const COOMatrix<float>& matrixC_HOST,
            COOMatrix<float>& matrixResult_HOST) const;

        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif  // NAIVE_COO_SDDMM_GPU_HPP
