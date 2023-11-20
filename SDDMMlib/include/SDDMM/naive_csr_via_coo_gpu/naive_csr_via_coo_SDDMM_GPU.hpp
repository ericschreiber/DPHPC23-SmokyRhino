// naive_csr_via_coo_SDDMM_GPU.hpp
#ifndef NAIVE_CSR_VIA_COO_SDDMM_GPU_HPP
#define NAIVE_CSR_VIA_COO_SDDMM_GPU_HPP

#include <cassert>
#include <type_traits>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_csr_via_coo_SDDMM_GPU : public SDDMMlib<T>
{
    public:
        naive_csr_via_coo_SDDMM_GPU() { assert(false && "Error: naive_csr_via_coo_SDDMM_GPU<T>::naive_csr_via_coo_SDDMM_GPU() is only implemented for float."); }
        naive_csr_via_coo_SDDMM_GPU(ExecutionTimer* timer) { assert(false && "Error: naive_csr_via_coo_SDDMM_GPU<T>::naive_csr_via_coo_SDDMM_GPU() is only implemented for float."); }
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST) const override {}
        virtual void start_run() const override {}  // Would need to be implemented but we don't need it because the class can never be constructed except for float
        virtual void stop_run() const override {}
};

template <>
class naive_csr_via_coo_SDDMM_GPU<float> : public SDDMMlib<float>
{
    public:
        naive_csr_via_coo_SDDMM_GPU() {}
        naive_csr_via_coo_SDDMM_GPU(ExecutionTimer* timer) { this->_timer = timer; }
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

        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif  // NAIVE_CSR_VIA_COO_SDDMM_GPU_HPP