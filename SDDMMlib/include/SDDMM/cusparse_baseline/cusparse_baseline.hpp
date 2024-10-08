#ifndef CUSPARSE_BASELINE_HPP
#define CUSPARSE_BASELINE_HPP

#include <cassert>
#include <type_traits>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class cusparse_baseline : public SDDMMlib<T>
{
    public:
        cusparse_baseline() { assert(false && "Error: cusparse_baseline<T>::cusparse_baseline() is only implemented for float."); }
        cusparse_baseline(ExecutionTimer* timer) { assert(false && "Error: cusparse_baseline<T>::cusparse_baseline() is only implemented for float."); }
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST,
            const int num_iterations) const override {}
        virtual void start_run() const override {}  // Would need to be implemented but we don't need it because the class can never be constructed except for float
        virtual void stop_run() const override {}
};

template <>
class cusparse_baseline<float> : public SDDMMlib<float>
{
    public:
        cusparse_baseline() {}
        cusparse_baseline(ExecutionTimer* timer) { this->_timer = timer; }
        virtual void SDDMM(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const SparseMatrix<float>& matrixC_HOST,
            SparseMatrix<float>& matrixResult_HOST,
            const int num_iterations) const override;

    private:
        void SDDMM_CSR(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const CSRMatrix<float>& matrixC_HOST_CSR,
            CSRMatrix<float>& matrixResult_HOST,
            const int num_iterations) const;

        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif  // CACHE_COO_SDDMM_GPU_HPP
