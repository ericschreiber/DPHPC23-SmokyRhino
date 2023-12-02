// test_hub.hpp
#ifndef TEST_HUB_GPU_HPP
#define TEST_HUB_GPU_HPP

#include <cassert>
#include <type_traits>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class test_hub_GPU : public SDDMMlib<T>
{
    public:
        test_hub_GPU() { assert(false && "Error: test_hub_GPU<T>::test_hub_GPU() is only implemented for float."); }
        test_hub_GPU(ExecutionTimer* timer) { assert(false && "Error: naive_SDDMM<T>::naive_SDDMM() is only implemented for float."); }
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST) const override;
        virtual void start_run() const override {}  // Would need to be implemented but we don't need it because the class can never be constructed except for float
        virtual void stop_run() const override {}
};

template <>
class test_hub_GPU<float> : public SDDMMlib<float>
{
    public:
        test_hub_GPU() {}
        test_hub_GPU(ExecutionTimer* timer) { this->_timer = timer; }
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

        void SDDMM_DENSE(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_transpose_HOST,
            const DenseMatrix<float>& matrixC_HOST,
            DenseMatrix<float>& matrixResult_HOST) const;
        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif  // TEST_HUB_GPU_HPP