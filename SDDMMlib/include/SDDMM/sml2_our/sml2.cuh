// sml2.cuh
#ifndef SML2_CUH
#define SML2_CUH

#include <cassert>
#include <type_traits>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class sml2_our : public SDDMMlib<T>
{
    public:
        sml2_our() { assert(false && "Error: semi_naive_CSR_SDDMM_GPU<T>::semi_naive_CSR_SDDMM_GPU() is only implemented for float."); }
        sml2_our(ExecutionTimer* timer) { assert(false && "Error: naive_SDDMM<T>::semi_naive_CSR_SDDMM_GPU() is only implemented for float."); }
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST,
            const int num_iterations) const override;
        virtual void start_run() const override {}  // Would need to be implemented but we don't need it because the class can never be constructed except for float
        virtual void stop_run() const override {}
};

template <>
class sml2_our<float> : public SDDMMlib<float>
{
    public:
        sml2_our() {}
        sml2_our(ExecutionTimer* timer) { this->_timer = timer; }
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
            const CSRMatrix<float>& matrixC_HOST,
            CSRMatrix<float>& matrixResult_HOST,
            const int num_iterations) const;
        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif  // SML2_CUH