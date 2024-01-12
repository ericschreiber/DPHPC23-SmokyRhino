#ifndef MERGED_HPP
#define MERGED_HPP

#include <cassert>
#include <type_traits>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class merged : public SDDMMlib<T>
{
    public:
        merged() { assert(false && "Error: merged<T>::merged() is only implemented for float."); }
        merged(ExecutionTimer* timer) { assert(false && "Error: merged<T>::merged() is only implemented for float."); }
        virtual void SDDMM(
            const DenseMatrix<T>& matrixA_HOST,
            const DenseMatrix<T>& matrixB_HOST,
            const SparseMatrix<T>& matrixC_HOST,
            SparseMatrix<T>& matrixResult_HOST,
            const int num_iterations) const override { assert(false && "Error: merged<T>::SDDMM() is only implemented for float."); }
        virtual void start_run() const override {}  // Would need to be implemented but we don't need it because the class can never be constructed except for float
        virtual void stop_run() const override {}
};

template <>
class merged<float> : public SDDMMlib<float>
{
    public:
        merged() {}
        merged(ExecutionTimer* timer) { this->_timer = timer; }
        virtual void SDDMM(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const SparseMatrix<float>& matrixC_HOST,
            SparseMatrix<float>& matrixResult_HOST,
            const int num_iterations) const override;

    private:
        void SDDMM_COO(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const COOMatrix<float>& matrixC_HOST,
            COOMatrix<float>& matrixResult_HOST,
            const int num_iterations) const;

        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif
