// naive__CPU_SDDMM.hpp
#ifndef NAIVE_CPU_SDDMM_HPP
#define NAIVE_CPU_SDDMM_HPP

#include <cassert>
#include <type_traits>

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_CPU_SDDMM : public SDDMMlib<T>
{
    public:
        naive_CPU_SDDMM() { assert(false && "Error: naive_SDDMM<T>::naive_SDDMM() is only implemented for float."); }
        naive_CPU_SDDMM(ExecutionTimer* timer) { assert(false && "Error: naive_SDDMM<T>::naive_SDDMM() is only implemented for float."); }
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            const SparseMatrix<T>& z,
            SparseMatrix<T>& result) const override {}
        virtual void start_run() const override {}  // Would need to be implemented but we don't need it because the class can never be constructed except for float
        virtual void stop_run() const override {}
};

template <>
class naive_CPU_SDDMM<float> : public SDDMMlib<float>
{
    public:
        naive_CPU_SDDMM() {}
        naive_CPU_SDDMM(ExecutionTimer* timer);
        virtual void SDDMM(
            const DenseMatrix<float>& x,
            const DenseMatrix<float>& y,
            const SparseMatrix<float>& z,
            SparseMatrix<float>& result) const override;

    private:
        void naive_CPU_SDDMM_CSR(
            const DenseMatrix<float>& x,
            const DenseMatrix<float>& y,
            const CSRMatrix<float>& z,
            CSRMatrix<float>& result) const;

        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif  // NAIVE_CPU_SDDMM_HPP