// naive_sequential_full_SDDMM_HOST.hpp
#ifndef NAIVE_SEQUENTIAL_FULL_SDDMM_HOST_HPP
#define NAIVE_SEQUENTIAL_FULL_SDDMM_HOST_HPP

#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_sequential_full_SDDMM_HOST : public SDDMMlib<T>
{
    public:
        naive_sequential_full_SDDMM_HOST() { assert(false && "Error: naive_sequential_full_SDDMM_HOST<T>::naive_SDDMM() is only implemented for float."); }
        naive_sequential_full_SDDMM_HOST(ExecutionTimer* timer) { assert(false && "Error: naive_sequential_full_SDDMM_HOST<T>::naive_SDDMM() is only implemented for float."); }
        
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            const SparseMatrix<T>& z,
            SparseMatrix<T>& result) const override;
        virtual void start_run() const override {}  // Would need to be implemented but we don't need it because the class can never be constructed except for float
        virtual void stop_run() const override {}
};

template <>
class naive_sequential_full_SDDMM_HOST<float> : public SDDMMlib<float>
{
    public:
        naive_sequential_full_SDDMM_HOST() {}
        naive_sequential_full_SDDMM_HOST(ExecutionTimer* timer) { this->_timer = timer; }
        virtual void SDDMM(
            const DenseMatrix<float>& x,
            const DenseMatrix<float>& y,
            const SparseMatrix<float>& z,
            SparseMatrix<float>& result) const override;
        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE

    private:
        void naive_sequential_full_SDDMM_HOST_CSR(
            const DenseMatrix<float>& x,
            const DenseMatrix<float>& y,
            const CSRMatrix<float>& z,
            CSRMatrix<float>& result) const;
};

#endif  // NAIVE_SEQUENTIAL_FULL_SDDMM_HOST_HPP