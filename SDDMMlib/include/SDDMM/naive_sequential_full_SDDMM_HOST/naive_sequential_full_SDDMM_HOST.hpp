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
        virtual void SDDMM(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const SparseMatrix<float>& matrixC_HOST,
            SparseMatrix<float>& matrixResult_HOST) const override;

    private:
        
        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif  // NAIVE_SEQUENTIAL_FULL_SDDMM_HOST_HPP