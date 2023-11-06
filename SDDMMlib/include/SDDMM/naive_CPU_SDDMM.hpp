// naive__CPU_SDDMM.hpp
#ifndef NAIVE_CPU_SDDMM_HPP
#define NAIVE_CPU_SDDMM_HPP

#include "CSRMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class naive_CPU_SDDMM : public SDDMMlib<T>
{
    public:
        naive_CPU_SDDMM() {}
        naive_CPU_SDDMM(ExecutionTimer* timer) { this->_timer = timer; }
        virtual void SDDMM(
            const DenseMatrix<T>& x,
            const DenseMatrix<T>& y,
            const SparseMatrix<T>& z,
            SparseMatrix<T>& result) const override;
        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif  // NAIVE_CPU_SDDMM_HPP