// SDDMMlib.hpp
#ifndef SDDMMLIB_HPP
#define SDDMMLIB_HPP

#include "DenseMatrix.hpp"
#include "ExecutionTimer.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class SDDMMlib
{
    public:
        virtual void SDDMM(
            const DenseMatrix<T> &x,
            const DenseMatrix<T> &y,
            const SparseMatrix<T> &z,
            SparseMatrix<T> &result) const = 0;
        virtual void start_run() const = 0;  // Start either cpu or gpu run
        virtual void stop_run() const = 0;   // Stop either cpu or gpu run
        void set_timer(ExecutionTimer *timer)
        {
            _timer = timer;
        }

    protected:
        ExecutionTimer *_timer;
};

#endif  // SDDMMLIB_HPP