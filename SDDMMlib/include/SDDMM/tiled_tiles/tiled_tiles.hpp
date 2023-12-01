#ifndef TILED_TILES_HPP
#define TILED_TILES_HPP

#include <cassert>
#include <type_traits>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class tiled_tiles : public SDDMMlib<T>
{
    public:
        tiled_tiles() { assert(false && "Error: tiled_tiles<T>::tiled_tiles() is only implemented for float."); }
        tiled_tiles(ExecutionTimer* timer) { assert(false && "Error: tiled_tiles<T>::tiled_tiles() is only implemented for float."); }
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
class tiled_tiles<float> : public SDDMMlib<float>
{
    public:
        tiled_tiles() {}
        tiled_tiles(ExecutionTimer* timer) { this->_timer = timer; }
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
