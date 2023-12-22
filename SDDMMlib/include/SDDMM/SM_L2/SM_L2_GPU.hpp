// SM_L2_GPU.hpp
#ifndef SM_L2_GPU_HPP
#define SM_L2_GPU_HPP

#include <cassert>
#include <type_traits>

#include "COOMatrix.hpp"
#include "DenseMatrix.hpp"
#include "SDDMMlib.hpp"
#include "SparseMatrix.hpp"

template <typename T>
class sm_l2_SDDMM_GPU : public SDDMMlib<T>
{
    public:
        sm_l2_SDDMM_GPU() { assert(false && "Error: sm_l2_SDDMM_GPU<T>::sm_l2_SDDMM_GPU() is only implemented for float."); }
        sm_l2_SDDMM_GPU(ExecutionTimer* timer) { assert(false && "Error: sm_l2_SDDMM_GPU<T>::sm_l2_SDDMM_GPU() is only implemented for float."); }
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
class sm_l2_SDDMM_GPU<float> : public SDDMMlib<float>
{
    public:
        sm_l2_SDDMM_GPU() {}
        sm_l2_SDDMM_GPU(ExecutionTimer* timer) { this->_timer = timer; }
        virtual void SDDMM(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const SparseMatrix<float>& matrixC_HOST,
            SparseMatrix<float>& matrixResult_HOST,
            const int num_iterations) const override;

    private:
        int tile_sizeX = 256;
        int tile_sizeY = 25000;
        int SM_CAPACITY = 12288;
        int BLOCKSIZE = 512;
        int actv_row_size = 180;

        void
        SDDMM_COO(
            const DenseMatrix<float>& matrixA_HOST,
            const DenseMatrix<float>& matrixB_HOST,
            const COOMatrix<float>& matrixC_HOST,
            COOMatrix<float>& matrixResult_HOST,
            const int num_iterations) const;

        virtual void start_run() const override;  // Start either cpu or gpu run CHOOSE ONE
        virtual void stop_run() const override;   // Stop either cpu or gpu run CHOOSE ONE
};

#endif  // SM_L2_GPU_HPP
