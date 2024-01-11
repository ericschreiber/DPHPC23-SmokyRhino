// SM_L2_GPU.cpp
#include "SM_L2/SM_L2_GPU.hpp"

#include <iostream>
#include <type_traits>
#include <typeinfo>

#include "SM_L2/SM_L2.cuh"
#include "utils.h"

// IMPORTANT: This methode only works once as it overwrites the input matrix S (should be resolved)
// IMPORTANT: Works only for K multiple of 32

// Their code in file sddmm.cu
void sm_l2_SDDMM_GPU<float>::sddmm_SM_L2_GPU(const Matrix S, TiledMatrix tS, float* P, vector<float> W, vector<float> H, int num_iterations, int k) const
{
    float *d_val, *d_W, *d_H, *d_W_t, *val_out;
    int *d_row_ptr, *d_col_ind, *d_row_ind, *d_tiled_ind, *d_lastIdx,
        *d_active_row, *d_lastIdx_block_tile, *d_passive_row;

    // Copy the dense matrices float4 aligned. E.g. pad them with zeros to be a multiple of 4
    int k_aligned = k;
    if (k % 4 != 0)
    {
        k_aligned = k + (4 - (k % 4));
    }
    assert(k_aligned % 4 == 0 && "Error: k_aligned is not a multiple of 4");

    //***********Starting GPU****************
    CUDA_CHECK(cudaMalloc((void**)&d_W, k_aligned * S.n_rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_H, k_aligned * S.n_cols * sizeof(float)));
    // CUDA_CHECK(cudaMalloc((void**)&d_row_ptr, (n_rows+1) * sizeof (int)),2);
    CUDA_CHECK(cudaMalloc((void**)&d_row_ind, tS.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_col_ind, tS.nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_val, tS.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&val_out, tS.nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_lastIdx, (tS.ntile_c + 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_active_row, tS.ntile_c * tS.max_active_row * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_lastIdx_block_tile, tS.ntile_c * tS.max_active_block * sizeof(int)));

    // CUDA_CHECK(cudaMemcpy(d_row_ptr,  &(row_ptr[0]), (n_rows+1) * sizeof (int),
    // cudaMemcpyHostToDevice),4);
    CUDA_CHECK(cudaMemcpy(d_row_ind, &(tS.rows[0]), tS.nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_ind, &(tS.cols[0]), tS.nnz * sizeof(int), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(d_val, &(new_vals[0]), tS.nnz * sizeof (float),
    // cudaMemcpyHostToDevice),4);
    cudaMemset(d_val, 0, S.nnz * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_lastIdx, &(tS.lastIdx_tile[0]), (tS.ntile_c + 1) * sizeof(int), cudaMemcpyHostToDevice));
    for (int i = 0; i < tS.ntile_c; ++i)
    {
        CUDA_CHECK(
            cudaMemcpy(d_lastIdx_block_tile + i * tS.max_active_block, &(tS.lastIdx_block_tile[i * tS.max_active_block]), tS.max_active_block * sizeof(int), cudaMemcpyHostToDevice));
        // cout <<i<<" "<< tS.lastIdx_tile[i]<<"
        // "<<tS.lastIdx_block_tile[i*tS.max_active_block]<< endl;
    }

    int sum = 0;
    for (int i = 0; i < tS.ntile_c; ++i)
    {
        CUDA_CHECK(
            cudaMemcpy(d_active_row + sum, &(tS.active_row[i * S.n_rows]), tS.n_actv_row[i] * sizeof(int), cudaMemcpyHostToDevice));
        sum += tS.n_actv_row[i];
    }
    // sum=0;
    // for (int i = 0; i < tS.ntile_c; ++i){
    //     CUDA_CHECK(cudaMemcpy(d_passive_row+sum, &(passive_row[i*S.n_rows]),
    //     S.n_rows * sizeof (int), cudaMemcpyHostToDevice),4); sum += S.n_rows;
    // }

    // cudaMemcpy(d_W, &(W[0]), S.n_rows * k * sizeof(float), cudaMemcpyHostToDevice);
    // // cudaMemcpy(d_W, &(W_t[0]),  S.n_rows * k  * sizeof (float),
    // // cudaMemcpyHostToDevice);
    // cudaMemcpy(d_H, &(H[0]), S.n_cols * k * sizeof(float), cudaMemcpyHostToDevice);
    // // cudaMemcpy(d_H, &(H_t[0]),  S.n_cols * k  * sizeof (float),
    // // cudaMemcpyHostToDevice);
    // // std::cout << "Done copying to GPU" << std::endl;
    // Copy float4 the dense matrices
    for (int i = 0; i < S.n_rows; i++)
    {
        float temp[k_aligned];
        for (int j = 0; j < k; j++)
        {
            temp[j] = W[i * k + j];
        }
        for (int j = k; j < k_aligned; j++)
        {
            temp[j] = 0;
        }
        CUDA_CHECK(
            cudaMemcpy(
                d_W + i * k_aligned,
                temp,
                k_aligned * sizeof(float),
                cudaMemcpyHostToDevice));
    }
    for (int i = 0; i < S.n_cols; i++)
    {
        float temp[k_aligned];
        for (int j = 0; j < k; j++)
        {
            temp[j] = H[i * k + j];
        }
        for (int j = k; j < k_aligned; j++)
        {
            temp[j] = 0;
        }
        CUDA_CHECK(
            cudaMemcpy(
                d_H + i * k_aligned,
                temp,
                k_aligned * sizeof(float),
                cudaMemcpyHostToDevice));
    }

    int n_tile = tS.ntile_c;  // S.n_cols/tile_sizeX + 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < num_iterations; i++)
    {
        this->start_run();

        cudaStream_t stream[n_tile];
        for (int i = 0; i < n_tile; i++)
        {
            cudaStreamCreate(&(stream[i]));
        }

        compute_sm_l2(
            this->BLOCKSIZE,
            this->SM_CAPACITY,
            this->actv_row_size,
            n_tile,
            S,
            tS,
            // k,
            k_aligned,
            stream,
            d_row_ind,
            d_col_ind,
            d_val,
            val_out,
            d_W,
            d_H,
            d_active_row,
            d_lastIdx_block_tile);
        this->stop_run();
    }

    CUDA_CHECK(cudaMemcpy(&(P[0]), val_out, tS.nnz * sizeof(float), cudaMemcpyDeviceToHost));

    // freeing device allocation
    // CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_row_ind));
    CUDA_CHECK(cudaFree(d_col_ind));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(val_out));
    CUDA_CHECK(cudaFree(d_active_row));
    // CUDA_CHECK(cudaFree(d_passive_row));
    CUDA_CHECK(cudaFree(d_lastIdx_block_tile));
    CUDA_CHECK(cudaFree(d_lastIdx));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_H));
}

void sm_l2_SDDMM_GPU<float>::SDDMM_COO(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const COOMatrix<float>& matrixC_HOST,
    COOMatrix<float>& matrixResult_HOST,
    const int num_iterations) const
{
    // Get all the sizes (A=mxk; B=kxn; C=mxn; Result=mxn)
    int m = matrixA_HOST.getNumRows();
    int k = matrixA_HOST.getNumCols();
    int n = matrixB_HOST.getNumCols();
    int numElementsC = matrixC_HOST.getValues().size();

    // check the dimensions of the matrices s.t. we can multiply them
    assert(matrixB_HOST.getNumRows() == k && "Error: matrixB has incompatible dimensions");
    assert(matrixC_HOST.getNumRows() == m && "Error: matrixC has incompatible dimensions m");
    assert(matrixC_HOST.getNumCols() == n && "Error: matrixC has incompatible dimensions n");
    assert(matrixResult_HOST.getNumRows() == m && "Error: matrixResult has incompatible dimensions m");
    assert(matrixResult_HOST.getNumCols() == n && "Error: matrixResult has incompatible dimensions n");

    DenseMatrix<float> matrixB_transpose_HOST(matrixB_HOST);
    matrixB_transpose_HOST.transpose();

    // Convert the matrices to their format
    Matrix S = Matrix();
    S.n_rows = m;
    S.n_cols = n;
    S.nnz = numElementsC;
    S.rows = matrixC_HOST.getRowArray();
    S.cols = matrixC_HOST.getColIndices();
    S.vals = matrixC_HOST.getValues();

    TiledMatrix tiledS(S, this->tile_sizeX, this->tile_sizeY, this->actv_row_size, this->BLOCKSIZE);
    tiledS.nnz = 0;

    // convert the matrix to CSR in their format
    int* row_ptr = new int[S.n_rows + 1];
    int* row_holder = new int[S.n_rows];
    // Fill arrays with 0
    std::fill(row_ptr, row_ptr + S.n_rows + 1, 0);
    std::fill(row_holder, row_holder + S.n_rows, 0);
    make_CSR(S.rows, S.cols, S.vals, S.nnz, S.n_rows, row_ptr, row_holder);

    tiledS.max_active_row =
        rewrite_matrix_1D(S, tiledS, row_ptr, this->tile_sizeX, row_holder, this->actv_row_size);

    // result matrix
    float* P = new float[S.nnz];

    sddmm_SM_L2_GPU(S, tiledS, P, std::vector<float>(matrixA_HOST.getValues(), matrixA_HOST.getValues() + matrixA_HOST.getNumRows() * matrixA_HOST.getNumCols()), std::vector<float>(matrixB_transpose_HOST.getValues(), matrixB_transpose_HOST.getValues() + matrixB_transpose_HOST.getNumRows() * matrixB_transpose_HOST.getNumCols()), num_iterations, k);

    // Build the result matrix
    matrixResult_HOST.setValues(std::vector<float>(P, P + S.nnz));
    std::vector<int> row_indices;
    std::vector<int> col_indices;

    // Copy from CSR to COO
    for (int i = 0; i < S.n_rows; i++)
    {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            row_indices.push_back(i);
            col_indices.push_back(S.cols[j]);
        }
    }
    matrixResult_HOST.setRowArray(row_indices);
    matrixResult_HOST.setColIndices(col_indices);

    return;
}

void sm_l2_SDDMM_GPU<float>::SDDMM(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const SparseMatrix<float>& matrixC_HOST,
    SparseMatrix<float>& matrixResult_HOST,
    const int num_iterations) const
{
    // Print a warning that this function is incorrect but the same as the paper
    std::cout << "Warning: This function is incorrect but the same as the paper" << std::endl;

    const COOMatrix<float>* cooMatrixC = dynamic_cast<const COOMatrix<float>*>(&matrixC_HOST);
    COOMatrix<float>* cooMatrixResult = dynamic_cast<COOMatrix<float>*>(&matrixResult_HOST);
    if (cooMatrixC == nullptr || cooMatrixResult == nullptr)
    {
        throw std::invalid_argument("Error: convert Sparse to COO before using this function");
    }
    else
    {
        SDDMM_COO(
            matrixA_HOST,
            matrixB_HOST,
            *cooMatrixC,
            *cooMatrixResult,
            num_iterations);
    }

    cooMatrixC = nullptr;
    cooMatrixResult = nullptr;

    return;
}

void sm_l2_SDDMM_GPU<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: sm_l2_SDDMM_GPU::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_gpu_run();
}

void sm_l2_SDDMM_GPU<float>::stop_run() const
{
    this->_timer->stop_gpu_run();
}

// Explicit template instantiation
// template class sm_l2_SDDMM_GPU<float>;
template class sm_l2_SDDMM_GPU<double>;
template class sm_l2_SDDMM_GPU<int>;