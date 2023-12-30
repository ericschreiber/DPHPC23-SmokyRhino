// semi_naive_CSR_SDDMM_GPU.cpp

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sml2_our/sml2.cuh>
#include <sml2_our/sml2_kernel.cuh>
#include <type_traits>
#include <typeinfo>

#include "utils.h"

void send_initital_B(
    cudaStream_t stream,
    float* matrixB_transpose_GPU_a,
    DenseMatrix<float>& matrixB_transpose_HOST,
    int t_j,
    int t_k,
    int k)
{
    // t_k % 4 == 0
    const float* values;
    values = matrixB_transpose_HOST.getValues();
    for (int i = 0; i < t_j; i++)
    {
        const float* temp[t_k];
        for (int j = 0; j < t_k; j++)
        {
            temp[j] = values + i * k + j;
        }
        CUDA_CHECK(
            cudaMemcpyAsync(
                matrixB_transpose_GPU_a + i * t_k,
                temp,
                t_k * sizeof(float),
                cudaMemcpyHostToDevice,
                stream));
    }
}

void send_B(
    cudaStream_t stream,
    float* matrixB_transpose_GPU_a,
    float* matrixB_transpose_GPU_b,
    DenseMatrix<float>& matrixB_transpose_HOST,
    int t_j,
    int t_k,
    int curr_col_id,  // col starts index of B
    int curr_row_id,  // row starts index of B
    int k,
    int target)
{
    // t_k % 4 == 0
    const float* values;
    values = matrixB_transpose_HOST.getValues();
    if (target % 2 == 0)
    {
        for (int i = 0; i < t_j; i++)
        {
            const float* temp[t_k];
            for (int j = 0; j < t_k; j++)
            {
                temp[j] = values + curr_col_id * k + curr_row_id + i * k + j;
            }
            CUDA_CHECK(
                cudaMemcpyAsync(
                    matrixB_transpose_GPU_a + i * t_k,
                    temp,
                    t_k * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream));
        }
    }
    else
    {
        for (int i = 0; i < t_j; i++)
        {
            const float* temp[t_k];
            for (int j = 0; j < t_k; j++)
            {
                temp[j] = values + curr_col_id * k + curr_row_id + i * k + j;
            }
            CUDA_CHECK(
                cudaMemcpyAsync(
                    matrixB_transpose_GPU_b + i * t_k,
                    temp,
                    t_k * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream));
        }
    }
}

void sml2_our<float>::SDDMM_CSR(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const CSRMatrix<float>& matrixC_HOST,
    CSRMatrix<float>& matrixResult_sparse_HOST,
    const int num_iterations) const
{
    // start the profiler
    // CUDA_CHECK(cudaProfilerStart());
    // transpose matrixB to B^t
    DenseMatrix<float> matrixB_transpose_HOST = DenseMatrix<float>(matrixB_HOST);
    matrixB_transpose_HOST.transpose();

    // get sizes of matrixA and matrixB {A=mxk; B=kxn; B_transpose=nxk}
    int m = matrixA_HOST.getNumRows();
    int k = matrixA_HOST.getNumCols();
    int n = matrixB_transpose_HOST.getNumRows();
    int nnz = matrixC_HOST.getNumValues();

    // check the dimensions of the matrices
    assert(matrixB_transpose_HOST.getNumCols() == k && "Error: matrixB_transpose has incompatible dimensions");
    assert(matrixC_HOST.getNumRows() == m && "Error: matrixC has incompatible dimensions m");
    assert(matrixC_HOST.getNumCols() == n && "Error: matrixC has incompatible dimensions n");
    assert(matrixResult_sparse_HOST.getNumRows() == m && "Error: matrixResult has incompatible dimensions m");
    assert(matrixResult_sparse_HOST.getNumCols() == n && "Error: matrixResult has incompatible dimensions n");

    // here we need some magic to define t_j, t_k, t_i and num_iterations
    int t_j = 10;
    int t_k = 4;
    int t_k_by_4 = 1;
    int t_i = 10;
    int num_iterations_t_j = 10;  // n / t_j
    int num_iterations_t_k = 10;  // k / t_k
    int curr_col_id = 0;
    int curr_row_id = 0;

    // allocate memory for the matrices on the GPU
    // _a is for the kernels 0-79, 160-239, ...
    // _b is for the remainig kernels
    float* matrixA_GPU_a;
    float* matrixA_GPU_b;
    float* matrixB_transpose_GPU_a;
    float* matrixB_transpose_GPU_b;
    float* matrixC_GPU_a;
    float* matrixC_GPU_b;
    float* matrixResult_GPU_a;
    float* matrixResult_GPU_b;
    int* col_idx_GPU_a;
    int* col_idx_GPU_b;
    int* row_ptr_GPU_a;
    int* row_ptr_GPU_b;

    CUDA_CHECK(
        cudaMalloc(
            &matrixA_GPU_a,
            80 * t_i * t_k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixA_GPU_b,
            80 * t_i * t_k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixB_transpose_GPU_a,
            t_j * t_k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixB_transpose_GPU_b,
            t_j * t_k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixC_GPU_a,
            t_i * t_j * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixC_GPU_b,
            t_i * t_j * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixResult_GPU_a,
            t_i * t_j * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixResult_GPU_b,
            t_i * t_j * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &col_idx_GPU_a,
            t_i * t_j * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &col_idx_GPU_b,
            t_i * t_j * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &row_ptr_GPU_a,
            (80 * t_i + 1) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &row_ptr_GPU_b,
            (80 * t_i + 1) * sizeof(int)));

    cudaStream_t stream_a_send, stream_b_send, stream_receive, stream_compute;
    cudaStreamCreate(&stream_a_send);
    cudaStreamCreate(&stream_b_send);
    cudaStreamCreate(&stream_receive);
    cudaStreamCreate(&stream_compute);

    // // copy matrices to the GPU
    // CUDA_CHECK(
    //     cudaMemcpy(
    //         matrixA_GPU,
    //         matrixA_HOST.getValues(),
    //         m * k * sizeof(float),
    //         cudaMemcpyHostToDevice));
    // CUDA_CHECK(
    //     cudaMemcpy(
    //         matrixB_transpose_GPU,
    //         matrixB_transpose_HOST.getValues(),
    //         n * k * sizeof(float),
    //         cudaMemcpyHostToDevice));
    // CUDA_CHECK(
    //     cudaMemcpy(
    //         matrixC_GPU,
    //         (matrixC_HOST.getValues()).data(),
    //         nnz * sizeof(float),
    //         cudaMemcpyHostToDevice));
    // CUDA_CHECK(
    //     cudaMemcpy(
    //         col_idx_GPU,
    //         (matrixC_HOST.getColIndices()).data(),
    //         nnz * sizeof(int),
    //         cudaMemcpyHostToDevice));
    // CUDA_CHECK(
    //     cudaMemcpy(
    //         row_ptr_GPU,
    //         (matrixC_HOST.getRowArray()).data(),
    //         (m + 1) * sizeof(int),
    //         cudaMemcpyHostToDevice));

    // ints to differentiate between loading to _a or _b
    int target_b = 0;
    int target_a = 0;

    // start the timer
    this->start_run();

    // transfer the memory for the first iteration
    // whole block of B
    send_B(
        stream_b_send,
        matrixB_transpose_GPU_a,
        matrixB_transpose_GPU_b,
        matrixB_transpose_HOST,
        t_j,
        t_k,
        curr_col_id,
        curr_row_id,
        k,
        target_b);

    // the correspnding blocks of A
    // m % 80 == 0; for m % 80 != 0 we need to add functionality
    for (int q = 0; q < 80; q++)
    {
        // load 1 block of A
    }

    // we want to limit the memory transfers between host and device

    for (int i = 0; i < num_iterations_t_k; i = i++)
    {
        // check that the memory transfer for this iteration is finished (B)
        cudaStreamSynchronize(stream_b_send);

        target_b++;
        curr_row_id = i * t_k;
        for (int j = 0; j < num_iterations_t_j; j = j++)
        {
            curr_col_id = j * t_j;
            // check that the memory transfer for this iteration is finished (A)
            cudaStreamSynchronize(stream_a_send);

            target_a++;
            // start memory transfer for the next iteration (not on last iteration)
            if (i != num_iterations_t_k - 1 || j != num_iterations_t_j - 1)
            {
                // load the next 80 blocks of A
                // m % 80 == 0
                for (int q = 0; q < 80; q++)
                {
                    // load 1 block of A
                }
            }
            // B can be loaded throughout all loop iterations so it only has to be started once
            if (j == 0 && i != num_iterations_t_k - 1)
            {
                // this could also be split over num_iterations_t_j iterations
                send_B(
                    stream_b_send,
                    matrixB_transpose_GPU_a,
                    matrixB_transpose_GPU_b,
                    matrixB_transpose_HOST,
                    t_j,
                    t_k,
                    curr_col_id,
                    curr_row_id,
                    k,
                    target_b);
            }

            for (int q = 0; q < 80; q++)
            {
                // Call the kernel to execute the acutal SDDMM
                compute_lml2();
            }
            // check that the memory transfer from device to host has finished
            cudaStreamSynchronize(stream_receive);

            // check that computation has finished
            cudaStreamSynchronize(stream_compute);
            // start memory transfer from device to host
        }
    }
    // wait until the last results are loaded back

    // stop the timer
    this->stop_run();

    // // copy result from the GPU to the CPU
    // float* return_values = new float[nnz];
    // CUDA_CHECK(
    //     cudaMemcpy(
    //         return_values,
    //         matrixResult_GPU,
    //         nnz * sizeof(float),
    //         cudaMemcpyDeviceToHost));

    // // Convert pointer to std::vector
    // std::vector<float> result_vector(return_values, return_values + nnz);

    // // set the result matrix
    // matrixResult_sparse_HOST.setValues(result_vector);
    // matrixResult_sparse_HOST.setColIndices(matrixC_HOST.getColIndices());
    // matrixResult_sparse_HOST.setRowArray(matrixC_HOST.getRowArray());

    // free memory on the device and destroy the handle
    CUDA_CHECK(
        cudaFree(
            matrixA_GPU_a));
    CUDA_CHECK(
        cudaFree(
            matrixA_GPU_b));
    CUDA_CHECK(
        cudaFree(
            matrixB_transpose_GPU_a));
    CUDA_CHECK(
        cudaFree(
            matrixB_transpose_GPU_b));
    CUDA_CHECK(
        cudaFree(
            matrixC_GPU_a));
    CUDA_CHECK(
        cudaFree(
            matrixC_GPU_b));
    CUDA_CHECK(
        cudaFree(
            matrixResult_GPU_a));
    CUDA_CHECK(
        cudaFree(
            matrixResult_GPU_b));
    CUDA_CHECK(
        cudaFree(
            col_idx_GPU_a));
    CUDA_CHECK(
        cudaFree(
            col_idx_GPU_b));
    CUDA_CHECK(
        cudaFree(
            row_ptr_GPU_a));
    CUDA_CHECK(
        cudaFree(
            row_ptr_GPU_b));

    // stop the profiler
    // CUDA_CHECK(cudaProfilerStop());

    return;
}

void sml2_our<float>::SDDMM(
    const DenseMatrix<float>& matrixA_HOST,
    const DenseMatrix<float>& matrixB_HOST,
    const SparseMatrix<float>& matrixC_HOST,
    SparseMatrix<float>& matrixResult_HOST,
    const int num_iterations) const
{
    const CSRMatrix<float>* csrMatrixC = dynamic_cast<const CSRMatrix<float>*>(&matrixC_HOST);
    CSRMatrix<float>* csrMatrixResult = dynamic_cast<CSRMatrix<float>*>(&matrixResult_HOST);
    if (csrMatrixC == nullptr || csrMatrixResult == nullptr)
    {
        throw std::invalid_argument("Error: convert Sparse to CSR before using this function");
    }
    else
    {
        SDDMM_CSR(
            matrixA_HOST,
            matrixB_HOST,
            *csrMatrixC,
            *csrMatrixResult,
            num_iterations);
    }
}

template <typename T>
void sml2_our<T>::SDDMM(
    const DenseMatrix<T>& matrixA_HOST,
    const DenseMatrix<T>& matrixB_HOST,
    const SparseMatrix<T>& matrixC_HOST,
    SparseMatrix<T>& matrixResult_HOST,
    const int num_iterations) const
{
    assert(false && "Error: semi_naive_CSR_SDDMM_GPU::SDDMM() only accepts float as input. Other types are not supported");
}

void sml2_our<float>::start_run() const
{
    assert(this->_timer != nullptr && "Error: semi_naive_CSR_SDDMM_GPU::start_run() timer is nullptr. Check that you have set the timer with <SDDMM>.set_timer()");
    this->_timer->start_gpu_run();
}

void sml2_our<float>::stop_run() const
{
    this->_timer->stop_gpu_run();
}

// Explicit template instantiation
// template class semi_naive_CSR_SDDMM_GPU<float>;
template class sml2_our<double>;
template class sml2_our<int>;
