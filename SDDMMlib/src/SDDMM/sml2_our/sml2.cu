// semi_naive_CSR_SDDMM_GPU.cpp

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <iostream>
#include <sml2_our/sml2.cuh>
#include <sml2_our/sml2_kernel.cuh>
#include <type_traits>
#include <typeinfo>

#include "utils.h"

void send_B(
    cudaStream_t stream,
    float* matrixB_transpose_GPU_a,
    float* matrixB_transpose_GPU_b,
    const float* values,
    int t_j,
    int t_k,
    int col_id,  // col starts index of B
    int row_id,  // row starts index of B
    int k,
    int target)
{
    // t_k % 4 == 0
    if (target % 2 == 0)
    {
        for (int i = 0; i < t_j; i++)
        {
            const float* temp[t_k];
            for (int j = 0; j < t_k; j++)
            {
                temp[j] = values + col_id * k + row_id + i * k + j;
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
                temp[j] = values + col_id * k + row_id + i * k + j;
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

void send_A(
    cudaStream_t stream,
    float* matrixA_GPU_a,
    float* matrixA_GPU_b,
    const float* values,
    int t_i,
    int t_k,
    int col_id,  // col starts index of A - curr_row_id * t_k
    int row_id,  // row starts index of A - curr_t_i_id * t_i
    int k,
    int target)
{
    // t_k % 4 == 0
    if (target % 2 == 0)
    {
        for (int i = 0; i < t_i; i++)
        {
            const float* temp[t_k];
            for (int j = 0; j < t_k; j++)
            {
                temp[j] = values + row_id * k + col_id + i * k + j;
            }
            CUDA_CHECK(
                cudaMemcpyAsync(
                    matrixA_GPU_a + i * t_k,
                    temp,
                    t_k * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream));
        }
    }
    else
    {
        for (int i = 0; i < t_i; i++)
        {
            const float* temp[t_k];
            for (int j = 0; j < t_k; j++)
            {
                temp[j] = values + row_id * k + col_id + i * k + j;
            }
            CUDA_CHECK(
                cudaMemcpyAsync(
                    matrixA_GPU_b + i * t_k,
                    temp,
                    t_k * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream));
        }
    }
}

void send_row_ptr_and_col_id(
    cudaStream_t stream_rp,
    cudaStream_t stream_ci,
    int* row_ptr_GPU_a,
    int* row_ptr_GPU_b,
    int* col_idx_GPU_a,
    int* col_idx_GPU_b,
    int* num_nnz_a,
    int* num_nnz_b,
    const int* row_ptr,
    const int* col_idx,
    int t_i,
    int t_j,
    int row_id,
    int col_id,
    int target)
{
}

void send_C(
    cudaStream_t stream,
    float* matrixC_GPU_a,
    float* matrixC_GPU_b,
    const float* values,
    const int* row_ptr_a,
    const int* row_ptr_b,
    const int* col_idx_a,
    const int* col_idx_b,
    int* num_nnz_a,
    int* num_nnz_b,
    int target)
{
}

void send_result(
    cudaStream_t stream,
    float* matrixResult_GPU_a,
    float* matrixResult_GPU_b,
    float* result_from_gpu,
    int* num_nnz_a,
    int* num_nnz_b,
    int t_i,
    int target)
{
    if (target % 2 == 0)
    {
        int nnz = 0;
        for (int i = 0; i < 80 * t_i; i++)
        {
            nnz += num_nnz_a[i];
        }
        CUDA_CHECK(
            cudaMemcpyAsync(
                result_from_gpu,
                matrixResult_GPU_a,
                nnz * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream));
    }
    else
    {
        int nnz = 0;
        for (int i = 0; i < 80 * t_i; i++)
        {
            nnz += num_nnz_b[i];
        }
        CUDA_CHECK(
            cudaMemcpyAsync(
                result_from_gpu,
                matrixResult_GPU_b,
                nnz * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream));
    }
}

void save_result(
    float* result_from_gpu,
    float* result_HOST,
    int* row_ptr_a,
    int* row_ptr_b,
    int* col_idx_a,
    int* col_idx_b,
    int* num_nnz_a,
    int* num_nnz_b,
    int target)
{
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
    int t_k = 4;  // this probably has to be around 16 for p=1% to fit everything on the GPU
    int t_k_by_4 = 1;
    int t_i = 10;
    int num_iterations_t_j = 10;  // n / t_j
    int num_iterations_t_k = 10;  // k / t_k
    int num_iterations_t_i = 10;  // m / t_i
    int curr_col_id = 0;
    int curr_row_id = 0;
    int curr_t_i_id = 0;
    float p = 0.01;  // density of matrixC

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
    int* num_nnz_GPU_a;  // number of non-zero elements per row for iteration a
    int* num_nnz_GPU_b;  // number of non-zero elements per row for iteration b

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
            80 * 10 * p * t_i * t_j * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixC_GPU_b,
            80 * 10 * p * t_i * t_j * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixResult_GPU_a,
            80 * 10 * p * t_i * t_j * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixResult_GPU_b,
            80 * 10 * p * t_i * t_j * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &col_idx_GPU_a,
            80 * 10 * p * t_i * t_j * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &col_idx_GPU_b,
            80 * 10 * p * t_i * t_j * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &row_ptr_GPU_a,
            (80 * t_i + 1) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &row_ptr_GPU_b,
            (80 * t_i + 1) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &num_nnz_GPU_a,
            80 * t_i * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &num_nnz_GPU_b,
            80 * t_i * sizeof(int)));

    cudaStream_t stream_a_send, stream_b_send, stream_receive, stream_compute;
    cudaStream_t stream_c_send, stream_rp_send, stream_ci_send, stream_set_zero;
    cudaStreamCreate(&stream_a_send);
    cudaStreamCreate(&stream_b_send);
    cudaStreamCreate(&stream_receive);
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_c_send);
    cudaStreamCreate(&stream_rp_send);
    cudaStreamCreate(&stream_ci_send);
    cudaStreamCreate(&stream_set_zero);

    // ints to differentiate between loading to _a or _b
    int target_b = 0;
    int target_a = 0;

    // save row_ptr and col_idx on the host
    int* row_ptr_HOST_a = new int[80 * t_i + 1];
    int* row_ptr_HOST_b = new int[80 * t_i + 1];
    int* num_nnz_a = new int[80 * t_i];
    int* num_nnz_b = new int[80 * t_i];
    int* col_idx_HOST_a = new int[80 * 10 * p * t_i * t_j];
    int* col_idx_HOST_b = new int[80 * 10 * p * t_i * t_j];

    // create memory for the result on the host
    float* result_from_gpu = new float[10 * p * t_i * t_j];

    // local copy of values of all matrices
    const float* values_A = matrixA_HOST.getValues();
    const float* values_B = matrixB_transpose_HOST.getValues();
    const float* values_C = matrixC_HOST.getValues().data();
    const int* col_idx_C = matrixC_HOST.getColIndices().data();
    const int* row_ptr_C = matrixC_HOST.getRowArray().data();
    float* values_result = new float[nnz];

    // start the timer
    this->start_run();

    // transfer the memory for the first iteration
    // whole block of B
    send_B(
        stream_b_send,
        matrixB_transpose_GPU_a,
        matrixB_transpose_GPU_b,
        values_B,
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
        send_A(
            stream_a_send,
            matrixA_GPU_a,
            matrixA_GPU_b,
            values_A,
            t_i,
            t_k,
            curr_row_id * t_k,
            curr_t_i_id * t_i,
            k,
            target_a);
        curr_t_i_id++;
    }

    // set result to zero
    cudaMemsetAsync(
        matrixResult_GPU_a,
        0,
        10 * p * t_i * t_j * sizeof(float),
        stream_set_zero);

    // set initial row_ptr and col_idx
    send_row_ptr_and_col_id(
        stream_rp_send,
        stream_ci_send,
        row_ptr_GPU_a,
        row_ptr_GPU_b,
        col_idx_GPU_a,
        col_idx_GPU_b,
        num_nnz_a,
        num_nnz_b,
        row_ptr_C,
        col_idx_C,
        t_i,
        t_j,
        curr_row_id,
        curr_col_id,
        target_a);

    // set initial matrixC
    send_C(
        stream_c_send,
        matrixC_GPU_a,
        matrixC_GPU_b,
        values_C,
        row_ptr_HOST_a,
        row_ptr_HOST_b,
        col_idx_HOST_a,
        col_idx_HOST_b,
        num_nnz_a,
        num_nnz_b,
        target_a);

    for (int i = 0; i < num_iterations_t_k; i = i++)
    {
        // check that the memory transfer for this iteration is finished (B)
        cudaStreamSynchronize(stream_b_send);

        // curr_row_id = i * t_k;
        for (int j = 0; j < num_iterations_t_j; j = j++)
        {
            // curr_col_id = j * t_j;
            //  B can be loaded throughout all loop iterations so it only has to be started once
            target_b++;
            if (j == 0 && i != num_iterations_t_k - 1)
            {
                // this could also be split over num_iterations_t_j iterations
                send_B(
                    stream_b_send,
                    matrixB_transpose_GPU_a,
                    matrixB_transpose_GPU_b,
                    values_B,
                    t_j,
                    t_k,
                    curr_col_id,
                    curr_row_id,
                    k,
                    target_b);
            }

            for (int w = 0; w < num_iterations_t_i; w++)
            {
                // check that the memory transfer for this iteration is finished (A, row_ptr, col_idx, C, set_to_zero)
                cudaStreamSynchronize(stream_a_send);
                cudaStreamSynchronize(stream_rp_send);
                cudaStreamSynchronize(stream_ci_send);
                cudaStreamSynchronize(stream_c_send);
                cudaStreamSynchronize(stream_set_zero);

                for (int q = 0; q < 80; q++)
                {
                    // Call the kernel to execute the acutal SDDMM
                    compute_lml2();
                }

                target_a++;
                // start memory transfer for the next iteration (not on last iteration)
                if (i != num_iterations_t_k - 1 || j != num_iterations_t_j - 1 || w != num_iterations_t_i - 1)
                {
                    // check if we need to start at the top again
                    if (w == num_iterations_t_i - 1)
                    {
                        curr_t_i_id = 0;
                        if (j == num_iterations_t_j - 1)
                        {
                            curr_col_id = 0;
                            curr_row_id += t_k;
                        }
                        else
                        {
                            curr_col_id += t_j;
                        }
                    }

                    // load the next 80 blocks of A
                    // m % 80 == 0
                    for (int q = 0; q < 80; q++)
                    {
                        // load 1 block of A
                        send_A(
                            stream_a_send,
                            matrixA_GPU_a,
                            matrixA_GPU_b,
                            values_A,
                            t_i,
                            t_k,
                            curr_row_id * t_k,
                            curr_t_i_id * t_i,
                            k,
                            target_a);
                        curr_t_i_id++;
                    }

                    // load the next row_ptr and col_idx
                    send_row_ptr_and_col_id(
                        stream_rp_send,
                        stream_ci_send,
                        row_ptr_GPU_a,
                        row_ptr_GPU_b,
                        col_idx_GPU_a,
                        col_idx_GPU_b,
                        num_nnz_a,
                        num_nnz_b,
                        row_ptr_C,
                        col_idx_C,
                        t_i,
                        t_j,
                        curr_row_id,
                        curr_col_id,
                        target_a);

                    // load the next matrixC
                    send_C(
                        stream_c_send,
                        matrixC_GPU_a,
                        matrixC_GPU_b,
                        values_C,
                        row_ptr_HOST_a,
                        row_ptr_HOST_b,
                        col_idx_HOST_a,
                        col_idx_HOST_b,
                        num_nnz_a,
                        num_nnz_b,
                        target_a);
                }

                // save the result on the host from the previous iteration (not on first iteration)
                if (i != 0 || j != 0 || w != 0)
                {
                    // check that the memory transfer from device to host has finished
                    cudaStreamSynchronize(stream_receive);
                    // save the result on the host
                    save_result(
                        result_from_gpu,
                        values_result,
                        row_ptr_HOST_a,
                        row_ptr_HOST_b,
                        col_idx_HOST_a,
                        col_idx_HOST_b,
                        num_nnz_a,
                        num_nnz_b,
                        target_a);
                }

                // not sure if this is needed - probably not
                if (i % target_a == 0)
                {
                    cudaMemsetAsync(
                        matrixResult_GPU_b,
                        0,
                        10 * p * t_i * t_j * sizeof(float),
                        stream_set_zero);
                }
                else
                {
                    cudaMemsetAsync(
                        matrixResult_GPU_a,
                        0,
                        10 * p * t_i * t_j * sizeof(float),
                        stream_set_zero);
                }

                // check that computation has finished
                cudaStreamSynchronize(stream_compute);
                // start memory transfer from device to host
                send_result(
                    stream_receive,
                    matrixResult_GPU_a,
                    matrixResult_GPU_b,
                    result_from_gpu,
                    num_nnz_a,
                    num_nnz_b,
                    t_i,
                    target_a);
            }
        }
    }
    // wait until the last results are loaded back
    cudaStreamSynchronize(stream_receive);
    // save the last result on the host
    save_result(
        result_from_gpu,
        values_result,
        row_ptr_HOST_a,
        row_ptr_HOST_b,
        col_idx_HOST_a,
        col_idx_HOST_b,
        num_nnz_a,
        num_nnz_b,
        target_a);

    // stop the timer
    this->stop_run();

    // set the result matrix
    matrixResult_sparse_HOST.setValues(std::vector<float>(values_result, values_result + nnz));
    matrixResult_sparse_HOST.setColIndices(matrixC_HOST.getColIndices());
    matrixResult_sparse_HOST.setRowArray(matrixC_HOST.getRowArray());

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
    CUDA_CHECK(
        cudaFree(
            num_nnz_GPU_a));
    CUDA_CHECK(
        cudaFree(
            num_nnz_GPU_b));
    delete[] row_ptr_HOST_a;
    delete[] row_ptr_HOST_b;
    delete[] col_idx_HOST_a;
    delete[] col_idx_HOST_b;
    delete[] values_A;
    delete[] values_B;
    delete[] values_C;
    delete[] col_idx_C;
    delete[] row_ptr_C;
    delete[] result_from_gpu;
    delete[] values_result;
    delete[] num_nnz_a;
    delete[] num_nnz_b;

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
