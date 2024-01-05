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
    cudaStream_t stream_a,
    cudaStream_t stream_b,
    float* matrixB_transpose_GPU_a,
    float* matrixB_transpose_GPU_b,
    const float* values,
    int t_j,
    int t_k,
    int row_id,  // of B_T
    int col_id,  // of B_T
    int k,
    int target)
{
    // std::cout << "row_id=" << row_id << " | col_id=" << col_id << " | target=" << target << " | k=" << k << std::endl;
    //  t_k % 4 == 0
    if (target % 2 == 0)
    {
        if (col_id + t_k > k)
        {
            for (int i = 0; i < t_j; i++)
            {
                float temp[t_k];
                for (int j = 0; j < k - col_id; j++)
                {
                    temp[j] = values[row_id * k + col_id + i * k + j];
                }
                for (int j = k - col_id; j < t_k; j++)
                {
                    temp[j] = 0;
                }

                // std::cout << "temp" << std::endl;
                // for (int j = 0; j < t_k; j++)
                // {
                //     std::cout << temp[j] << " ";
                // }
                // std::cout << std::endl;

                CUDA_CHECK(
                    cudaMemcpyAsync(
                        matrixB_transpose_GPU_a + i * t_k,
                        temp,
                        t_k * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_a));
            }
        }
        else
        {
            for (int i = 0; i < t_j; i++)
            {
                float temp[t_k];
                for (int j = 0; j < t_k; j++)
                {
                    temp[j] = values[row_id * k + col_id + i * k + j];
                }

                // std::cout << "temp" << std::endl;
                // for (int j = 0; j < t_k; j++)
                // {
                //     std::cout << temp[j] << " ";
                // }
                // std::cout << std::endl;

                CUDA_CHECK(
                    cudaMemcpyAsync(
                        matrixB_transpose_GPU_a + i * t_k,
                        temp,
                        t_k * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_a));
            }
        }
    }
    else
    {
        if (col_id + t_k > k)
        {
            for (int i = 0; i < t_j; i++)
            {
                float temp[t_k];
                for (int j = 0; j < k - col_id; j++)
                {
                    temp[j] = values[row_id * k + col_id + i * k + j];
                }
                for (int j = k - col_id; j < t_k; j++)
                {
                    temp[j] = 0;
                }

                // std::cout << "temp" << std::endl;
                // for (int j = 0; j < t_k; j++)
                // {
                //     std::cout << temp[j] << " ";
                // }
                // std::cout << std::endl;

                CUDA_CHECK(
                    cudaMemcpyAsync(
                        matrixB_transpose_GPU_b + i * t_k,
                        temp,
                        t_k * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_b));
            }
        }
        else
        {
            for (int i = 0; i < t_j; i++)
            {
                float temp[t_k];
                for (int j = 0; j < t_k; j++)
                {
                    temp[j] = values[row_id * k + col_id + i * k + j];
                }

                // std::cout << "temp" << std::endl;
                // for (int j = 0; j < t_k; j++)
                // {
                //     std::cout << temp[j] << " ";
                // }
                // std::cout << std::endl;

                CUDA_CHECK(
                    cudaMemcpyAsync(
                        matrixB_transpose_GPU_b + i * t_k,
                        temp,
                        t_k * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_b));
            }
        }
    }
}

void send_A(
    cudaStream_t stream_a,
    cudaStream_t stream_b,
    float* matrixA_GPU_a,
    float* matrixA_GPU_b,
    const float* values,
    int t_i,
    int t_k,
    int col_id,  // col starts index of A - curr_col_id
    int row_id,  // row starts index of A - curr_t_i_id * t_i
    int row_GPU,
    int k,
    int target)
{
    // t_k % 4 == 0
    if (target % 2 == 0)
    {
        if (col_id + t_k > k)
        {
            for (int i = 0; i < t_i; i++)
            {
                float temp[t_k];
                for (int j = 0; j < k - col_id; j++)
                {
                    temp[j] = values[row_id * k + col_id + i * k + j];
                }
                for (int j = k - col_id; j < t_k; j++)
                {
                    temp[j] = 0;
                }

                // std::cout << "temp" << std::endl;
                // for (int j = 0; j < t_k; j++)
                // {
                //     std::cout << temp[j] << " ";
                // }
                // std::cout << std::endl;

                CUDA_CHECK(
                    cudaMemcpyAsync(
                        matrixA_GPU_a + row_GPU * t_k + i * t_k,
                        temp,
                        t_k * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_a));
            }
        }
        else
        {
            for (int i = 0; i < t_i; i++)
            {
                float temp[t_k];
                for (int j = 0; j < t_k; j++)
                {
                    temp[j] = values[row_id * k + col_id + i * k + j];
                }

                // std::cout << "temp" << std::endl;
                // for (int j = 0; j < t_k; j++)
                // {
                //     std::cout << temp[j] << " ";
                // }
                // std::cout << std::endl;

                CUDA_CHECK(
                    cudaMemcpyAsync(
                        matrixA_GPU_a + row_GPU * t_k + i * t_k,
                        temp,
                        t_k * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_a));
            }
        }
    }
    else
    {
        if (col_id + t_k > k)
        {
            for (int i = 0; i < t_i; i++)
            {
                float temp[t_k];
                for (int j = 0; j < k - col_id; j++)
                {
                    temp[j] = values[row_id * k + col_id + i * k + j];
                }
                for (int j = k - col_id; j < t_k; j++)
                {
                    temp[j] = 0;
                }

                // std::cout << "temp" << std::endl;
                // for (int j = 0; j < t_k; j++)
                // {
                //     std::cout << temp[j] << " ";
                // }
                // std::cout << std::endl;

                CUDA_CHECK(
                    cudaMemcpyAsync(
                        matrixA_GPU_b + row_GPU * t_k + i * t_k,
                        temp,
                        t_k * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_b));
            }
        }
        else
        {
            for (int i = 0; i < t_i; i++)
            {
                float temp[t_k];
                for (int j = 0; j < t_k; j++)
                {
                    temp[j] = values[row_id * k + col_id + i * k + j];
                }

                // std::cout << "temp" << std::endl;
                // for (int j = 0; j < t_k; j++)
                // {
                //     std::cout << temp[j] << " ";
                // }
                // std::cout << std::endl;

                CUDA_CHECK(
                    cudaMemcpyAsync(
                        matrixA_GPU_b + row_GPU * t_k + i * t_k,
                        temp,
                        t_k * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream_b));
            }
        }
    }
}

void send_row_ptr_and_col_id(
    cudaStream_t stream_rp_a,
    cudaStream_t stream_ci_a,
    cudaStream_t stream_nnz_a,
    cudaStream_t stream_rp_b,
    cudaStream_t stream_ci_b,
    cudaStream_t stream_nnz_b,
    int* row_ptr_GPU_a,
    int* row_ptr_GPU_b,
    int* col_idx_GPU_a,
    int* col_idx_GPU_b,
    int* num_nnz_GPU_a,
    int* num_nnz_GPU_b,
    int* num_nnz_a,
    int* num_nnz_b,
    int* nnz_GPU_a,
    int* nnz_GPU_b,
    int* nnz_HOST_a,
    int* nnz_HOST_b,
    const int* row_ptr,
    const int* col_idx,
    int t_i,
    int t_j,
    int m,
    int row_id,  // row starts index of C
    int col_id,  // col starts index of C
    int* row_ptr_HOST_a,
    int* row_ptr_HOST_b,
    int* col_idx_HOST_a,
    int* col_idx_HOST_b,
    int target)
{
    // might want to use a adaptive t_i for the last iteration

    // std::cout << "row_id=" << row_id << " | col_id=" << col_id << " | target=" << target << " | m=" << m << std::endl;
    if (target % 2 == 0)
    {
        // std::cout << "row_ptr" << std::endl;
        // for (int i = 0; i < 8; i++)
        // {
        //     std::cout << row_ptr[i] << " ";
        // }
        // std::cout << std::endl;

        for (int i = 0; i < 2 * t_i + 1; i++)  // for (int i = 0; i < 80 * t_i + 1; i++)
        {
            row_ptr_HOST_a[i] = row_ptr[row_id + i];
        }

        int counter = 0;
        int start = 0;
        int sum = 0;
        num_nnz_a[0] = 0;
        for (int i = 0; i < 2 * t_i; i++)  // for (int i = 0; i < 80 * t_i; i++)
        {
            start = 0;
            for (int j = 0; j < row_ptr_HOST_a[i + 1] - row_ptr_HOST_a[i]; j++)
            {
                if (col_idx[row_ptr_HOST_a[i] + j] < col_id)
                {
                    start++;
                }
                else if (col_idx[row_ptr_HOST_a[i] + j] < col_id + t_j)
                {
                    sum++;
                    col_idx_HOST_a[counter] = col_idx[row_ptr_HOST_a[i] + j];
                    counter++;
                }
                nnz_HOST_a[i] = sum;
            }
            num_nnz_a[i + 1] = sum;
            row_ptr_HOST_a[i] += start;
        }

        // // print row_ptr_HOST_a
        // std::cout << "row_ptr_HOST_a" << std::endl;
        // for (int i = 0; i < 2 * t_i + 1; i++)  // for (int i = 0; i < 80 * t_i; i++)
        // {
        //     std::cout << row_ptr_HOST_a[i] << " ";
        // }
        // std::cout << std::endl;
        // // print num_nnz_a
        // std::cout << "num_nnz_a" << std::endl;
        // for (int i = 0; i < 2 * t_i + 1; i++)  // for (int i = 0; i < 80 * t_i + 1; i++)
        // {
        //     std::cout << num_nnz_a[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "col_idx_HOST_a" << std::endl;
        // for (int i = 0; i < counter; i++)  // for (int i = 0; i < 80 * t_i; i++)
        // {
        //     std::cout << col_idx_HOST_a[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;

        CUDA_CHECK(
            cudaMemcpyAsync(
                row_ptr_GPU_a,
                row_ptr_HOST_a,
                (2 * t_i + 1) * sizeof(int),  //(80 * t_i + 1) * sizeof(int),
                cudaMemcpyHostToDevice,
                stream_rp_a));
        CUDA_CHECK(
            cudaMemcpyAsync(
                num_nnz_GPU_a,
                num_nnz_a,
                (2 * t_i + 1) * sizeof(int),  //(80 * t_i + 1) * sizeof(int),
                cudaMemcpyHostToDevice,
                stream_nnz_a));
        CUDA_CHECK(
            cudaMemcpyAsync(
                col_idx_GPU_a,
                col_idx_HOST_a,
                row_ptr_HOST_a[2 * t_i] * sizeof(int),  // row_ptr_HOST_b[80 * t_i] * sizeof(int),
                cudaMemcpyHostToDevice,
                stream_ci_a));
        CUDA_CHECK(
            cudaMemcpyAsync(
                nnz_GPU_a,
                nnz_HOST_a,
                2 * sizeof(int),  // 80 * sizeof(int),
                cudaMemcpyHostToDevice,
                stream_nnz_a));
    }
    else
    {
        // std::cout << "row_ptr" << std::endl;
        // for (int i = 0; i < 8; i++)
        // {
        //     std::cout << row_ptr[i] << " ";
        // }
        // std::cout << std::endl;

        for (int i = 0; i < 2 * t_i + 1; i++)  // for (int i = 0; i < 80 * t_i + 1; i++)
        {
            row_ptr_HOST_b[i] = row_ptr[row_id + i];
        }

        int counter = 0;
        int start = 0;
        int sum = 0;
        num_nnz_b[0] = 0;
        for (int i = 0; i < 2 * t_i; i++)  // for (int i = 0; i < 80 * t_i; i++)
        {
            start = 0;
            for (int j = 0; j < row_ptr_HOST_b[i + 1] - row_ptr_HOST_b[i]; j++)
            {
                if (col_idx[row_ptr_HOST_b[i] + j] < col_id)
                {
                    start++;
                }
                else if (col_idx[row_ptr_HOST_b[i] + j] < col_id + t_j)
                {
                    sum++;
                    col_idx_HOST_b[counter] = col_idx[row_ptr_HOST_b[i] + j];
                    counter++;
                }
                nnz_HOST_b[i] = sum;
            }
            num_nnz_b[i + 1] = sum;
            row_ptr_HOST_b[i] += start;
        }

        // // print row_ptr_HOST_b
        // std::cout << "row_ptr_HOST_b" << std::endl;
        // for (int i = 0; i < 2 * t_i + 1; i++)  // for (int i = 0; i < 80 * t_i; i++)
        // {
        //     std::cout << row_ptr_HOST_b[i] << " ";
        // }
        // std::cout << std::endl;
        // // print num_nnz_b
        // std::cout << "num_nnz_b" << std::endl;
        // for (int i = 0; i < 2 * t_i + 1; i++)  // for (int i = 0; i < 80 * t_i + 1; i++)
        // {
        //     std::cout << num_nnz_b[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "col_idx_HOST_b" << std::endl;
        // for (int i = 0; i < counter; i++)  // for (int i = 0; i < 80 * t_i; i++)
        // {
        //     std::cout << col_idx_HOST_b[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;

        CUDA_CHECK(
            cudaMemcpyAsync(
                row_ptr_GPU_b,
                row_ptr_HOST_b,
                (2 * t_i + 1) * sizeof(int),  //(80 * t_i + 1) * sizeof(int),
                cudaMemcpyHostToDevice,
                stream_rp_b));
        CUDA_CHECK(
            cudaMemcpyAsync(
                num_nnz_GPU_b,
                num_nnz_b,
                (2 * t_i + 1) * sizeof(int),  //(80 * t_i + 1) * sizeof(int),
                cudaMemcpyHostToDevice,
                stream_nnz_b));
        CUDA_CHECK(
            cudaMemcpyAsync(
                col_idx_GPU_b,
                col_idx_HOST_b,
                row_ptr_HOST_b[2 * t_i] * sizeof(int),  // row_ptr_HOST_b[80 * t_i] * sizeof(int),
                cudaMemcpyHostToDevice,
                stream_ci_b));
        CUDA_CHECK(
            cudaMemcpyAsync(
                nnz_GPU_b,
                nnz_HOST_b,
                2 * sizeof(int),  // 80 * sizeof(int),
                cudaMemcpyHostToDevice,
                stream_nnz_b));
    }
}

// void send_C(
//     cudaStream_t stream_a,
//     cudaStream_t stream_b,
//     float* matrixC_GPU_a,
//     float* matrixC_GPU_b,
//     const float* values,
//     const int* row_ptr_a,
//     const int* row_ptr_b,
//     int* num_nnz_a,
//     int* num_nnz_b,
//     int t_i,
//     int t_j,
//     int target)
// {
//     if (target % 2 == 0)
//     {
//         for (int i = 0; i < 1 * t_i; i++)  // for (int i = 0; i < 80 * t_i; i++)
//         {
//             int nnz = num_nnz_a[i + 1] - num_nnz_a[i];
//             const float* temp[nnz];
//             for (int j = 0; j < nnz; j++)
//             {
//                 temp[j] = values + row_ptr_a[i] + j;
//             }
//             CUDA_CHECK(
//                 cudaMemcpyAsync(
//                     matrixC_GPU_a + num_nnz_a[i],
//                     temp,
//                     nnz * sizeof(float),
//                     cudaMemcpyHostToDevice,
//                     stream_a));
//         }
//     }
//     else
//     {
//         for (int i = 0; i < 1 * t_i; i++)  // for (int i = 0; i < 80 * t_i; i++)
//         {
//             int nnz = num_nnz_b[i + 1] - num_nnz_b[i];
//             const float* temp[nnz];
//             for (int j = 0; j < nnz; j++)
//             {
//                 temp[j] = values + row_ptr_b[i] + j;
//             }
//             CUDA_CHECK(
//                 cudaMemcpyAsync(
//                     matrixC_GPU_b + num_nnz_a[i],
//                     temp,
//                     nnz * sizeof(float),
//                     cudaMemcpyHostToDevice,
//                     stream_b));
//         }
//     }
// }

void send_result(
    cudaStream_t stream_a,
    cudaStream_t stream_b,
    float* matrixResult_GPU_a,
    float* matrixResult_GPU_b,
    float* result_from_gpu,
    int* num_nnz_a,
    int* num_nnz_b,
    int* nnz_HOST_a,
    int* nnz_HOST_b,
    int t_i,
    int target)
{
    if (target % 2 == 0)
    {
        int nnz = num_nnz_b[2 * t_i];  // int nnz = num_nnz_b[80 * t_i];
        // std::cout << "nnz=" << nnz << std::endl;
        CUDA_CHECK(
            cudaMemcpyAsync(
                result_from_gpu,
                matrixResult_GPU_b,
                nnz * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream_b));

        cudaStreamSynchronize(stream_b);  // if nnz is correct this works and this barrier has to be fixed
        // std::cout << "result_from_gpu" << std::endl;
        // for (int i = 0; i < nnz; i++)
        // {
        //     std::cout << result_from_gpu[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;
    }
    else
    {
        int nnz = num_nnz_a[2 * t_i];  // int nnz = num_nnz_b[80 * t_i];
        // std::cout << "nnz=" << nnz << std::endl;
        CUDA_CHECK(
            cudaMemcpyAsync(
                result_from_gpu,
                matrixResult_GPU_a,
                nnz * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream_a));

        cudaStreamSynchronize(stream_a);  // if nnz is correct this works and this barrier has to be fixed
        // std::cout << "result_from_gpu" << std::endl;
        // for (int i = 0; i < nnz; i++)
        // {
        //     std::cout << result_from_gpu[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << std::endl;
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
    int t_i,
    int target)
{
    // std::cout << "result_HOST" << std::endl;
    // for (int i = 0; i < 8; i++)
    // {
    //     std::cout << result_HOST[i] << " ";
    // }
    // std::cout << std::endl;

    if (target % 2 == 0)
    {
        for (int i = 0; i < 2 * t_i; i++)  // for (int i = 0; i < 80 * t_i; i++)
        {
            int nnz = num_nnz_a[i + 1] - num_nnz_a[i];
            for (int j = 0; j < nnz; j++)
            {
                result_HOST[row_ptr_a[i] + j] += result_from_gpu[num_nnz_a[i] + j];
                // std::cout << result_from_gpu[num_nnz_a[i] + j] << " ";
            }
            // std::cout << std::endl;
        }
    }
    else
    {
        for (int i = 0; i < 2 * t_i; i++)  // for (int i = 0; i < 80 * t_i; i++)
        {
            int nnz = num_nnz_b[i + 1] - num_nnz_b[i];
            for (int j = 0; j < nnz; j++)
            {
                result_HOST[row_ptr_b[i] + j] += result_from_gpu[num_nnz_b[i] + j];
                // std::cout << result_from_gpu[num_nnz_b[i] + j] << " ";
            }
            // std::cout << std::endl;
        }
    }

    // std::cout << "result_HOST" << std::endl;
    // for (int i = 0; i < 8; i++)  // for (int i = 0; i < 80 * t_i; i++)
    // {
    //     std::cout << result_HOST[i] << " ";
    // }
    // std::cout << std::endl;
}

void launch_computation_even(
    cudaStream_t stream_a_send_a,
    cudaStream_t stream_rp_send_a,
    cudaStream_t stream_ci_send_a,
    // cudaStream_t stream_c_send_a,
    cudaStream_t stream_nnz_a,
    cudaStream_t stream_b_send_a,
    cudaStream_t stream_b_send_b,
    cudaStream_t stream_compute,
    cudaStream_t stream_a_send_b,
    cudaStream_t stream_rp_send_b,
    cudaStream_t stream_ci_send_b,
    // cudaStream_t stream_c_send_b,
    cudaStream_t stream_nnz_b,
    float* matrixA_GPU_a,
    float* matrixA_GPU_b,
    float* matrixB_GPU_a,
    float* matrixB_GPU_b,
    float* matrixResult_GPU_a,
    float* matrixResult_GPU_b,
    int* row_ptr_GPU_a,
    int* row_ptr_GPU_b,
    int* col_idx_GPU_a,
    int* col_idx_GPU_b,
    int* num_nnz_GPU_a,
    int* num_nnz_GPU_b,
    int* nnz_GPU_a,
    int* nnz_GPU_b,
    int t_i,
    int t_j,
    int t_k,
    int start_row,
    int start_col,
    int t_k_by_4,
    int target_a)
{
    if (target_a % 2 == 0)
    {
        // check that the memory transfer for this iteration is finished (A, row_ptr, col_idx, C, set_to_zero)
        cudaStreamSynchronize(stream_a_send_a);
        cudaStreamSynchronize(stream_rp_send_a);
        cudaStreamSynchronize(stream_ci_send_a);
        // cudaStreamSynchronize(stream_c_send_a);
        cudaStreamSynchronize(stream_nnz_a);

        // check that the memory transfer for this iteration is finished (B)
        cudaStreamSynchronize(stream_b_send_a);
        cudaStreamSynchronize(stream_b_send_b);
        for (int q = 0; q < 2; q++)  // for (int q = 0; q < 80; q++)
        {
            // std::cout << "even | on _a | target_a = " << target_a << std::endl;
            // Call the kernel to execute the acutal SDDMM
            // compute_lml2<<<1, 1>>>(matrixA_GPU_a); // used A
            // compute_lml2<<<1, 1>>>(matrixB_GPU_a); // used B
            // compute_lml2<<<1, 1, 0, stream_compute>>>(matrixResult_GPU_a, target_a); // writeback for correct nnz
            compute_lml2<<<1, 2, 98304, stream_compute>>>(matrixA_GPU_a, matrixB_GPU_a, num_nnz_GPU_a, col_idx_GPU_a, t_i, matrixResult_GPU_a, q * t_i, start_col, t_k_by_4);
        }
    }
    else
    {
        // check that the memory transfer for this iteration is finished (A, row_ptr, col_idx, C, set_to_zero)
        cudaStreamSynchronize(stream_a_send_b);
        cudaStreamSynchronize(stream_rp_send_b);
        cudaStreamSynchronize(stream_ci_send_b);
        // cudaStreamSynchronize(stream_c_send_b);
        cudaStreamSynchronize(stream_nnz_b);

        // check that the memory transfer for this iteration is finished (B)
        cudaStreamSynchronize(stream_b_send_a);
        cudaStreamSynchronize(stream_b_send_b);
        for (int q = 0; q < 2; q++)  // for (int q = 0; q < 80; q++)
        {
            // std::cout << "even | on _b | target_a = " << target_a << std::endl;
            // Call the kernel to execute the acutal SDDMM
            // compute_lml2<<<1, 1>>>(matrixA_GPU_b); // used A
            // compute_lml2<<<1, 1>>>(matrixB_GPU_a); // used B
            // compute_lml2<<<1, 1, 0, stream_compute>>>(matrixResult_GPU_b, target_a); // writeback for correct nnz
            compute_lml2<<<1, 2, 98304, stream_compute>>>(matrixA_GPU_b, matrixB_GPU_a, num_nnz_GPU_b, col_idx_GPU_b, t_i, matrixResult_GPU_b, q * t_i, start_col, t_k_by_4);
        }
    }
}

void launch_computation_odd(
    cudaStream_t stream_a_send_a,
    cudaStream_t stream_rp_send_a,
    cudaStream_t stream_ci_send_a,
    // cudaStream_t stream_c_send_a,
    cudaStream_t stream_nnz_a,
    cudaStream_t stream_b_send_a,
    cudaStream_t stream_b_send_b,
    cudaStream_t stream_compute,
    cudaStream_t stream_a_send_b,
    cudaStream_t stream_rp_send_b,
    cudaStream_t stream_ci_send_b,
    // cudaStream_t stream_c_send_b,
    cudaStream_t stream_nnz_b,
    float* matrixA_GPU_a,
    float* matrixA_GPU_b,
    float* matrixB_GPU_a,
    float* matrixB_GPU_b,
    float* matrixResult_GPU_a,
    float* matrixResult_GPU_b,
    int* row_ptr_GPU_a,
    int* row_ptr_GPU_b,
    int* col_idx_GPU_a,
    int* col_idx_GPU_b,
    int* num_nnz_GPU_a,
    int* num_nnz_GPU_b,
    int* nnz_GPU_a,
    int* nnz_GPU_b,
    int t_i,
    int t_j,
    int t_k,
    int start_row,
    int start_col,
    int t_k_by_4,
    int target_a)
{
    if (target_a % 2 == 0)
    {
        // check that the memory transfer for this iteration is finished (A, row_ptr, col_idx, C, set_to_zero)
        cudaStreamSynchronize(stream_a_send_a);
        cudaStreamSynchronize(stream_rp_send_a);
        cudaStreamSynchronize(stream_ci_send_a);
        // cudaStreamSynchronize(stream_c_send_a);
        cudaStreamSynchronize(stream_nnz_a);

        // check that the memory transfer for this iteration is finished (B)
        cudaStreamSynchronize(stream_b_send_a);
        cudaStreamSynchronize(stream_b_send_b);
        for (int q = 0; q < 2; q++)  // for (int q = 0; q < 80; q++)
        {
            // std::cout << "odd | on _a | target_a = " << target_a << std::endl;
            // Call the kernel to execute the acutal SDDMM
            // compute_lml2<<<1, 1>>>(matrixA_GPU_a); // used A
            // compute_lml2<<<1, 1>>>(matrixB_GPU_b); // used B
            // compute_lml2<<<1, 1, 0, stream_compute>>>(matrixResult_GPU_a, target_a); // writeback for correct nnz
            compute_lml2<<<1, 2, 98304, stream_compute>>>(matrixA_GPU_a, matrixB_GPU_b, num_nnz_GPU_a, col_idx_GPU_a, t_i, matrixResult_GPU_a, q * t_i, start_col, t_k_by_4);
        }
    }
    else
    {
        // check that the memory transfer for this iteration is finished (A, row_ptr, col_idx, C, set_to_zero)
        cudaStreamSynchronize(stream_a_send_b);
        cudaStreamSynchronize(stream_rp_send_b);
        cudaStreamSynchronize(stream_ci_send_b);
        // cudaStreamSynchronize(stream_c_send_b);
        cudaStreamSynchronize(stream_nnz_b);

        // check that the memory transfer for this iteration is finished (B)
        cudaStreamSynchronize(stream_b_send_a);
        cudaStreamSynchronize(stream_b_send_b);
        for (int q = 0; q < 2; q++)  // for (int q = 0; q < 80; q++)
        {
            // std::cout << "odd | on _b | target_a = " << target_a << std::endl;
            // Call the kernel to execute the acutal SDDMM
            // compute_lml2<<<1, 1>>>(matrixA_GPU_b); // used A
            // compute_lml2<<<1, 1>>>(matrixB_GPU_b); // used B
            // compute_lml2<<<1, 1, 0, stream_compute>>>(matrixResult_GPU_b, target_a); // writeback for correct nnz
            compute_lml2<<<1, 2, 98304, stream_compute>>>(matrixA_GPU_b, matrixB_GPU_b, num_nnz_GPU_b, col_idx_GPU_b, t_i, matrixResult_GPU_b, q * t_i, start_col, t_k_by_4);
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
    std::cout << "SDDMM_CSR" << std::endl;
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

    // // check the dimensions of the matrices
    // assert(matrixB_transpose_HOST.getNumCols() == k && "Error: matrixB_transpose has incompatible dimensions");
    // assert(matrixC_HOST.getNumRows() == m && "Error: matrixC has incompatible dimensions m");
    // assert(matrixC_HOST.getNumCols() == n && "Error: matrixC has incompatible dimensions n");
    // assert(matrixResult_sparse_HOST.getNumRows() == m && "Error: matrixResult has incompatible dimensions m");
    // assert(matrixResult_sparse_HOST.getNumCols() == n && "Error: matrixResult has incompatible dimensions n");

    // here we need some magic to define t_j, t_k, t_i and num_iterations
    int t_j = 2;
    int t_k = 8;  // this probably has to be around 16 for p=1% to fit everything on the GPU
    int t_i = 2;
    int t_k_by_4 = 2;            // t_k / 4
    int num_iterations_t_j = 2;  // n / t_j
    int num_iterations_t_k = 2;  // k / t_k
    int num_iterations_t_i = 3;  // m / 80 * t_i
    int curr_col_id = 0;         // of B_T
    int curr_row_id = 0;         // of B_T
    int curr_t_i_id = 0;         // of A
    int curr_row_id_C = 0;       // of C
    int curr_col_id_C = 0;       // of C
    float p = 1;                 // density of matrixC

    // std::cout << "t_j=" << t_j << " | t_k=" << t_k << " | t_i=" << t_i << std::endl;

    // allow max shared memory usage
    cudaFuncSetAttribute(compute_lml2, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    // set max number of cudastreams in the system
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "32", 1);

    // allocate memory for the matrices on the GPU
    // _a is for the kernels 0-79, 160-239, ...
    // _b is for the remainig kernels
    float* matrixA_GPU_a;
    float* matrixA_GPU_b;
    float* matrixB_transpose_GPU_a;
    float* matrixB_transpose_GPU_b;
    // float* matrixC_GPU_a;
    // float* matrixC_GPU_b;
    float* matrixResult_GPU_a;
    float* matrixResult_GPU_b;
    int* col_idx_GPU_a;
    int* col_idx_GPU_b;
    int* row_ptr_GPU_a;
    int* row_ptr_GPU_b;
    int* num_nnz_GPU_a;  // number of non-zero elements per row for iteration a as an internal row_ptr
    int* num_nnz_GPU_b;  // number of non-zero elements per row for iteration b as an internal row_ptr
    int* nnz_GPU_a;      // number of non-zero elements per row for iteration a
    int* nnz_GPU_b;      // number of non-zero elements per row for iteration b

    CUDA_CHECK(
        cudaMalloc(
            &matrixA_GPU_a,
            2 * t_i * t_k * sizeof(float)));  // 80 * t_i * t_k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixA_GPU_b,
            2 * t_i * t_k * sizeof(float)));  // 80 * t_i * t_k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixB_transpose_GPU_a,
            t_j * t_k * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixB_transpose_GPU_b,
            t_j * t_k * sizeof(float)));
    // CUDA_CHECK(
    //     cudaMalloc(
    //         &matrixC_GPU_a,
    //         int(1 * 10 * p * t_i * t_j) * sizeof(float)));  // int(80 * 10 * p * t_i * t_j) * sizeof(float)));
    // CUDA_CHECK(
    //     cudaMalloc(
    //         &matrixC_GPU_b,
    //         int(1 * 10 * p * t_i * t_j) * sizeof(float)));  // int(80 * 10 * p * t_i * t_j) * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixResult_GPU_a,
            int(2 * 10 * p * t_i * t_j) * sizeof(float)));  // int(80 * 10 * p * t_i * t_j) * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &matrixResult_GPU_b,
            int(2 * 10 * p * t_i * t_j) * sizeof(float)));  // int(80 * 10 * p * t_i * t_j) * sizeof(float)));
    CUDA_CHECK(
        cudaMalloc(
            &col_idx_GPU_a,
            int(2 * 10 * p * t_i * t_j) * sizeof(int)));  // int(80 * 10 * p * t_i * t_j) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &col_idx_GPU_b,
            int(2 * 10 * p * t_i * t_j) * sizeof(int)));  // int(80 * 10 * p * t_i * t_j) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &row_ptr_GPU_a,
            (2 * t_i + 1) * sizeof(int)));  //(80 * t_i + 1) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &row_ptr_GPU_b,
            (2 * t_i + 1) * sizeof(int)));  //(80 * t_i + 1) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &num_nnz_GPU_a,
            (2 * t_i + 1) * sizeof(int)));  //(80 * t_i + 1) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &num_nnz_GPU_b,
            (2 * t_i + 1) * sizeof(int)));  //(80 * t_i + 1) * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &nnz_GPU_a,
            2 * sizeof(int)));  // 80 * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(
            &nnz_GPU_b,
            2 * sizeof(int)));  // 80 * sizeof(int)));

    cudaStream_t stream_a_send_a, stream_b_send_a, stream_receive_a, stream_compute;
    cudaStream_t stream_a_send_b, stream_b_send_b, stream_receive_b;
    cudaStream_t stream_rp_send_a, stream_ci_send_a;
    cudaStream_t stream_rp_send_b, stream_ci_send_b;
    cudaStream_t stream_nnz_a, stream_nnz_b;
    // cudaStream_t stream_c_send_a, stream_c_send_b;
    cudaStreamCreate(&stream_a_send_a);
    cudaStreamCreate(&stream_b_send_a);
    cudaStreamCreate(&stream_receive_a);
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_rp_send_a);
    cudaStreamCreate(&stream_ci_send_a);
    cudaStreamCreate(&stream_nnz_a);
    cudaStreamCreate(&stream_a_send_b);
    cudaStreamCreate(&stream_b_send_b);
    cudaStreamCreate(&stream_receive_b);
    cudaStreamCreate(&stream_rp_send_b);
    cudaStreamCreate(&stream_ci_send_b);
    cudaStreamCreate(&stream_nnz_b);
    // cudaStreamCreate(&stream_c_send_a);
    // cudaStreamCreate(&stream_c_send_b);

    // ints to differentiate between loading to _a or _b
    int target_b = 0;
    int target_a = 0;

    // save row_ptr and col_idx on the host
    int* row_ptr_HOST_a = new int[2 * t_i + 1];                  //[80 * t_i + 1];
    int* row_ptr_HOST_b = new int[2 * t_i + 1];                  //[80 * t_i + 1];
    int* num_nnz_a = new int[2 * t_i + 1];                       //[80 * t_i];
    int* num_nnz_b = new int[2 * t_i + 1];                       //[80 * t_i];
    int* col_idx_HOST_a = new int[int(2 * 10 * p * t_i * t_j)];  //[int(80 * 10 * p * t_i * t_j)];
    int* col_idx_HOST_b = new int[int(2 * 10 * p * t_i * t_j)];  //[int(80 * 10 * p * t_i * t_j)];
    int* nnz_HOST_a = new int[2];                                //[80];
    int* nnz_HOST_b = new int[2];                                //[80];

    // create memory for the result on the host
    float* result_from_gpu = new float[int(2 * 10 * p * t_i * t_j)];  //[int(80 * 10 * p * t_i * t_j)];

    // local copy of values of all matrices
    const float* values_A = matrixA_HOST.getValues();
    const float* values_B = matrixB_transpose_HOST.getValues();
    // const float* values_C = matrixC_HOST.getValues().data();
    const int* col_idx_C = matrixC_HOST.getColIndices().data();
    // const int* row_ptr_C = matrixC_HOST.getRowArray().data();
    float* values_result = new float[nnz];
    memset(values_result, 0, nnz * sizeof(float));
    int row_GPU = 0;

    // build padded row_ptr to cheese m % (80 * t_i) != 0
    std::vector<int> row_ptr = matrixC_HOST.getRowArray();
    int last = row_ptr[row_ptr.size() - 1];
    for (int i = 0; i < ((2 * t_i) - (m % (2 * t_i))); i++)  // for (int i = 0; i < ((80 * t_i) - (m % (80 * t_i))); i++)
    {
        row_ptr.push_back(last);
    }
    const int* row_ptr_C = row_ptr.data();

    // // std::cout << "row_ptr_C" << std::endl;
    // for (int i = 0; i < 15; i++)
    // {
    //     std::cout << row_ptr_C[i] << " ";
    // }
    // std::cout << std::endl;

    std::cout << "setup finished" << std::endl;
    // int remove_this_counter = 0;

    // start the timer
    this->start_run();

    // transfer the memory for the first iteration
    // whole block of B
    send_B(
        stream_b_send_a,
        stream_b_send_b,
        matrixB_transpose_GPU_a,
        matrixB_transpose_GPU_b,
        values_B,
        t_j,
        t_k,
        curr_row_id,
        curr_col_id,
        k,
        target_b);

    // the correspnding blocks of A
    // m % 80 == 0; for m % 80 != 0 we need to add functionality
    for (int q = 0; q < 2; q++)  // for (int q = 0; q < 80; q++)
    {
        // load 1 block of A
        send_A(
            stream_a_send_a,
            stream_a_send_b,
            matrixA_GPU_a,
            matrixA_GPU_b,
            values_A,
            t_i,
            t_k,
            curr_col_id,
            curr_t_i_id * t_i,
            row_GPU * t_i,
            k,
            target_a);
        row_GPU++;
        curr_t_i_id++;
    }
    row_GPU = 0;

    // set initial row_ptr and col_idx
    send_row_ptr_and_col_id(
        stream_rp_send_a,
        stream_ci_send_a,
        stream_nnz_a,
        stream_rp_send_b,
        stream_ci_send_b,
        stream_nnz_b,
        row_ptr_GPU_a,
        row_ptr_GPU_b,
        col_idx_GPU_a,
        col_idx_GPU_b,
        num_nnz_GPU_a,
        num_nnz_GPU_b,
        num_nnz_a,
        num_nnz_b,
        nnz_GPU_a,
        nnz_GPU_b,
        nnz_HOST_a,
        nnz_HOST_b,
        row_ptr_C,
        col_idx_C,
        t_i,
        t_j,
        m,
        curr_row_id_C * 2,  // curr_row_id_C * 80,
        curr_col_id_C,
        row_ptr_HOST_a,
        row_ptr_HOST_b,
        col_idx_HOST_a,
        col_idx_HOST_b,
        target_a);
    curr_row_id_C += t_i;

    // // set initial matrixC
    // send_C(
    //     stream_c_send_a,
    //     stream_c_send_b,
    //     matrixC_GPU_a,
    //     matrixC_GPU_b,
    //     values_C,
    //     row_ptr_HOST_a,
    //     row_ptr_HOST_b,
    //     num_nnz_a,
    //     num_nnz_b,
    //     t_i,
    //     t_j,
    //     target_a);

    for (int i = 0; i < num_iterations_t_k; i++)
    {
        for (int j = 0; j < num_iterations_t_j; j++)
        {
            for (int w = 0; w < num_iterations_t_i; w++)
            {
                // std::cout << "even | i=" << i << " | j=" << j << " | w=" << w << " | target_b=" << target_b << " | target_a=" << target_a << std::endl;
                if (target_b % 2 == 0)
                {
                    // remove_this_counter++;
                    // std::cout << "remove_this_counter=" << remove_this_counter << " | target_a=" << target_a << " | target_b=" << target_b << std::endl;
                    launch_computation_even(
                        stream_a_send_a,
                        stream_rp_send_a,
                        stream_ci_send_a,
                        // stream_c_send_a,
                        stream_nnz_a,
                        stream_b_send_a,
                        stream_b_send_b,
                        stream_compute,
                        stream_a_send_b,
                        stream_rp_send_b,
                        stream_ci_send_b,
                        // stream_c_send_b,
                        stream_nnz_b,
                        matrixA_GPU_a,
                        matrixA_GPU_b,
                        matrixB_transpose_GPU_a,
                        matrixB_transpose_GPU_b,
                        matrixResult_GPU_a,
                        matrixResult_GPU_b,
                        row_ptr_GPU_a,
                        row_ptr_GPU_b,
                        col_idx_GPU_a,
                        col_idx_GPU_b,
                        num_nnz_GPU_a,
                        num_nnz_GPU_b,
                        nnz_GPU_a,
                        nnz_GPU_b,
                        t_i,
                        t_j,
                        t_k,
                        (curr_t_i_id - 2) * t_i,  //(curr_t_i_id - 80) * t_i,
                        curr_row_id,
                        t_k_by_4,
                        target_a);
                }
                else
                {
                    // remove_this_counter++;
                    // std::cout << "remove_this_counter=" << remove_this_counter << " | target_a=" << target_a << " | target_b=" << target_b << std::endl;
                    launch_computation_odd(
                        stream_a_send_b,
                        stream_rp_send_b,
                        stream_ci_send_b,
                        // stream_c_send_b,
                        stream_nnz_b,
                        stream_b_send_a,
                        stream_b_send_b,
                        stream_compute,
                        stream_a_send_a,
                        stream_rp_send_a,
                        stream_ci_send_a,
                        // stream_c_send_a,
                        stream_nnz_a,
                        matrixA_GPU_a,
                        matrixA_GPU_b,
                        matrixB_transpose_GPU_a,
                        matrixB_transpose_GPU_b,
                        matrixResult_GPU_a,
                        matrixResult_GPU_b,
                        row_ptr_GPU_a,
                        row_ptr_GPU_b,
                        col_idx_GPU_a,
                        col_idx_GPU_b,
                        num_nnz_GPU_a,
                        num_nnz_GPU_b,
                        nnz_GPU_a,
                        nnz_GPU_b,
                        t_i,
                        t_j,
                        t_k,
                        (curr_t_i_id - 2) * t_i,  //(curr_t_i_id - 80) * t_i,
                        curr_row_id,
                        t_k_by_4,
                        target_a);
                }

                target_a++;
                // save the result on the host from the previous iteration (not on first iteration)
                if (i != 0 || j != 0 || w != 0)
                {
                    // check that the memory transfer from device to host has finished
                    cudaStreamSynchronize(stream_receive_a);
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
                        t_i,
                        target_a);
                }

                // start memory transfer for the next iteration (not on last iteration)
                if (i != num_iterations_t_k - 1 || j != num_iterations_t_j - 1 || w != num_iterations_t_i - 1)
                {
                    // check if we need to start at the top again
                    if (w == num_iterations_t_i - 1)
                    {
                        curr_t_i_id = 0;
                        curr_row_id_C = 0;
                        if (j == num_iterations_t_j - 1)
                        {
                            // std::cout << "curr_row_id=" << curr_row_id << " | curr_col_id=" << curr_col_id << std::endl;
                            curr_col_id += t_k;
                            curr_row_id = 0;
                            curr_col_id_C = 0;
                            // std::cout << "curr_row_id=" << curr_row_id << " | curr_col_id=" << curr_col_id << std::endl;
                        }
                        else
                        {
                            // std::cout << "curr_row_id=" << curr_row_id << " | curr_col_id=" << curr_col_id << std::endl;
                            curr_row_id += t_j;
                            curr_col_id_C += t_j;
                            // std::cout << "curr_row_id=" << curr_row_id << " | curr_col_id=" << curr_col_id << std::endl;
                        }
                    }

                    if (w == num_iterations_t_i - 1)
                    {
                        //  B can be loaded throughout all loop iterations so it only has to be started once
                        target_b++;
                        // this could also be split over num_iterations_t_j iterations
                        // std::cout << "curr_row_id=" << curr_row_id << " | curr_col_id=" << curr_col_id << std::endl;
                        send_B(
                            stream_b_send_a,
                            stream_b_send_b,
                            matrixB_transpose_GPU_a,
                            matrixB_transpose_GPU_b,
                            values_B,
                            t_j,
                            t_k,
                            curr_row_id,
                            curr_col_id,
                            k,
                            target_b);
                    }

                    // load the next 80 blocks of A
                    // m % 80 == 0
                    for (int q = 0; q < 2; q++)  // for (int q = 0; q < 80; q++)
                    {
                        // load 1 block of A
                        send_A(
                            stream_a_send_a,
                            stream_a_send_b,
                            matrixA_GPU_a,
                            matrixA_GPU_b,
                            values_A,
                            t_i,
                            t_k,
                            curr_col_id,
                            curr_t_i_id * t_i,
                            row_GPU * t_i,
                            k,
                            target_a);
                        row_GPU++;
                        curr_t_i_id++;
                    }
                    row_GPU = 0;

                    // load the next row_ptr and col_idx
                    send_row_ptr_and_col_id(
                        stream_rp_send_a,
                        stream_ci_send_a,
                        stream_nnz_a,
                        stream_rp_send_b,
                        stream_ci_send_b,
                        stream_nnz_b,
                        row_ptr_GPU_a,
                        row_ptr_GPU_b,
                        col_idx_GPU_a,
                        col_idx_GPU_b,
                        num_nnz_GPU_a,
                        num_nnz_GPU_b,
                        num_nnz_a,
                        num_nnz_b,
                        nnz_GPU_a,
                        nnz_GPU_b,
                        nnz_HOST_a,
                        nnz_HOST_b,
                        row_ptr_C,
                        col_idx_C,
                        t_i,
                        t_j,
                        m,
                        curr_row_id_C * 2,  // curr_row_id_C * 80,
                        curr_col_id_C,
                        row_ptr_HOST_a,
                        row_ptr_HOST_b,
                        col_idx_HOST_a,
                        col_idx_HOST_b,
                        target_a);
                    curr_row_id_C += t_i;

                    // // load the next matrixC
                    // send_C(
                    //     stream_c_send_a,
                    //     stream_c_send_b,
                    //     matrixC_GPU_a,
                    //     matrixC_GPU_b,
                    //     values_C,
                    //     row_ptr_HOST_a,
                    //     row_ptr_HOST_b,
                    //     num_nnz_a,
                    //     num_nnz_b,
                    //     t_i,
                    //     t_j,
                    //     target_a);
                }

                // check that computation has finished
                cudaStreamSynchronize(stream_compute);
                // start memory transfer from device to host
                send_result(
                    stream_receive_a,
                    stream_receive_b,
                    matrixResult_GPU_a,
                    matrixResult_GPU_b,
                    result_from_gpu,
                    num_nnz_a,
                    num_nnz_b,
                    nnz_HOST_a,
                    nnz_HOST_b,
                    t_i,
                    target_a);
            }
        }
    }
    cudaStreamSynchronize(stream_compute);
    // wait until the last results are loaded back
    send_result(
        stream_receive_a,
        stream_receive_b,
        matrixResult_GPU_a,
        matrixResult_GPU_b,
        result_from_gpu,
        num_nnz_a,
        num_nnz_b,
        nnz_HOST_a,
        nnz_HOST_b,
        t_i,
        target_a);

    // save the last result on the host
    target_a++;
    save_result(
        result_from_gpu,
        values_result,
        row_ptr_HOST_a,
        row_ptr_HOST_b,
        col_idx_HOST_a,
        col_idx_HOST_b,
        num_nnz_a,
        num_nnz_b,
        t_i,
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
    // CUDA_CHECK(
    //     cudaFree(
    //         matrixC_GPU_a));
    // CUDA_CHECK(
    //     cudaFree(
    //         matrixC_GPU_b));
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
    CUDA_CHECK(
        cudaFree(
            nnz_GPU_a));
    CUDA_CHECK(
        cudaFree(
            nnz_GPU_b));
    cudaStreamDestroy(stream_a_send_a);
    cudaStreamDestroy(stream_b_send_a);
    cudaStreamDestroy(stream_receive_a);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_rp_send_a);
    cudaStreamDestroy(stream_ci_send_a);
    cudaStreamDestroy(stream_nnz_a);
    cudaStreamDestroy(stream_a_send_b);
    cudaStreamDestroy(stream_b_send_b);
    cudaStreamDestroy(stream_receive_b);
    cudaStreamDestroy(stream_rp_send_b);
    cudaStreamDestroy(stream_ci_send_b);
    cudaStreamDestroy(stream_nnz_b);
    // cudaStreamDestroy(stream_c_send_a);
    // cudaStreamDestroy(stream_c_send_b);

    // if this is not commented out, the program crashes
    delete[] row_ptr_HOST_a;
    delete[] row_ptr_HOST_b;
    delete[] col_idx_HOST_a;
    delete[] col_idx_HOST_b;
    // delete[] values_C;
    // delete[] col_idx_C;
    // delete[] row_ptr_C;
    delete[] result_from_gpu;
    delete[] values_result;
    delete[] num_nnz_a;
    delete[] num_nnz_b;
    delete[] nnz_HOST_a;
    delete[] nnz_HOST_b;
    row_ptr_HOST_a = nullptr;
    row_ptr_HOST_b = nullptr;
    col_idx_HOST_a = nullptr;
    col_idx_HOST_b = nullptr;
    values_A = nullptr;
    values_B = nullptr;
    // values_C = nullptr;
    col_idx_C = nullptr;
    row_ptr_C = nullptr;
    result_from_gpu = nullptr;
    values_result = nullptr;
    num_nnz_a = nullptr;
    num_nnz_b = nullptr;
    nnz_HOST_a = nullptr;
    nnz_HOST_b = nullptr;

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
