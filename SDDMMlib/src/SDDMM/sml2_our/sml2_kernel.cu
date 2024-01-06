// Assumption: B is transposed in Memory <3

#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "utils.h"

__global__ void compute_lml2(float* matrix_A, float* matrix_B, int* row_ptr, int* col_idx, int t_i, float* result, int start_row, int start_col, int t_k_by_4, int m)
{
    // used for A and B | B always starts at 0
    // const float4* new_array = reinterpret_cast<const float4*>(matrix_B);
    // if (threadIdx.x == 0)
    // {
    //     printf("Matrix_B \n");
    //     for (int i = 0; i < 8; i++)
    //     {
    //         printf("starting at %d: %f %f %f %f\n", start_row, new_array[i].x, new_array[i].y, new_array[i].z, new_array[i].w);
    //     }
    // }
    // printf("%d\n", row_ptr[t_i]);
    // for (int i = row_ptr[start]; i < row_ptr[start + t_i]; i++)
    // {
    //     result[i] = col_idx[i];
    //     // printf("from GPU %d: %d\n", start, col_idx[i]);
    // }
    // for (int i = start; i < start + t_i + 1; i++)
    // {
    //     result[i] = row_ptr[i];
    //     printf("from GPU %d: %d\n", start, row_ptr[i]);
    // }

    int tid = threadIdx.x;
    const float4* m_A = reinterpret_cast<const float4*>(matrix_A);
    const float4* m_B = reinterpret_cast<const float4*>(matrix_B);
    float temp;
    int row;
    int col;

    // shared memory has to be extern to use max shared memory
    extern __shared__ float4 shared_A[];
    int bound = (t_k_by_4 * t_i * 80) < m ? (t_k_by_4 * t_i * 80) : m;
    for (int i = tid; i < bound; i += 1024)
    {
        shared_A[i] = m_A[i];
    }

    __syncthreads();

    // if (tid == 1 && start_row == 0 && start_col == 0)
    // {
    //     for (int i = 0; i < 15; i++)
    //     {
    //         printf("%i\n", bound);
    //         printf("%i %f %f %f %f\n", start_row, shared_A[i].x, shared_A[i].y, shared_A[i].z, shared_A[i].w);
    //         printf("%i %f %f %f %f\n\n", start_row, m_A[i].x, m_A[i].y, m_A[i].z, m_A[i].w);
    //     }
    // }
    // __syncthreads();

    // printf("thread %d starting at %d\n", tid, start_row);

    // for (int i = (start_row + tid) * t_k_by_4; i < (start_row + tid) * t_k_by_4 + 2; i++)
    // {
    //     printf("thread %d starting at %d printing A: %f %f %f %f\n", tid, start_row, m_A[i].x, m_A[i].y, m_A[i].z, m_A[i].w);
    // }

    // for (int i = 0; i < 4; i++)
    // {
    //     printf("thread %d starting at %d printing B: %f %f %f %f\n", tid, start_row, m_B[i].x, m_B[i].y, m_B[i].z, m_B[i].w);
    // }
    for (int q = start_row + tid; q < ((start_row + t_i) < 5 ? (start_row + t_i) : 5); q += 1024)
    {
        // printf("thread %d starting at %d | row_ptr[q]= %d | row_ptr[q+1]= %d\n", tid, start_row, row_ptr[q], row_ptr[q + 1]);
        for (int i = row_ptr[q]; i < row_ptr[q + 1]; i++)
        {
            temp = 0;
            row = (q)*t_k_by_4;
            col = (col_idx[i] - start_col) * t_k_by_4;
            // printf("from GPU %d on thread %d: %d | %d ~ %d\n", start_row, tid, start_row + tid, col_idx[i], col_idx[i] - start_col);
            //  for loop over t_k
            for (int j = 0; j < t_k_by_4; j++)
            {
                // if (q == 1)
                // {
                //     printf("thread %d starting at %d printing A: %f %f %f %f\n", tid, start_row, m_A[row].x, m_A[row].y, m_A[row].z, m_A[row].w);
                //     printf("thread %d starting at %d printing A: %f %f %f %f\n", tid, start_row, shared_A[row].x, shared_A[row].y, shared_A[row].z, shared_A[row].w);
                //     printf("thread %d starting at %d printing B: %f %f %f %f\n", tid, start_row, m_B[col].x, m_B[col].y, m_B[col].z, m_B[col].w);
                // }
                temp += shared_A[row].x * m_B[col].x;
                temp += shared_A[row].y * m_B[col].y;
                temp += shared_A[row].z * m_B[col].z;
                temp += shared_A[row].w * m_B[col].w;
                row++;
                col++;
            }
            result[i] = temp;
            // printf("from GPU %d on thread %d: %d | %d ~ %d | result= %f\n", start_row, tid, start_row + tid, col_idx[i], col_idx[i] - start_col, temp);
        }
    }
}
