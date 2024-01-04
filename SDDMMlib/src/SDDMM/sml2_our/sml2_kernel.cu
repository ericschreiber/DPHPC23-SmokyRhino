// Assumption: B is transposed in Memory <3

#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "utils.h"

__global__ void compute_lml2(float* matrix_A, float* matrixB, int* row_ptr, int* col_idx, int t_i, float* result, int start)
{
    // used for A and B
    // const float4* new_array = reinterpret_cast<const float4*>(matrix_A);
    // for (int i = start; i < start + 2; i++)
    // {
    //     printf("starting at %d: %f %f %f %f\n", start, new_array[i].x, new_array[i].y, new_array[i].z, new_array[i].w);
    // }
    // printf("%d\n", row_ptr[t_i]);
    // for (int i = 0; i < row_ptr[t_i]; i++)
    // {
    //     result[i] = col_idx[i];
    //     printf("from GPU: %d\n", col_idx[i]);
    // }
    // for (int i = 0; i < t_i + 1; i++)
    // {
    //     result[i] = row_ptr[i];
    //     printf("%d\n", row_ptr[i]);
    // }
}
