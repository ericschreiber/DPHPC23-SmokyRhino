// Assumption: B is transposed in Memory <3

#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "utils.h"

__global__ void compute_lml2(float* array)
{
    const float4* new_array = reinterpret_cast<const float4*>(array);
    // for (int i = 0; i < 2; i++)
    // {
    //     printf("%f %f %f %f\n", new_array[i].x, new_array[i].y, new_array[i].z, new_array[i].w);
    // }
}
