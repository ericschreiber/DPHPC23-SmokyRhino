#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.h"

__global__ void blocked_SDDMM_kernel(m, n, k, d_A, d_B, d_C, d_rowPtr, d_colIdx, d_result){

    int idx =  blockIdx.x * blockDim.x + threadIdx.x;

    // iterate over all rows assigned to a certain block
    for(int i = blockIdx.x; i < m; i += gridDim.x){
        // every thread multiplies two numbers, then we collaborate to reduce the numbers
        for(int j = threadIdx.x; j < n; j += blockDim.x){

            float my_sum = d_A[i][];
            for(int j = d_rowPtr[blockIdx.x]; j < m; j += blockDim.x){
                // iterate o
                for(int l = 0; l < k; l += BlockDim){

                }
        }
        
    }

    }

    // for(int j = z.getRowPtr()[i]; j < z.getRowPtr()[i+1]; j++){
    //         for(int l = 0; l < k; l++){
    //             result.getValues()[j] += x.at(i, l) * y.at(z.getColIndices()[j], l);

}


void compute_blockwise(int m, int n, int k, float *d_A, float *d_B, float *d_C, int *d_rowPtr, int *d_colIdx, float *d_result) {
    

    int max_blocks = 2024;
    int num_blocks = min(max_blocks, m);

    blocked_SDDMM_kernel<<<num_blocks, 1024>>>(m, n, k, d_A, d_B, d_C, d_rowPtr, d_colIdx, d_result);


    cublasHandle_t handle;
    CUDA_CHECK(cublasCreate(&handle));



}


