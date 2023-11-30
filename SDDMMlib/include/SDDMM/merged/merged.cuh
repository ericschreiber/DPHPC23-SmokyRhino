#ifndef MERGED_H
#define MERGED_H

void compute_m(
    const int m,
    const int n,
    const int k,  // number of rows of B
    const int numElementsC,
    const int numElementsCrowPtr,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values);

#endif