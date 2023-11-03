// naive_coo_SDDMM.cuh
#ifndef NAIVE_COO_SDDMM_H
#define NAIVE_COO_SDDMM_H

void compute(
    int m,
    int n,
    int k,
    int numElementsC,
    const float* const matrixA_GPU_values,
    const float* const matrixB_GPU_values,
    const float* const matrixC_GPU_values,
    const int* const matrixC_GPU_row_indices,
    const int* const matrixC_GPU_col_indices,
    float* matrixResult_GPU_values);

#endif  // NAIVE_COO_SDDMM_H