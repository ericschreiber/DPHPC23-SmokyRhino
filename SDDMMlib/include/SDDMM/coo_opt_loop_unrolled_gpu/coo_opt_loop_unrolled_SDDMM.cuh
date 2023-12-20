// coo_opt_loop_unrolled_SDDMM.cuh
#ifndef COO_OPT_LOOP_UNROLLED_SDDMM_H
#define COO_OPT_LOOP_UNROLLED_SDDMM_H

void compute_coo_opt_loop_unrolled(
    const int m,
    const int n,
    const int k,
    const int numElementsC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values);

#endif  // COO_OPT_LOOP_UNROLLED_SDDMM_H