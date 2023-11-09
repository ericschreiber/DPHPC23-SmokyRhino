// naive_csr_via_coo_SDDMM.cuh
#ifndef NAIVE_CSR_VIA_COO_SDDMM_H
#define NAIVE_CSR_VIA_COO_SDDMM_H

// Assumes matrixB is transposed
void compute_naive_csr_via_coo(
    const int m,
    const int n,
    const int k,
    const int numElementsC,
    const int rowPtrSizeC,
    const float* __restrict__ const matrixA_GPU_values,
    const float* __restrict__ const matrixB_transposed_GPU_values,
    const float* __restrict__ const matrixC_GPU_values,
    const int* __restrict__ const matrixC_GPU_row_indices,
    const int* __restrict__ const matrixC_GPU_col_indices,
    float* __restrict__ const matrixResult_GPU_values);

#endif  // NAIVE_CSR_VIA_COO_SDDMM_H