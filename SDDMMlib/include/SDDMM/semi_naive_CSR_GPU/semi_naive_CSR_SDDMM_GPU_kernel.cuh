// semi_naive_CSR_DDMM_GPU_kernel.cuh
#ifndef SEMI_NAIVE_CSR_SDDMM_GPU_KERNEL_CUH
#define SEMI_NAIVE_CSR_SDDMM_GPU_KERNEL_CUH

void compute_blockwise(
    int m,
    int n,
    int k,
    float *d_A,
    float *d_B,
    float *d_C,
    int *d_rowPtr,
    int *d_colIdx,
    float *d_result);

#endif  // SEMI_NAIVE_CSR_SDDMM_GPU_KERNEL_CUH