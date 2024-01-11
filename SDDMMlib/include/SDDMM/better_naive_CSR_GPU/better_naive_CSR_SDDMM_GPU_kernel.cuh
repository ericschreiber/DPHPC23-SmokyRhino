// better_naive_CSR_DDMM_GPU_kernel.cuh
#ifndef BETTER_NAIVE_CSR_SDDMM_GPU_KERNEL_CUH
#define BETTER_NAIVE_CSR_SDDMM_GPU_KERNEL_CUH

void compute_blockwise(
    int lines_per_block,
    int warps_per_line,
    int m,
    int k_aligned,
    float *d_A,
    float *d_B,
    int *d_rowPtr,
    int *d_colIdx,
    float *d_result);

#endif  // BETTER_NAIVE_CSR_SDDMM_GPU_KERNEL_CUH