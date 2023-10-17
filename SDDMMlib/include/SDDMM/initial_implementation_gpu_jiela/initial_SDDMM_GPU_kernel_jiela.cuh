// initial_SDDMM_GPU_kernel_jiela.cuh
#ifndef INITIAL_SDDMM_GPU_KERNEL_JIELA_CUH
#define INITIAL_SDDMM_GPU_KERNEL_JIELA_CUH

void compute_blockwise(
    int m,
    int n,
    int k,
    float *d_A,
    float *d_B,
    float *d_C,
    int *d_rowPtr,
    int *d_colIdx,
    float *d_result)

#endif  // INITIAL_SDDMM_GPU_KERNEL_JIELA_CUH