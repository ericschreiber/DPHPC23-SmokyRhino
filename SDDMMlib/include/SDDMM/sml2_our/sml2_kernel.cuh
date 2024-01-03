// sml2_kernel.cuh
#ifndef SML2_KERNEL_CUH
#define SML2_KERNEL_CUH

__global__ void compute_lml2(
    int* row_ptr,
    int* col_idx,
    int t_i,
    float* result);

#endif  // SML2_KERNEL_CUH