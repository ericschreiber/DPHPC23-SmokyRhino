// sml2_kernel.cuh
#ifndef SML2_KERNEL_CUH
#define SML2_KERNEL_CUH

__global__ void compute_lml2(
    float* matrix_A,
    float* matrix_B,
    int* row_ptr,
    int* col_idx,
    int t_i,
    float* result,
    int start_row,
    int start_col,
    int t_k_by_4,
    int m);

#endif  // SML2_KERNEL_CUH