// naive_dense_dense.cuh
#ifndef NAIVE_DENSE_DENSE_CUH
#define NAIVE_DENSE_DENSE_CUH

void compute(
    int m,
    int n,
    int k,
    const float *const d_A,
    const float *const d_B,
    float *const d_C,
    float *const d_D);

#endif  // NAIVE_DENSE_DENSE_CUH