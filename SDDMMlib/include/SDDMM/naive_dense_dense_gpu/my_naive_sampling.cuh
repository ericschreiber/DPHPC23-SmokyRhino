// my_naive_sampling.cuh
#ifndef MY_NAIVE_SAMPLING_CUH
#define MY_NAIVE_SAMPLING_CUH

void my_naive_sampling(
    const int size,
    const float* __restrict__ const A,
    float* __restrict__ const B);

#endif  // MY_NAIVE_SAMPLING_CUH