// initial_SDDMM_solver.cuh
#ifndef INITIAL_SDDMM_SOLVER_CUH
#define INITIAL_SDDMM_SOLVER_CUH

void compute(
    int n_rows,
    int m,
    int n,
    int k,
    const float *const matrixA_GPU,
    const float *const matrixB_GPU,
    const float *const values_matrixC_GPU,
    float *const values_matrixResult_GPU,
    float *const colIndices_matrixResult_GPU,
    float *const rowPtr_matrixResult_GPU);

#endif  // INITIAL_SDDMM_SOLVER_CUH