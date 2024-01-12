// SM_L2.cuh
#ifndef SM_L2_H
#define SM_L2_H

#include "SM_L2/SM_L2_util.h"

void compute_sm_l2(
    int BLOCKSIZE,
    int SM_CAPACITY,
    int actv_row_size,
    int n_tile,
    const Matrix &S,
    TiledMatrix &tS,
    const int k,
    cudaStream_t *stream,
    const int *__restrict__ const d_row_ind,
    const int *__restrict__ const d_col_ind,
    const float *__restrict__ const d_val,
    float *__restrict__ const out,
    const float *__restrict__ const d_W,
    const float *__restrict__ const d_H,
    const int *__restrict__ const d_active_row,
    const int *__restrict__ const d_lastIdx_block_tile);

#endif  // SM_L2_H