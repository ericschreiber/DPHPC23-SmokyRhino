#include <cuda_runtime.h>

#include <iostream>
#include <random>

#include "SM_L2/SM_L2.cuh"
#include "utils.h"
// with VWARP more K
__global__ void comp_kernel_COO(int const *__restrict__ row_ind, int const *__restrict__ col_ind, const float *val, float *val_out, const float *__restrict__ u, const float *__restrict__ v, int nnz, int n_rows, int n_cols, int k, int tile_stIdx, int tile_limIdx, const int *d_last_blockIdx, const int *active_row, int tile_no, int t_st, int act_rlimit, int sh_tile, int k_slc)
{
    unsigned int tId = threadIdx.x;
    unsigned int laneId = tId & 1;
    int block_st = 0;
    if (blockIdx.x == 0)
        block_st = tile_stIdx;
    else
        block_st = d_last_blockIdx[blockIdx.x - 1];
    int block_lim = d_last_blockIdx[blockIdx.x];

    __shared__ float sh_r[32 * 192];
    int WARP_ID = tId >> 5;
    int tid_in_WARP = tId & 31;
    int WARP_SIZE = 32;

    int step = blockDim.x >> 5;

    int t = tid_in_WARP;

    for (int i = WARP_ID; i < sh_tile && (blockIdx.x * sh_tile + i) < act_rlimit;
         i += step)
    {
        for (int w_r = 0; w_r < k_slc; w_r += WARP_SIZE)
        {
            sh_r[i * k_slc + t + w_r] =
                u[active_row[blockIdx.x * sh_tile + i] * k + t + t_st + w_r];
        }
    }

    int block_lim_upscaled_to_1024 = (block_lim + 1023) >> 10;
    for (int c = block_st + (tId >> 1); c < block_lim_upscaled_to_1024;
         c += (blockDim.x >> 1))
    {
        float sm1 = 0, sm2 = 0;
        if (c > block_lim)
        {
            sm1 = 0;
            sm2 = 0;
        }
        else
        {
            int row = row_ind[c];
            int col = col_ind[c];
            int sh_row = row - blockIdx.x * sh_tile;

            for (int t = laneId * k_slc / 2; t < (laneId + 1) * k_slc / 2; t += 8)
            {
                float4 rtmp1 = *((float4 *)&sh_r[sh_row * k_slc + t]);
                float4 ctmp1 = *((float4 *)&v[col * k + t_st + t]);
                sm1 += rtmp1.x * ctmp1.x + rtmp1.y * ctmp1.y + rtmp1.z * ctmp1.z +
                       rtmp1.w * ctmp1.w;
                printf("calculating sm1: %f * %f + %f * %f + %f * %f + %f * %f\n", rtmp1.x, ctmp1.x, rtmp1.y, ctmp1.y, rtmp1.z, ctmp1.z, rtmp1.w, ctmp1.w);

                float4 rtmp2 = *((float4 *)&sh_r[sh_row * k_slc + t + 4]);
                float4 ctmp2 = *((float4 *)&v[col * k + t_st + t + 4]);
                sm2 += rtmp2.x * ctmp2.x + rtmp2.y * ctmp2.y + rtmp2.z * ctmp2.z +
                       rtmp2.w * ctmp2.w;
                printf("calculating sm2: %f * %f + %f * %f + %f * %f + %f * %f\n", rtmp2.x, ctmp2.x, rtmp2.y, ctmp2.y, rtmp2.z, ctmp2.z, rtmp2.w, ctmp2.w);
            }

            if (tId > block_lim)
            {
                printf("Resetting sm1 and sm2 to 0\n");
                sm1 = 0;
                sm2 = 0;
            }
        }
        sm1 += __shfl_xor_sync(0xFFFFFFFF, sm1,
                               1);  // not all threads of one warp are in the loop
                                    // with the old shuffle
        // this was not synced. Not it is and therefore it does not
        // work.
        sm2 += __shfl_xor_sync(0xFFFFFFFF, sm2, 1);

        val_out[c] = val[c] * (sm1 + sm2);
        val_out[c] += (sm1 + sm2);
    }
}

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
    const int *__restrict__ const d_lastIdx_block_tile)
{
    dim3 block(BLOCKSIZE, 1, 1), grid(1, 1, 1);
    int sum = 0;

    int k_slice = SM_CAPACITY / actv_row_size;

    for (int tile = 0; tile < n_tile; ++tile)
    {
        int nnz_tile = tS.lastIdx_tile[tile + 1] - tS.lastIdx_tile[tile];
        // grid.x = (nnz_tile + BLOCKSIZE - 1) / BLOCKSIZE;
        int active_block_this_tile = tS.n_actv_row[tile] / actv_row_size + 1;

        grid.x = active_block_this_tile;
        std::cout << "Tile: " << tile
                  << " active blocks: " << active_block_this_tile
                  << " active rows: " << tS.n_actv_row[tile] << std::endl;
        for (int t_st = 0; t_st < k; t_st += k_slice)
        {
            std::cout << "Start Kernel with t_st: " << t_st << " in tile: " << tile
                      << std::endl;
            comp_kernel_COO<<<grid, block, 0, stream[0]>>>(
                d_row_ind, d_col_ind, d_val, out, d_W, d_H, S.nnz, S.n_rows, S.n_cols, k, tS.lastIdx_tile[tile], tS.lastIdx_tile[tile + 1], &(d_lastIdx_block_tile[(tile)*tS.max_active_block]), d_active_row + sum, tile, t_st, tS.n_actv_row[tile], actv_row_size, k_slice);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        sum += tS.n_actv_row[tile];
    }
}