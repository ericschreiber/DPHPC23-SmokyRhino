#ifndef SM_L2_UTIL_H
#define SM_L2_UTIL_H

#include <bits/stdc++.h>
#include <sys/time.h>
#include <time.h>

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>
using namespace std;

inline double seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

class Matrix
{
    public:
        long n_rows, n_cols;
        long nnz;

        vector<int> rows;
        vector<int> cols;
        vector<float> vals;
};

class TiledMatrix
{
    public:
        int ntile_c;
        int ntile_r;
        int max_active_block;
        int max_active_row;
        long nnz;

        vector<int> rows;
        vector<int> cols;
        vector<float> vals;
        vector<int> row_holder;
        vector<int> active_row;
        vector<int> lastIdx_block_tile;
        vector<int> n_actv_row;
        vector<int> lastIdx_tile;
        vector<int> tiled_ind;

        TiledMatrix(Matrix S, int tile_sizeX, int tile_sizeY, int actv_row_size, int BLOCKSIZE)
        {
            ntile_c = S.n_cols / tile_sizeX + 1;
            ntile_r = S.n_rows / tile_sizeY + 1;
            max_active_block = (S.n_rows / actv_row_size + 1);
            lastIdx_block_tile.resize((ntile_c + 1) * (S.n_rows / actv_row_size + 1));
            lastIdx_tile.resize(ntile_c + 1);
            rows.resize(S.nnz + ntile_c * BLOCKSIZE - 1);
            cols.resize(S.nnz + ntile_c * BLOCKSIZE - 1);
            vals.resize(S.nnz + ntile_c * BLOCKSIZE - 1);
            tiled_ind.resize(S.nnz + ntile_c * BLOCKSIZE - 1);
            active_row.resize(S.n_rows * ntile_c);
            row_holder.resize(S.n_rows);
            n_actv_row.resize(ntile_c);
        }
};

static void make_HTasH(const vector<float> H, vector<float> &H_t, int n_cols, int k)
{
    // """Transpose of H"""

    for (long r = 0; r < n_cols; ++r)
    {
        for (long t = 0; t < k; ++t)
            H_t[t * n_cols + r] = H[r * k + t];  //-1;
    }
}

static void make_CSR(vector<int> rows, vector<int> cols, vector<float> vals, long nnz, long n_rows, int *row_ptr, int *row_holder)
{
    // assuming sorted
    // if CSR
    long idx = 0;
    row_ptr[0] = 0;
    int holder = 0;
    int r = rows[idx];

    while (idx < nnz)
    {
        row_holder[holder] = r;
        while (rows[idx] == r && idx < nnz)
        {
            idx++;
        }
        holder++;
        row_ptr[holder] = idx;
        r = rows[idx];
    }
    row_ptr[holder + 1] = idx;

    // Correct CSR conversion. But they do something else.

    // // Compute the row pointer array for the sampling matrix
    // std::vector<int> matrixC_CPU_row_ptr;
    // int ptr = 0;
    // matrixC_CPU_row_ptr.push_back(0);

    // int r_temp = rows[0];
    // int idx = 0;
    // row_holder[idx] = 0;

    // for (int i = 0; i < n_rows; i++)
    // {
    //     if (ptr < nnz && i < rows[ptr])
    //     {
    //         matrixC_CPU_row_ptr.push_back(matrixC_CPU_row_ptr[i]);
    //     }
    //     else if (ptr >= nnz)
    //     {
    //         matrixC_CPU_row_ptr.push_back(matrixC_CPU_row_ptr[i]);
    //     }
    //     else
    //     {
    //         int counter = 0;
    //         while (ptr < nnz && i == rows[ptr])
    //         {
    //             counter++;
    //             ptr++;
    //         }
    //         matrixC_CPU_row_ptr.push_back(matrixC_CPU_row_ptr[i] + counter);
    //     }
    //     // make the row_holder array (the non empty rows)
    //     if (r_temp != rows[i])
    //     {
    //         row_holder[idx] = i;
    //         idx++;
    //         r_temp = rows[i];
    //     }
    // }

    // // check that matrixC_CPU_row_ptr is n_rows + 1
    // if (matrixC_CPU_row_ptr.size() != n_rows + 1)
    // {
    //     std::cout << "ERROR: matrixC_CPU_row_ptr.size() != n_rows + 1" << std::endl;
    //     exit(1);
    // }

    // // Copy the row_ptr vector to the row_ptr array
    // std::copy(matrixC_CPU_row_ptr.begin(), matrixC_CPU_row_ptr.end(), row_ptr);
}

static void make_2DBlocks(int *row_ptr, int *row_ind, int *col_ind, float *val_ind, long nnz, long n_rows, long n_cols)
{
    int *new_row_ind = new int[nnz];
    int *new_col_ind = new int[nnz];
    float *new_val_ind = new float[nnz];
    int block_dimX = 2;
    int block_dimY = 2;
    int n_blockX = n_rows / block_dimY + 1;
    int n_block = (n_rows / block_dimY + 1) * (n_cols / block_dimX + 1);
    int *new_ind = new int[nnz];
    int *list = new int[n_block];
    int block_no = 0;

    // initialization
    for (int i = 0; i < n_block; ++i)
        list[i] = 0;

    // #pragma omp parallel for
    for (int r = 0; r < n_rows; ++r)
    {
        int block_noY = r / block_dimY;
        for (long idx = row_ptr[r]; idx < row_ptr[r + 1]; ++idx)
        {
            int block_noX = col_ind[idx] / block_dimX;  // - 1;
            block_no = block_noY * n_blockX + block_noX;
            cout << "processing " << r << " " << col_ind[idx] << " ::: "
                 << block_noY << " " << block_noX << " " << block_no << endl;
            list[block_no]++;
            // new_ind[n_rows * i + count[i]++] = idx;

            // list[bucket_no]++ = idx;
            // while((idx-tiled_bin[tile_no-1][c]) < TS && idx < R.col_ptr[c+1]){ //CHANGED for nnz tiles
        }
    }
    for (int i = 0; i < n_block; ++i)
        cout << " adf " << i << " " << list[i] << endl;
}

static void rewrite_matrix_2D(int *row_ptr, int *row_ind, int *col_ind, float *val_ind, int *new_rows, int *new_cols, float *new_vals, long nnz, long n_rows, long n_cols, int TS, int *tiled_ind, int *lastIdx_tile)
{
    int TS_r = 2;
    long new_idx = 0, idx = 0;
    int n_tile_c = n_cols / TS + 1, n_tile_r = n_rows / TS_r + 1, tile_no = 0;
    int *row_lim = new int[(n_tile_c + 1) * n_rows];
    lastIdx_tile[0] = 0;
    for (int i = 0; i < nnz; ++i)
        cout << "orig " << i << " : " << row_ind[i] << " " << col_ind[i] << endl;

    // #pragma omp parallel for
    for (int tile_lim = TS; tile_lim <= (n_cols + TS - 1); tile_lim += TS)
    {
        int tile_no_c = tile_lim / TS;
        for (int tile_lim_r = 0; tile_lim_r < n_rows + TS_r - 1; tile_lim_r += TS_r)
        {
            tile_no = tile_no_c * n_tile_r + tile_lim_r / TS_r;
            for (int r = tile_lim_r; r < tile_lim_r + TS_r && r < n_rows; ++r)
            {
                if (tile_lim == TS)
                {
                    idx = row_ptr[r];
                    row_lim[r] = idx;
                }
                else
                    idx = row_lim[(tile_no - 1) * n_rows + r];
                while (col_ind[idx] < tile_lim && idx < row_ptr[r + 1])
                {
                    cout << " inside " << r << ":" << new_idx << " " << idx << endl;
                    tiled_ind[new_idx] = idx;
                    // new_rows[new_idx] = row_ind[idx];
                    // new_cols[new_idx] = col_ind[idx];
                    // new_vals[new_idx] = val_ind[idx];
                    new_idx++;
                    idx++;
                }
                row_lim[tile_no_c * n_rows + r] = idx;
            }
            // lastIdx_tile[tile_no] = new_idx;
        }
    }
    // for (int i = 0; i <10; ++i)
    //       cout  << i <<" : "<<row_ind[i] <<" " << col_ind[i] << " new: " << tiled_ind[i]
    //        <<" , "<< new_rows[i] <<" "<< new_cols[i]<< endl;
    delete (row_lim);
}
static void rewrite_col_sorted_matrix(int *row_ptr, int *row_ind, int *col_ind, float *val_ind, int *new_rows, int *new_cols, float *new_vals, long nnz, long n_rows, long n_cols, int TS, int *tiled_ind, int *lastIdx_tile, int block, long &new_nnz)
{
    long new_idx = 0, idx = 0;
    int tile_no = 0;
    lastIdx_tile[0] = 0;

    // #pragma omp parallel for
    for (int tile_lim = TS; tile_lim <= (n_cols + TS - 1); tile_lim += TS)
    {
        tile_no = tile_lim / TS;
        // being lazy ..can skip the part
        for (int c = 0; c < tile_lim && c < n_cols; ++c)
        {
            while (col_ind[idx] == c)
            {
                tiled_ind[new_idx] = idx;
                new_rows[new_idx] = row_ind[idx];
                new_cols[new_idx] = col_ind[idx];
                new_vals[new_idx] = val_ind[idx];
                new_idx++;
                idx++;
            }
        }
        lastIdx_tile[tile_no] = new_idx;
        if (tile_no < 5)
            cout << "lastIdx_tile " << tile_no << " " << lastIdx_tile[tile_no] << endl;
    }
    new_nnz = nnz;
}

static int rewrite_matrix_1D(const Matrix S, TiledMatrix &tS, int *row_ptr, int TS, int *row_holder, int actv_row_size)
{
    long new_idx = 0, idx = 0;
    int max_block_inAtile = S.n_rows / actv_row_size + 1;
    int tile_no = 0;
    tS.lastIdx_tile[0] = 0;
    long n_rows = S.n_rows;
    long n_cols = S.n_cols;
    vector<int> row_lim(n_rows);

    // #pragma omp parallel for
    for (int tile_lim = TS; tile_lim <= (n_cols + TS - 1); tile_lim += TS)
    {
        int block_count = 0;
        int cur_block = 0, r = 0;
        tile_no = tile_lim / TS;
        tS.n_actv_row[tile_no - 1] = 0;

        if (tile_no == 1)
            tS.lastIdx_block_tile[tile_no * max_block_inAtile + 0] = 0;
        else
            tS.lastIdx_block_tile[tile_no * max_block_inAtile + 0] = new_idx;

        for (int holder = 0; holder < n_rows; ++holder)
        {
            r = row_holder[holder];
            // for(int r = 0; r <n_rows; ++r){
            if (tile_lim == TS)
            {
                idx = row_ptr[holder];
                row_lim[holder] = idx;
            }
            else
                idx = row_lim[holder];

            while (S.cols[idx] < tile_lim && idx < row_ptr[holder + 1])
            {
                tS.tiled_ind[new_idx] = idx;
                tS.rows[new_idx] = tS.n_actv_row[tile_no - 1];  // S.rows[idx];
                tS.cols[new_idx] = S.cols[idx];

                // ******* bit mask start *******
                // row = tS.n_actv_row[tile_no-1];;//S.rows[idx];
                // col = S.cols[idx]%95;
                // c[0] = (col>>0) & 0xff;
                // c[1] = (row>>16) & 0xFF;
                // c[2] = (row>>8) & 0xFF;
                // c[3] = (row>>0) & 0xff;
                // final_int = ((c[1]) << 24) | ((c[2]) << 16) | c[3] << 8 | c[0];
                // tS.rows[new_idx] = final_int;
                // // ******* bit mask finish ******

                tS.vals[new_idx] = S.vals[idx];
                new_idx++;
                idx++;
            }
            if (idx != row_lim[holder])
            {
                tS.active_row[(tile_no - 1) * n_rows + tS.n_actv_row[tile_no - 1]++] = r;
                // passive_row[(tile_no-1) * n_rows + holder] = tS.n_actv_row[tile_no-1]-1;
                cur_block++;
            }
            row_lim[holder] = idx;
            if (cur_block >= actv_row_size)
            {
                cur_block = 0;
                tS.lastIdx_block_tile[(tile_no - 1) * max_block_inAtile + block_count] = new_idx;
                block_count++;
            }

            if (holder == n_rows - 1 && cur_block > 0 && cur_block < actv_row_size)
                tS.lastIdx_block_tile[(tile_no - 1) * max_block_inAtile + block_count] = new_idx;
        }
        if (tS.n_actv_row[tile_no - 1] > tS.max_active_row)
            tS.max_active_row = tS.n_actv_row[tile_no - 1];
        tS.lastIdx_tile[tile_no] = new_idx;
    }
    tS.nnz = S.nnz;
    return tS.max_active_row;
}

static void make_2Dtile(int *row_ptr, int *row_ind, int *col_ind, float *val_ind, long nnz, long n_rows, long n_cols, int TS, int *row_lim)
{
    int *tiled_matrix = new int[TS * n_rows];
    // #pragma omp parallel for
    for (int r = 0; r < n_rows; ++r)
    {
        long idx = row_ptr[r];
        row_lim[r] = idx;
        for (int tile = TS; tile <= (n_cols + TS - 1); tile += TS)
        {
            while (col_ind[idx] < tile && idx < row_ptr[r + 1])
            {
                // cout << "processing: " << r <<" "<<col_ind[idx] << " "<<tile<<" "<<idx << endl;
                idx++;
            }
            int tile_no = tile / TS - 1;
            row_lim[tile_no * n_rows + r] = idx;
        }
    }
    // for (int ii = 0; ii < 4; ++ii)
    //  for (int i = 0; i < 4; ++i)
    //    {
    //       cout << ii << "i: "<< i<<" row lim " <<row_lim[ii * n_rows +i]<< endl;
    //    }
    //    cout << endl;
}

#endif  // SM_L2_UTIL_H