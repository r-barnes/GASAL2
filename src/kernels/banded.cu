#pragma once

#include <gasal2/gasal.h>

#include <cstdint>

__global__ void gasal_banded_tiled_kernel(uint32_t *packed_query_batch, uint32_t *packed_target_batch,  uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, gasal_res_t *device_res, int n_tasks, const int32_t k_band_width)
{
	int32_t i, j, k, m, l;
	int32_t e;
	int32_t maxHH = 0;//initialize the maximum score to zero
	int32_t prev_maxHH = 0;
	int32_t subScore;
	int32_t ridx, gidx;
	short2 HD;
	short2 initHD = make_short2(0, 0);
	int32_t maxXY_x = 0;
	int32_t maxXY_y = 0;
	const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;//thread ID
	if (tid >= n_tasks) return;
	uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
	uint32_t packed_query_batch_idx = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
	uint32_t read_len = query_batch_lens[tid];
	uint32_t ref_len = target_batch_lens[tid];
	uint32_t query_batch_regs = (read_len >> 3) + (read_len&7 ? 1 : 0);//number of 32-bit words holding query_batch sequence
	uint32_t target_batch_regs = (ref_len >> 3) + (ref_len&7 ? 1 : 0);//number of 32-bit words holding target_batch sequence
	//-----arrays for saving intermediate values------
	short2 global[MAX_QUERY_LEN];
	int32_t h[9];
	int32_t f[9];
	int32_t p[9];
 	const int32_t k_other_band_width = (target_batch_regs - (query_batch_regs - k_band_width));

	//------------------------
	for (i = 0; i < MAX_QUERY_LEN; i++) {
		global[i] = initHD;
	}

	for (i = 0; i < target_batch_regs; i++) { //target_batch sequence in rows
		for (m = 0; m < 9; m++) {
			h[m] = 0;
			f[m] = 0;
			p[m] = 0;
		}

		uint32_t gpac =packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence
		gidx = i << 3;

		ridx = max(0, i - k_other_band_width+1) << 3;
		int32_t last_tile = min( k_band_width + i, (int32_t)query_batch_regs);
		for (j = ridx >> 3  ; j < last_tile; j++) { //query_batch sequence in columns --- the beginning and end are defined with the tile-based band, to avoid unneccessary calculations.

			uint32_t rpac =packed_query_batch[packed_query_batch_idx + j];//load 8 bases from query_batch sequence

				//--------------compute a tile of 8x8 cells-------------------
				for (k = 28; k >= 0; k -= 4) {
					uint32_t rbase = (rpac >> k) & 0x0F;//get a base from query_batch sequence
					//-----load intermediate values--------------
					HD = global[ridx];
					h[0] = HD.x;
					e = HD.y;
					//-------------------------------------------
					//int32_t prev_hm_diff = h[0] - _cudaGapOE;

					#pragma unroll 8
					for (l = 28, m = 1; m < 9; l -= 4, m++) {
						uint32_t gbase = (gpac >> l) & 15;//get a base from target_batch sequence
						subScore = DEV_GET_SUB_SCORE_LOCAL(rbase, gbase);//check equality of rbase and gbase
						//int32_t curr_hm_diff = h[m] - _cudaGapOE;
						f[m] = max(h[m]- _cudaGapOE, f[m] - _cudaGapExtend);//whether to introduce or extend a gap in query_batch sequence
						h[m] = p[m] + subScore;//score if rbase is aligned to gbase
						h[m] = max(h[m], f[m]);
						h[m] = max(h[m], 0);
						e = max(h[m - 1] - _cudaGapOE, e - _cudaGapExtend);//whether to introduce or extend a gap in target_batch sequence
						//prev_hm_diff=curr_hm_diff;
						h[m] = max(h[m], e);

						//The current maximum score and corresponding end position on target_batch sequence
						maxXY_y = (maxHH < h[m]) ? gidx + (m-1) : maxXY_y;
						maxHH   = (maxHH < h[m]) ? h[m]         : maxHH;

						p[m] = h[m-1];
					}
					//----------save intermediate values------------
					HD.x = h[m-1];
					HD.y = e;
					global[ridx] = HD;
					//---------------------------------------------
					maxXY_x = (prev_maxHH < maxHH) ? ridx : maxXY_x;//end position on query_batch sequence corresponding to current maximum score
					prev_maxHH = max(maxHH, prev_maxHH);
					ridx++;

			}
			//-------------------------------------------------------
			// 8*8 patch done

		}

	}

	device_res->aln_score[tid] = maxHH;//copy the max score to the output array in the GPU mem
	device_res->query_batch_end[tid] = maxXY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
	device_res->target_batch_end[tid] = maxXY_y;//copy the end position on target_batch sequence to the output array in the GPU mem
}