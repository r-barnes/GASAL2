#pragma once

struct MaxValues {
  int32_t HH      = 0;
  int32_t XY_x    = 0;
  int32_t XY_y    = 0;
  int32_t prev_HH = 0;

  int32_t HH_second      = 0; //TODO: Drop "__attribute__((unused))"
  int32_t XY_x_second    = 0; //TODO: Drop "__attribute__((unused))"
  int32_t XY_y_second    = 0; //TODO: Drop "__attribute__((unused))"
  int32_t prev_HH_second = 0; //TODO: Drop "__attribute__((unused))"
};



__device__ int32_t compute_local_cell(
  const uint32_t gbase,
  const uint32_t rbase,
  const int32_t  p,
  short   &e,
  int32_t &h,
  int32_t &f
){
  const auto subScore = DEV_GET_SUB_SCORE_LOCAL(rbase, gbase);
  int32_t tmp_hm = p + subScore;
  h = max(tmp_hm, f);
  h = max(h, e);
  h = max(h, 0);
  f = max(tmp_hm - _cudaGapOE, f - _cudaGapExtend);
  e = max(tmp_hm - _cudaGapOE, e - _cudaGapExtend);
  return tmp_hm;
}



__device__ void core_local_compute_start(
  const uint32_t gpac,
  const uint32_t rbase,
  int32_t p[9],
  short   &e,
  int32_t h[9],
  int32_t f[9],
  const int32_t l,
  const int32_t m,
  MaxValues &mv,
  const int32_t gidx
){
  uint32_t gbase = (gpac >> l) & 0xF;
  compute_local_cell(gbase, rbase, p[m], e, h[m], f[m]);
  mv.XY_y = (mv.HH < h[m]) ? gidx + (m-1) : mv.XY_y;
  mv.HH   = (mv.HH < h[m]) ? h[m] : mv.HH;
  p[m] = h[m-1];
}



__device__ void CORE_LOCAL_COMPUTE_TB(
  const uint32_t gpac,
  const uint32_t rbase,
  int32_t p[9],
  short   &e,
  int32_t h[9],
  int32_t f[9],
  const int32_t l,
  const int32_t m,
  MaxValues &mv,
  uint &direction_reg,
  const int32_t gidx
){
  const auto fm_old = f[m];
  const auto e_old = e;
  uint32_t gbase = (gpac >> l) & 0xF;
  const auto tmp_hm = compute_local_cell(gbase, rbase, p[m], e, h[m], f[m]);
  uint32_t m_or_x = tmp_hm >= p[m] ? 0 : 1;
  direction_reg |= h[m] == tmp_hm ? m_or_x << (28 - ((m - 1) << 2)) : (h[m] == fm_old ? (uint32_t)3 << (28 - ((m - 1) << 2)) : (uint32_t)2 << (28 - ((m - 1) << 2)));
  direction_reg |= (tmp_hm - _cudaGapOE) > (fm_old - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (31 - ((m - 1) << 2));
  direction_reg |= (tmp_hm - _cudaGapOE) > (e_old - _cudaGapExtend) ?  (uint32_t)0 : (uint32_t)1 << (30 - ((m - 1) << 2));
  mv.XY_y = (mv.HH < h[m]) ? gidx + (m-1) : mv.XY_y;
  mv.HH   = (mv.HH < h[m]) ? h[m] : mv.HH;
  p[m] = h[m-1];
}



template<Bool B>
__device__ void compute_local_row(
  const uint32_t gpac,
  const uint32_t rbase,
  const int32_t  gidx,
  short   &e,
  MaxValues &mv,
  int32_t h[9],
  int32_t f[9],
  int32_t p[9]
){
  #pragma unroll 8
  for(int32_t l = 28, m = 1; m < 9; l -= 4, m++) {
    const uint32_t gbase = (gpac >> l) & 0xF;
    compute_local_cell(gbase, rbase, p[m], e, h[m], f[m]);
    mv.XY_y = (mv.HH < h[m]) ? gidx + (m-1) : mv.XY_y;
    mv.HH = (mv.HH < h[m]) ? h[m] : mv.HH;
    p[m] = h[m-1];
    if (B==Bool::TRUE){
      const bool override_second = (mv.HH_second < h[m]) && (mv.HH > h[m]);
      mv.XY_y_second = (override_second) ? gidx + (m-1) : mv.XY_y_second;
      mv.HH_second = (override_second) ? h[m] : mv.HH_second;
    }
  }
}



template<Bool B>
__device__ void compute_local_block(
  short2 *const global,
  const uint32_t gpac,
  const uint32_t rpac,
  int32_t  &ridx,
  const int32_t  gidx,
  MaxValues &mv,
  int32_t h[9],
  int32_t f[9],
  int32_t p[9]
){
  for (int32_t k = 28; k >= 0; k -= 4) {
    const uint32_t rbase = (rpac >> k) & 0xF; //Get a column from the query sequence

    //-----load intermediate values--------------
    short2 HD = global[ridx];
    h[0] = HD.x;
    compute_local_row<B>(gpac, rbase, gidx, HD.y, mv, h, f, p);
    //----------save intermediate values------------
    HD.x = h[8];
    global[ridx] = HD;
    //---------------------------------------------

    mv.XY_x = (mv.prev_HH < mv.HH) ? ridx : mv.XY_x;//end position on query_batch sequence corresponding to current maximum score

    if (B==Bool::TRUE)
    {
      mv.XY_x_second = (mv.prev_HH_second < mv.HH) ? ridx : mv.XY_x_second;
      mv.prev_HH_second = max(mv.HH_second, mv.prev_HH_second);
    }
    mv.prev_HH = max(mv.HH, mv.prev_HH);
    ridx++;
  }
}




template<Bool B>
__device__ void compute_something(
  const uint32_t rpac,
  const uint32_t rpac_shift,
  const uint32_t gpac,
  short2 *const global,
  int32_t  &ridx,
  int32_t p[9],
  int32_t h[9],
  int32_t f[9],
  MaxValues &mv,
  const int32_t gidx,
  uint &direction
){
  uint32_t rbase = (rpac >> rpac_shift) & 0xF;//get a base from query_batch sequence
  auto HD = global[ridx];
  h[0] = HD.x;
  auto e = HD.y;
  int32_t l, m;
  for (l = 28, m = 1; m < 9; l -= 4, m++) {
    CORE_LOCAL_COMPUTE_TB(gpac, rbase, p, e, h, f, l, m, mv, direction, gidx);
    if (B==Bool::TRUE){
      bool override_second = (mv.HH_second < h[m]) && (mv.HH > h[m]);
      mv.XY_y_second = (override_second) ? gidx + (m-1) : mv.XY_y_second;
      mv.HH_second   = (override_second) ? h[m] : mv.HH_second;
    }
  }
  HD.x = h[m-1];
  HD.y = e;
  global[ridx] = HD;

  mv.XY_x = (mv.prev_HH < mv.HH) ? ridx : mv.XY_x;//end position on query_batch sequence corresponding to current maximum score

  if (B==Bool::TRUE){
    mv.XY_x_second    = (mv.prev_HH_second < mv.HH) ? ridx : mv.XY_x_second;
    mv.prev_HH_second = max(mv.HH_second, mv.prev_HH_second);
  }
  mv.prev_HH = max(mv.HH, mv.prev_HH);
  ridx++;
}



__device__ void find_start(
  MaxValues mv,
  uint32_t query_batch_regs,
  uint32_t target_batch_regs,
  uint32_t packed_target_batch_idx,
  uint32_t packed_query_batch_idx,
  short2 *global,
  int32_t h[9],
  int32_t f[9],
  int32_t p[9],
  int32_t ridx,
  int32_t gidx,
  uint32_t    *packed_query_batch,
  uint32_t    *packed_target_batch,
  gasal_res_t *device_res
){
  const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  const short2 initHD = make_short2(0, 0);

  int32_t rend_pos = mv.XY_x; //end position on query_batch sequence
  int32_t gend_pos = mv.XY_y; //end position on target_batch sequence
  int32_t fwd_score = mv.HH;  // the computed score

  //the index of 32-bit word containing the end position on query_batch sequence
  int32_t rend_reg = ((rend_pos >> 3) + 1) < query_batch_regs ? ((rend_pos >> 3) + 1) : query_batch_regs;
  //the index of 32-bit word containing to end position on target_batch sequence
  int32_t gend_reg = ((gend_pos >> 3) + 1) < target_batch_regs ? ((gend_pos >> 3) + 1) : target_batch_regs;

  packed_query_batch_idx += (rend_reg - 1);
  packed_target_batch_idx += (gend_reg - 1);

  mv.HH      = 0;
  mv.prev_HH = 0;
  mv.XY_x    = 0;
  mv.XY_y    = 0;

  for(int32_t i = 0; i < MAX_QUERY_LEN; i++) {
    global[i] = initHD;
  }
  //------starting from the gend_reg and rend_reg, align the sequences in the reverse direction and exit if the max score >= fwd_score------
  gidx = ((gend_reg << 3) + 8) - 1;
  for(int32_t i = 0; i < gend_reg && mv.HH < fwd_score; i++) {
    for(int32_t m = 0; m < 9; m++) {
        h[m] = 0;
        f[m] = 0;
        p[m] = 0;
    }
    register uint32_t gpac = packed_target_batch[packed_target_batch_idx - i];//load 8 packed bases from target_batch sequence
    gidx = gidx - 8;
    ridx = (rend_reg << 3) - 1;
    int32_t global_idx = 0;
    for(int32_t j = 0; j < rend_reg && mv.HH < fwd_score; j+=1) {
      register uint32_t rpac =packed_query_batch[packed_query_batch_idx - j];//load 8 packed bases from query_batch sequence
      //--------------compute a tile of 8x8 cells-------------------
      for(int32_t k = 0; k <= 28 && mv.HH < fwd_score; k += 4) {
        uint32_t rbase = (rpac >> k) & 0xF;//get a base from query_batch sequence
        //----------load intermediate values--------------
        auto HD = global[global_idx];
        h[0] = HD.x;
        auto e = HD.y;

        #pragma unroll 8
        for (int32_t l = 0, m = 1; l <= 28; l += 4, m++) {
          core_local_compute_start(gpac,rbase,p,e,h,f,l,m,mv,gidx);
        }

        //------------save intermediate values----------------
        HD.x = h[8];
        HD.y = e;
        global[global_idx] = HD;
        //----------------------------------------------------
        mv.XY_x = (mv.prev_HH < mv.HH) ? ridx : mv.XY_x;//start position on query_batch sequence corresponding to current maximum score
        mv.prev_HH = max(mv.HH, mv.prev_HH);
        ridx--;
        global_idx++;
      }
    }
  }

  device_res->query_batch_start[tid] = mv.XY_x;//copy the start position on query_batch sequence to the output array in the GPU mem
  device_res->target_batch_start[tid] = mv.XY_y;//copy the start position on target_batch sequence to the output array in the GPU mem
}



/* typename meaning :
    - S is WITH_ or WIHTOUT_START
    - B is for computing the Second Best Score. Its values are on enum FALSE(0)/TRUE(1).
    (sidenote: it's based on an enum instead of a bool in order to generalize its type from its Int value, with Int2Type meta-programming-template)
*/
template <CompStart S, Bool B>
__global__ void gasal_local_kernel(
  uint32_t    *packed_query_batch,
  uint32_t    *packed_target_batch,
  uint32_t    *query_batch_lens,
  uint32_t    *target_batch_lens,
  uint32_t    *query_batch_offsets,
  uint32_t    *target_batch_offsets,
  gasal_res_t *device_res,
  gasal_res_t *device_res_second,
  uint4       *packed_tb_matrices,
  int n_tasks
){
  const uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= n_tasks) return;

  MaxValues mv;

  int tile_no = 0;

  int32_t ridx;
  int32_t gidx;
  const short2 initHD = make_short2(0, 0);

  uint32_t packed_target_batch_idx = target_batch_offsets[tid] >> 3; //starting index of the target_batch sequence
  uint32_t packed_query_batch_idx  = query_batch_offsets[tid] >> 3;//starting index of the query_batch sequence
  uint32_t read_len = query_batch_lens[tid];
  uint32_t ref_len  = target_batch_lens[tid];
  uint32_t query_batch_regs  = (read_len >> 3) + (read_len%8!=0);//number of 32-bit words holding query_batch sequence
  uint32_t target_batch_regs = (ref_len >> 3) + (ref_len%8!=0);//number of 32-bit words holding target_batch sequence
  //-----arrays for saving intermediate values------
  short2 global[MAX_QUERY_LEN];
  int32_t h[9];
  int32_t f[9];
  int32_t p[9];

  for (int i = 0; i < MAX_QUERY_LEN; i++) {
    global[i] = initHD;
  }

  for (int i = 0; i < target_batch_regs; i++) { //target_batch sequence in rows
    for (int m = 0; m < 9; m++) {
      h[m] = 0;
      f[m] = 0;
      p[m] = 0;
    }

    register uint32_t gpac = packed_target_batch[packed_target_batch_idx + i];//load 8 packed bases from target_batch sequence
    gidx = i << 3;
    ridx = 0;

    for(int j = 0; j < query_batch_regs; j+=1) { //query_batch sequence in columns
      register uint32_t rpac = packed_query_batch[packed_query_batch_idx + j];//load 8 bases from query_batch sequence

      //--------------compute a tile of 8x8 cells-------------------
      if (S==CompStart::WITH_TB) {
        uint4 direction = make_uint4(0, 0, 0, 0);

        compute_something<B>(rpac, 28, gpac, global, ridx, p, h, f, mv, gidx, direction.x);
        compute_something<B>(rpac, 24, gpac, global, ridx, p, h, f, mv, gidx, direction.y);
        compute_something<B>(rpac, 20, gpac, global, ridx, p, h, f, mv, gidx, direction.z);
        compute_something<B>(rpac, 16, gpac, global, ridx, p, h, f, mv, gidx, direction.w);

        packed_tb_matrices[(tile_no*n_tasks) + tid] = direction;
        tile_no++;

        direction = make_uint4(0,0,0,0);

        compute_something<B>(rpac, 12, gpac, global, ridx, p, h, f, mv, gidx, direction.x);
        compute_something<B>(rpac,  8, gpac, global, ridx, p, h, f, mv, gidx, direction.y);
        compute_something<B>(rpac,  4, gpac, global, ridx, p, h, f, mv, gidx, direction.z);
        compute_something<B>(rpac,  0, gpac, global, ridx, p, h, f, mv, gidx, direction.w);

        packed_tb_matrices[(tile_no*n_tasks) + tid] = direction;
        tile_no++;
      } else {
        compute_local_block<B>(global, gpac, rpac, ridx, gidx, mv, h, f, p);
      }
    }
  }

  device_res->aln_score[tid] = mv.HH;//copy the max score to the output array in the GPU mem
  device_res->query_batch_end[tid] = mv.XY_x;//copy the end position on query_batch sequence to the output array in the GPU mem
  device_res->target_batch_end[tid] = mv.XY_y;//copy the end position on target_batch sequence to the output array in the GPU mem

  if (B==Bool::TRUE){
    device_res_second->aln_score[tid] = mv.HH_second;
    device_res_second->query_batch_end[tid] = mv.XY_x_second;
    device_res_second->target_batch_end[tid] = mv.XY_y_second;
  }

  /*------------------Now to find the start position-----------------------*/
  if (S==CompStart::WITH_START){
    find_start(mv, query_batch_regs, target_batch_regs, packed_target_batch_idx, packed_query_batch_idx, global, h, f, p, ridx, gidx, packed_query_batch, packed_target_batch, device_res);
  }
}
