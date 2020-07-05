#pragma once

#include <cstdint>

// Template-meta-programming types construction from Int values
// This allows to cut down kernel code at compilation time.

__constant__ int32_t _cudaGapO;          // gap open penalty
__constant__ int32_t _cudaGapOE;         // sum of gap open and extension penalties
__constant__ int32_t _cudaGapExtend;     // sum of gap extend
__constant__ int32_t _cudaMatchScore;    // score for a match
__constant__ int32_t _cudaMismatchScore; // penalty for a mismatch

#define MINUS_INF SHRT_MIN

#define N_VALUE (N_CODE & 0xF)

__device__ int32_t DEV_GET_SUB_SCORE_LOCAL(const uint32_t rbase, const uint32_t gbase){
  int32_t score = (rbase == gbase) ? _cudaMatchScore : -_cudaMismatchScore;
  #ifdef N_PENALTY
    score = (rbase == N_VALUE || gbase == N_VALUE) ? -N_PENALTY : score;
  #else
    score = (rbase == N_VALUE || gbase == N_VALUE) ? 0 : score;
  #endif
  return score;
}

#ifdef N_PENALTY
	#define DEV_GET_SUB_SCORE_GLOBAL(score, rbase, gbase) \
		score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\
	score = ((rbase == N_VALUE) || (gbase == N_VALUE)) ? -N_PENALTY : score;\

#else
	#define DEV_GET_SUB_SCORE_GLOBAL(score, rbase, gbase) \
		score = (rbase == gbase) ?_cudaMatchScore : -_cudaMismatchScore;\

#endif

#define FIND_MAX(curr, gidx) \
	maxXY_y = (maxHH < curr) ? gidx : maxXY_y;\
maxHH = (maxHH < curr) ? curr : maxHH;


// Kernel files

#include "kernels/banded.cu"
#include "kernels/get_tb.cu"
#include "kernels/global.cu"
#include "kernels/ksw_kernel_template.cu"
#include "kernels/local_kernel_template.cu"
#include "kernels/pack_rc_seqs.cu"
#include "kernels/semiglobal_kernel_template.cu"
#include "kernels/tests.cu"