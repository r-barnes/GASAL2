#pragma once

__global__ void test_DEV_GET_SUB_SCORE_LOCAL(bool *d_good){
  *d_good = true;

  const uint32_t bases[5] = {'A' & 0xF, 'C' & 0xF, 'G' & 0xF, 'T' & 0xF};
  for(int i=0;i<4;i++)
  for(int j=0;j<4;j++){
    const auto score = DEV_GET_SUB_SCORE_LOCAL(bases[i], bases[j]);
    if(i==j){
      *d_good &= score==_cudaMatchScore;
    } else {
      *d_good &= score==-_cudaMismatchScore;
    }
  }

  for(int i=0;i<4;i++){
    const auto score1 = DEV_GET_SUB_SCORE_LOCAL(bases[i], 'N' & 0xF);
    const auto score2 = DEV_GET_SUB_SCORE_LOCAL('N' & 0xF, bases[i]);
    #ifdef N_PENALTY
      *d_good &= score1==-N_PENALTY;
      *d_good &= score2==-N_PENALTY;
    #else
      *d_good &= score1==0;
      *d_good &= score2==0;
    #endif
  }
}



// __device__ void compute_local_cell(
//   const uint32_t gbase,
//   const uint32_t rbase,
//   const int32_t  p,
//   short   &e,
//   int32_t &h,
//   int32_t &f
// ){


// __global__ void test_compute_local_cell(){

// }