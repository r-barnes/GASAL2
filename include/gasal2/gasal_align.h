#pragma once

#include <gasal2/args_parser.h>

#include <thrust/device_vector.h>

#include <cstdlib>

enum class AlignmentStatus {
  StreamFree,
  NotReady,
  Finished
};

void gasal_copy_subst_scores(const gasal_subst_scores &subst);

void gasal_aln_async(gasal_gpu_storage_t &gpu_storage, const uint32_t actual_query_batch_bytes, const uint32_t actual_target_batch_bytes, const uint32_t actual_n_alns, const Parameters &params);

inline void gasal_kernel_launcher(int32_t N_BLOCKS, int32_t BLOCKDIM, algo_type algo, CompStart start, gasal_gpu_storage_t *gpu_storage, int32_t actual_n_alns, int32_t k_band);

AlignmentStatus gasal_is_aln_async_done(gasal_gpu_storage_t &gpu_storage);
