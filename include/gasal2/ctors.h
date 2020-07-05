#pragma once

void gasal_init_streams(gasal_gpu_storage_v &gpu_storage_vec, int max_query_len, int max_target_len, int max_n_alns, const Parameters &params);

void gasal_destroy_streams(gasal_gpu_storage_v &gpu_storage_vec, const Parameters &params);
