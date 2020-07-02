#pragma once

gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams);

void gasal_init_streams(gasal_gpu_storage_v *gpu_storage_vec,  int max_query_len, int max_target_len, int max_n_alns,  Parameters *params);

void gasal_destroy_streams(gasal_gpu_storage_v *gpu_storage_vec, Parameters *params);

void gasal_destroy_gpu_storage_v(gasal_gpu_storage_v *gpu_storage_vec);
