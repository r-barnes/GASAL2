
#include <gasal2/gasal.h>
#include <gasal2/args_parser.h>
#include <gasal2/host_batch.h>
#include <gasal2/res.h>
#include <gasal2/ctors.h>
#include <gasal2/interfaces.h>

#include <cmath>


gasal_gpu_storage_v gasal_init_gpu_storage_v(int n_streams) {
	gasal_gpu_storage_v v;
	v.a = new gasal_gpu_storage_t[n_streams];
	v.n = n_streams;
	return v;
}


void gasal_init_streams(gasal_gpu_storage_v *gpu_storage_vec,  int max_query_len, int max_target_len, int max_n_alns, const Parameters &params) {
	int max_query_len_8 = max_query_len % 8 ? max_query_len + (8 - (max_query_len % 8)) : max_query_len;
	int max_target_len_8 = max_target_len % 8 ? max_target_len + (8 - (max_target_len % 8)) : max_target_len;

	int host_max_query_batch_bytes = max_n_alns * max_query_len_8;
	int gpu_max_query_batch_bytes = max_n_alns * max_query_len_8;
	int host_max_target_batch_bytes =  max_n_alns * max_target_len_8;
	int gpu_max_target_batch_bytes =  max_n_alns * max_target_len_8;
	int host_max_n_alns = max_n_alns;
	int gpu_max_n_alns = max_n_alns;

	for (int i = 0; i < gpu_storage_vec->n; i++) {
		gpu_storage_vec->a[i].extensible_host_unpacked_query_batch = gasal_host_batch_new(host_max_query_batch_bytes, 0);
		gpu_storage_vec->a[i].extensible_host_unpacked_target_batch = gasal_host_batch_new(host_max_target_batch_bytes, 0);

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked_query_batch), gpu_max_query_batch_bytes * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].unpacked_target_batch), gpu_max_target_batch_bytes * sizeof(uint8_t)));


		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_query_op), host_max_n_alns * sizeof(uint8_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_target_op), host_max_n_alns * sizeof(uint8_t), cudaHostAllocDefault));
		uint8_t *no_ops = (uint8_t*) calloc(host_max_n_alns * sizeof(uint8_t), sizeof(uint8_t)); //TODO: Is this right, or is too much space?
		gasal_op_fill(&(gpu_storage_vec->a[i]), no_ops, host_max_n_alns, QUERY);
		gasal_op_fill(&(gpu_storage_vec->a[i]), no_ops, host_max_n_alns, TARGET);
		free(no_ops);

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].query_op), gpu_max_n_alns * sizeof(uint8_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].target_op), gpu_max_n_alns * sizeof(uint8_t)));

		if (params.isPacked)
		{
			gpu_storage_vec->a[i].packed_query_batch = (uint32_t *) gpu_storage_vec->a[i].unpacked_query_batch;
			gpu_storage_vec->a[i].packed_target_batch = (uint32_t *) gpu_storage_vec->a[i].unpacked_target_batch;
		} else {
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed_query_batch), (gpu_max_query_batch_bytes/8) * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed_target_batch), (gpu_max_target_batch_bytes/8) * sizeof(uint32_t)));
		}

		if (params.algo == KSW)
		{
			CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_seed_scores), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].seed_scores), host_max_n_alns * sizeof(uint32_t)));
		} else {
			gpu_storage_vec->a[i].host_seed_scores = NULL;
			gpu_storage_vec->a[i].seed_scores = NULL;
		}

		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_query_batch_lens), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_target_batch_lens), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_query_batch_offsets), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));
		CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_target_batch_offsets), host_max_n_alns * sizeof(uint32_t), cudaHostAllocDefault));

		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].query_batch_lens), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].target_batch_lens), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].query_batch_offsets), gpu_max_n_alns * sizeof(uint32_t)));
		CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].target_batch_offsets), gpu_max_n_alns * sizeof(uint32_t)));

		gpu_storage_vec->a[i].host_res = gasal_res_new_host(host_max_n_alns, params);
		if(params.start_pos == WITH_TB) CHECKCUDAERROR(cudaHostAlloc(&(gpu_storage_vec->a[i].host_res->cigar), gpu_max_query_batch_bytes * sizeof(uint8_t),cudaHostAllocDefault));
		gpu_storage_vec->a[i].device_cpy = gasal_res_new_device_cpy(max_n_alns, params);
		gpu_storage_vec->a[i].device_res = gasal_res_new_device(gpu_storage_vec->a[i].device_cpy);

		if (params.secondBest){
			gpu_storage_vec->a[i].host_res_second = gasal_res_new_host(host_max_n_alns, params);
			gpu_storage_vec->a[i].device_cpy_second = gasal_res_new_device_cpy(host_max_n_alns, params);
			gpu_storage_vec->a[i].device_res_second = gasal_res_new_device(gpu_storage_vec->a[i].device_cpy_second);
		} else {
			gpu_storage_vec->a[i].host_res_second = NULL;
			gpu_storage_vec->a[i].device_cpy_second = NULL;
			gpu_storage_vec->a[i].device_res_second = NULL;
		}

		if (params.start_pos == WITH_TB) {
			gpu_storage_vec->a[i].packed_tb_matrix_size = ((uint32_t)ceil(((double)((uint64_t)max_query_len_8*(uint64_t)max_target_len_8))/32)) * gpu_max_n_alns;
			CHECKCUDAERROR(cudaMalloc(&(gpu_storage_vec->a[i].packed_tb_matrices), gpu_storage_vec->a[i].packed_tb_matrix_size * sizeof(uint4)));
		}

		CHECKCUDAERROR(cudaStreamCreate(&(gpu_storage_vec->a[i].str)));
		gpu_storage_vec->a[i].is_free = 1;
		gpu_storage_vec->a[i].host_max_query_batch_bytes = host_max_query_batch_bytes;
		gpu_storage_vec->a[i].host_max_target_batch_bytes = host_max_target_batch_bytes;
		gpu_storage_vec->a[i].host_max_n_alns = host_max_n_alns;
		gpu_storage_vec->a[i].gpu_max_query_batch_bytes = gpu_max_query_batch_bytes;
		gpu_storage_vec->a[i].gpu_max_target_batch_bytes = gpu_max_target_batch_bytes;
		gpu_storage_vec->a[i].gpu_max_n_alns = gpu_max_n_alns;
		gpu_storage_vec->a[i].current_n_alns = 0;
	}
}

void gasal_destroy_streams(gasal_gpu_storage_v *gpu_storage_vec, const Parameters &params) {
	for (int i = 0; i < gpu_storage_vec->n; i ++) {
		gasal_host_batch_destroy(gpu_storage_vec->a[i].extensible_host_unpacked_query_batch);
		gasal_host_batch_destroy(gpu_storage_vec->a[i].extensible_host_unpacked_target_batch);

		gasal_res_destroy_host(gpu_storage_vec->a[i].host_res);
		gasal_res_destroy_device(gpu_storage_vec->a[i].device_res, gpu_storage_vec->a[i].device_cpy);

		if (params.secondBest)
		{
			gasal_res_destroy_host(gpu_storage_vec->a[i].host_res_second);
			gasal_res_destroy_device(gpu_storage_vec->a[i].device_res_second, gpu_storage_vec->a[i].device_cpy_second);
		}

		if (!(params.algo == KSW))
		{
			if (gpu_storage_vec->a[i].seed_scores)      CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].seed_scores));
			if (gpu_storage_vec->a[i].host_seed_scores) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_seed_scores));
		}

		if (gpu_storage_vec->a[i].query_op)       CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].query_op));
		if (gpu_storage_vec->a[i].target_op)      CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].target_op));
		if (gpu_storage_vec->a[i].host_query_op)  CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_query_op));
		if (gpu_storage_vec->a[i].host_target_op) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_target_op));

		if (gpu_storage_vec->a[i].host_query_batch_offsets)  CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_query_batch_offsets));
		if (gpu_storage_vec->a[i].host_target_batch_offsets) CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_target_batch_offsets));
		if (gpu_storage_vec->a[i].host_query_batch_lens)     CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_query_batch_lens));
		if (gpu_storage_vec->a[i].host_target_batch_lens)    CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_target_batch_lens));
		if (gpu_storage_vec->a[i].host_res->cigar)           CHECKCUDAERROR(cudaFreeHost(gpu_storage_vec->a[i].host_res->cigar));

		if (gpu_storage_vec->a[i].unpacked_query_batch)  CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].unpacked_query_batch));
		if (gpu_storage_vec->a[i].unpacked_target_batch) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].unpacked_target_batch));
		if (!(params.isPacked))
		{
			if (gpu_storage_vec->a[i].packed_query_batch)  CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].packed_query_batch));
			if (gpu_storage_vec->a[i].packed_target_batch) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].packed_target_batch));
		}

		if (gpu_storage_vec->a[i].query_batch_offsets)  CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].query_batch_offsets));
		if (gpu_storage_vec->a[i].target_batch_offsets) CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].target_batch_offsets));
		if (gpu_storage_vec->a[i].query_batch_lens)     CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].query_batch_lens));
		if (gpu_storage_vec->a[i].target_batch_lens)    CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].target_batch_lens));
		if (gpu_storage_vec->a[i].packed_tb_matrices)   CHECKCUDAERROR(cudaFree(gpu_storage_vec->a[i].packed_tb_matrices));

		if (gpu_storage_vec->a[i].str) CHECKCUDAERROR(cudaStreamDestroy(gpu_storage_vec->a[i].str));
	}
}


void gasal_destroy_gpu_storage_v(gasal_gpu_storage_v *gpu_storage_vec) {
	if(gpu_storage_vec->a) free(gpu_storage_vec->a);
}
