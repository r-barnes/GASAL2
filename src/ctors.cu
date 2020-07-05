
#include <gasal2/gasal.h>
#include <gasal2/args_parser.h>
#include <gasal2/host_batch.h>
#include <gasal2/res.h>
#include <gasal2/ctors.h>
#include <gasal2/interfaces.h>

#include <cmath>



void gasal_init_streams(gasal_gpu_storage_v &gpu_storage_vec,  int max_query_len, int max_target_len, int max_n_alns, const Parameters &params) {
	int max_query_len_8 = max_query_len % 8 ? max_query_len + (8 - (max_query_len % 8)) : max_query_len;
	int max_target_len_8 = max_target_len % 8 ? max_target_len + (8 - (max_target_len % 8)) : max_target_len;

	int host_max_query_batch_bytes = max_n_alns * max_query_len_8;
	int gpu_max_query_batch_bytes = max_n_alns * max_query_len_8;
	int host_max_target_batch_bytes =  max_n_alns * max_target_len_8;
	int gpu_max_target_batch_bytes =  max_n_alns * max_target_len_8;
	int host_max_n_alns = max_n_alns;
	int gpu_max_n_alns = max_n_alns;

	for (auto &this_gpu_storage: gpu_storage_vec) {
		this_gpu_storage.extensible_host_unpacked_query_batch = gasal_host_batch_new(host_max_query_batch_bytes, 0);
		this_gpu_storage.extensible_host_unpacked_target_batch = gasal_host_batch_new(host_max_target_batch_bytes, 0);

		this_gpu_storage.unpacked_target_batch.resize(gpu_max_target_batch_bytes);

		this_gpu_storage.host_query_op.resize(host_max_n_alns);
		this_gpu_storage.host_target_op.resize(host_max_n_alns);
		uint8_t *no_ops = (uint8_t*) calloc(host_max_n_alns * sizeof(uint8_t), sizeof(uint8_t)); //TODO: Is this right, or is too much space?
		gasal_op_fill(this_gpu_storage, no_ops, host_max_n_alns, DataSource::QUERY);
		gasal_op_fill(this_gpu_storage, no_ops, host_max_n_alns, DataSource::TARGET);
		free(no_ops);

		this_gpu_storage.query_op.resize(gpu_max_n_alns);
		this_gpu_storage.target_op.resize(gpu_max_n_alns);

		if (params.isPacked)
		{
			this_gpu_storage.packed_query_batch = this_gpu_storage.unpacked_query_batch; //TODO: Avoid copy somehow?
			this_gpu_storage.packed_target_batch = this_gpu_storage.unpacked_target_batch; //TODO: Avoid copy somehow?
		} else {
			this_gpu_storage.packed_query_batch.resize(gpu_max_query_batch_bytes/8);
			this_gpu_storage.packed_target_batch.resize(gpu_max_target_batch_bytes/8);
		}

		if (params.algo == algo_type::KSW)
		{
			this_gpu_storage.host_seed_scores.resize(host_max_n_alns);
			CHECKCUDAERROR(cudaMalloc(&(this_gpu_storage.seed_scores), host_max_n_alns * sizeof(uint32_t)));
		} else {
			this_gpu_storage.seed_scores = NULL;
		}

		this_gpu_storage.host_query_batch_lens.resize(host_max_n_alns);
		this_gpu_storage.host_target_batch_lens.resize(host_max_n_alns);
		this_gpu_storage.host_query_batch_offsets.resize(host_max_n_alns);
		this_gpu_storage.host_target_batch_offsets.resize(host_max_n_alns);

		this_gpu_storage.query_batch_lens.resize(gpu_max_n_alns);
		this_gpu_storage.target_batch_lens.resize(gpu_max_n_alns);
		this_gpu_storage.query_batch_offsets.resize(gpu_max_n_alns);
		this_gpu_storage.target_batch_offsets.resize(gpu_max_n_alns);

		this_gpu_storage.host_res = gasal_res_new_host(host_max_n_alns, params);
		if(params.start_pos == CompStart::WITH_TB) CHECKCUDAERROR(cudaHostAlloc(&(this_gpu_storage.host_res->cigar), gpu_max_query_batch_bytes * sizeof(uint8_t),cudaHostAllocDefault));
		this_gpu_storage.device_cpy = gasal_res_new_device_cpy(max_n_alns, params);
		this_gpu_storage.device_res = gasal_res_new_device(this_gpu_storage.device_cpy);

		if (params.secondBest==Bool::TRUE){
			this_gpu_storage.host_res_second = gasal_res_new_host(host_max_n_alns, params);
			this_gpu_storage.device_cpy_second = gasal_res_new_device_cpy(host_max_n_alns, params);
			this_gpu_storage.device_res_second = gasal_res_new_device(this_gpu_storage.device_cpy_second);
		} else {
			this_gpu_storage.host_res_second = NULL;
			this_gpu_storage.device_cpy_second = NULL;
			this_gpu_storage.device_res_second = NULL;
		}

		if (params.start_pos == CompStart::WITH_TB) {
			const auto packed_tb_matrix_size = ((uint32_t)ceil(((double)((uint64_t)max_query_len_8*(uint64_t)max_target_len_8))/32)) * gpu_max_n_alns;
			this_gpu_storage.packed_tb_matrices.resize(packed_tb_matrix_size);
		}

		CHECKCUDAERROR(cudaStreamCreate(&(this_gpu_storage.str)));
		this_gpu_storage.is_free = 1;
		this_gpu_storage.host_max_query_batch_bytes = host_max_query_batch_bytes;
		this_gpu_storage.host_max_target_batch_bytes = host_max_target_batch_bytes;
		this_gpu_storage.host_max_n_alns = host_max_n_alns;
		this_gpu_storage.gpu_max_query_batch_bytes = gpu_max_query_batch_bytes;
		this_gpu_storage.gpu_max_target_batch_bytes = gpu_max_target_batch_bytes;
		this_gpu_storage.gpu_max_n_alns = gpu_max_n_alns;
		this_gpu_storage.current_n_alns = 0;
	}
}

void gasal_destroy_streams(gasal_gpu_storage_v &gpu_storage_vec, const Parameters &params) {
	for (auto &this_gpu_storage: gpu_storage_vec) {
		gasal_host_batch_destroy(this_gpu_storage.extensible_host_unpacked_query_batch);
		gasal_host_batch_destroy(this_gpu_storage.extensible_host_unpacked_target_batch);

		gasal_res_destroy_host(this_gpu_storage.host_res);
		gasal_res_destroy_device(this_gpu_storage.device_res, this_gpu_storage.device_cpy);

		if(params.secondBest==Bool::TRUE)
		{
			gasal_res_destroy_host(this_gpu_storage.host_res_second);
			gasal_res_destroy_device(this_gpu_storage.device_res_second, this_gpu_storage.device_cpy_second);
		}

		if (!(params.algo == algo_type::KSW))
		{
			if (this_gpu_storage.seed_scores)       CHECKCUDAERROR(cudaFree(this_gpu_storage.seed_scores));
		}

		if (this_gpu_storage.host_res->cigar)     CHECKCUDAERROR(cudaFreeHost(this_gpu_storage.host_res->cigar));
		if (this_gpu_storage.str)                 CHECKCUDAERROR(cudaStreamDestroy(this_gpu_storage.str));
	}
}
