#include <gasal2/gasal.h>
#include <gasal2/args_parser.h>
#include <gasal2/res.h>

#include <albp/memory.hpp>

#include <iostream>

gasal_res_t *gasal_res_new_host(uint32_t max_n_alns, const Parameters &params)
{
	auto *const res = new gasal_res_t();

	res->aln_score = albp::PageLockedMalloc<int32_t>(max_n_alns);

	if (params.algo != algo_type::GLOBAL){
		if (params.start_pos == CompStart::WITH_START || params.start_pos == CompStart::WITH_TB) {
			res->query_batch_start  = albp::PageLockedMalloc<int32_t>(max_n_alns);
			res->target_batch_start = albp::PageLockedMalloc<int32_t>(max_n_alns);
		}
		res->query_batch_end  = albp::PageLockedMalloc<int32_t>(max_n_alns);
		res->target_batch_end = albp::PageLockedMalloc<int32_t>(max_n_alns);

	}
	if (params.start_pos == CompStart::WITH_TB) {
		res->n_cigar_ops = albp::PageLockedMalloc<uint32_t>(max_n_alns);
	}

	return res;
}



gasal_res_t *gasal_res_new_device(gasal_res_t *device_cpy)
{
  // create class storage on device and copy top level class
  gasal_res_t *d_c;
  CHECKCUDAERROR(cudaMalloc((void **)&d_c, sizeof(gasal_res_t)));
	//    CHECKCUDAERROR(cudaMemcpy(d_c, res, sizeof(gasal_res_t), cudaMemcpyHostToDevice));

  // copy pointer to allocated device storage to device class
  CHECKCUDAERROR(cudaMemcpy(&(d_c->aln_score), &(device_cpy->aln_score), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->query_batch_start), &(device_cpy->query_batch_start), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->target_batch_start), &(device_cpy->target_batch_start), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->query_batch_end), &(device_cpy->query_batch_end), sizeof(int32_t*), cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpy(&(d_c->target_batch_end), &(device_cpy->target_batch_end), sizeof(int32_t*), cudaMemcpyHostToDevice));

	return d_c;
}



gasal_res_t *gasal_res_new_device_cpy(uint32_t max_n_alns, const Parameters &params)
{
	gasal_res_t *const res = new gasal_res_t();

	res->aln_score = albp::DeviceMalloc<int32_t>(max_n_alns);

	if (params.algo != algo_type::GLOBAL) {
		if (params.start_pos == CompStart::WITH_START || params.start_pos == CompStart::WITH_TB) {
			res->query_batch_start  = albp::DeviceMalloc<int32_t>(max_n_alns);
			res->target_batch_start = albp::DeviceMalloc<int32_t>(max_n_alns);
		}
		res->query_batch_end  = albp::DeviceMalloc<int32_t>(max_n_alns);
		res->target_batch_end = albp::DeviceMalloc<int32_t>(max_n_alns);
	}

	return res;
}

// TODO : make 2 destroys for host and device
void gasal_res_destroy_host(gasal_res_t *res){
	if (!res)
		return;

	if (res->aln_score)          CHECKCUDAERROR(cudaFreeHost(res->aln_score));
	if (res->query_batch_start)  CHECKCUDAERROR(cudaFreeHost(res->query_batch_start));
	if (res->target_batch_start) CHECKCUDAERROR(cudaFreeHost(res->target_batch_start));
	if (res->query_batch_end)    CHECKCUDAERROR(cudaFreeHost(res->query_batch_end));
	if (res->target_batch_end)   CHECKCUDAERROR(cudaFreeHost(res->target_batch_end));
	// if (res->n_cigar_ops)        CHECKCUDAERROR(cudaFreeHost(res->n_cigar_ops)); //TODO

	free(res);
}

void gasal_res_destroy_device(gasal_res_t *device_res, gasal_res_t *device_cpy){
	if (device_cpy == NULL || device_res == NULL)
		return;

	if (device_cpy->aln_score)          CHECKCUDAERROR(cudaFree(device_cpy->aln_score));
	if (device_cpy->query_batch_start)  CHECKCUDAERROR(cudaFree(device_cpy->query_batch_start));
	if (device_cpy->target_batch_start) CHECKCUDAERROR(cudaFree(device_cpy->target_batch_start));
	if (device_cpy->query_batch_end)    CHECKCUDAERROR(cudaFree(device_cpy->query_batch_end));
	if (device_cpy->target_batch_end)   CHECKCUDAERROR(cudaFree(device_cpy->target_batch_end));
	if (device_cpy->cigar)              CHECKCUDAERROR(cudaFree(device_cpy->cigar));

	CHECKCUDAERROR(cudaFree(device_res));

	free(device_cpy);
}
