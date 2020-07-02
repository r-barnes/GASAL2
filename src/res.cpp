#include <gasal2/gasal.h>
#include <gasal2/args_parser.h>
#include <gasal2/res.h>

#include <iostream>

gasal_res_t *gasal_res_new_host(uint32_t max_n_alns, const Parameters &params)
{
	auto *const res = new gasal_res_t();

	CHECKCUDAERROR(cudaHostAlloc(&(res->aln_score), max_n_alns * sizeof(int32_t),cudaHostAllocDefault));

	if (params.algo == GLOBAL) {
		res->query_batch_start = nullptr;
		res->target_batch_start = nullptr;
		res->query_batch_end = nullptr;
		res->target_batch_end = nullptr;
	} else {
		if (params.start_pos == WITH_START || params.start_pos == WITH_TB) {
			CHECKCUDAERROR(cudaHostAlloc(&(res->query_batch_start),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			CHECKCUDAERROR(cudaHostAlloc(&(res->target_batch_start),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			CHECKCUDAERROR(cudaHostAlloc(&(res->query_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			CHECKCUDAERROR(cudaHostAlloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));

		} else {
			CHECKCUDAERROR(cudaHostAlloc(&(res->query_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			CHECKCUDAERROR(cudaHostAlloc(&(res->target_batch_end),max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
			res->query_batch_start = nullptr;
			res->target_batch_start = nullptr;
		}

	}
	if (params.start_pos == WITH_TB) {
		CHECKCUDAERROR(cudaHostAlloc(&(res->n_cigar_ops), max_n_alns * sizeof(uint32_t),cudaHostAllocDefault));
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

	CHECKCUDAERROR(cudaMalloc(&(res->aln_score), max_n_alns * sizeof(int32_t)));

	if (params.algo == GLOBAL) {
		res->query_batch_start  = nullptr;
		res->target_batch_start = nullptr;
		res->query_batch_end    = nullptr;
		res->target_batch_end   = nullptr;
	} else {
		if (params.start_pos == WITH_START || params.start_pos == WITH_TB) {
			CHECKCUDAERROR(cudaMalloc(&(res->query_batch_start),  max_n_alns * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(res->target_batch_start), max_n_alns * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(res->query_batch_end),    max_n_alns * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(res->target_batch_end),   max_n_alns * sizeof(uint32_t)));
		} else {
			CHECKCUDAERROR(cudaMalloc(&(res->query_batch_end),    max_n_alns * sizeof(uint32_t)));
			CHECKCUDAERROR(cudaMalloc(&(res->target_batch_end),   max_n_alns * sizeof(uint32_t)));

			res->query_batch_start = nullptr;
			res->target_batch_start = nullptr;
		}
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
