#include <gasal2/gasal.h>
#include <gasal2/args_parser.h>
#include <gasal2/res.h>
#include <gasal2/gasal_align.h>
#include <gasal2/gasal_kernels.h>
#include <gasal2/host_batch.h>

#include <albp/memory.hpp>

#include <stdexcept>


/*  ####################################################################################
    SEMI_GLOBAL Kernels generation - read from the bottom one, all the way up. (the most specialized ones are written before the ones that call them)
    ####################################################################################
*/
#define SEMIGLOBAL_KERNEL_CALL(a,s,h,t,b)                                                \
  case t: {                                                                              \
    gasal_semi_global_kernel<a, s, b, h, t><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>( \
      thrust::raw_pointer_cast(gpu_storage.packed_query_batch.data()),                   \
      thrust::raw_pointer_cast(gpu_storage.packed_target_batch.data()),                  \
      thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()),                     \
      thrust::raw_pointer_cast(gpu_storage.target_batch_lens.data()),                    \
      thrust::raw_pointer_cast(gpu_storage.query_batch_offsets.data()),                  \
      thrust::raw_pointer_cast(gpu_storage.target_batch_offsets.data()),                 \
      gpu_storage.device_res,                                                            \
      gpu_storage.device_res_second,                                                     \
      thrust::raw_pointer_cast(gpu_storage.packed_tb_matrices.data()),                   \
      actual_n_alns                                                                      \
    );                                                                                   \
    break;                                                                               \
  }

#define SWITCH_SEMI_GLOBAL_TAIL(a,s,h,t,b)                                               \
  case h:                                                                                \
  switch(t) {                                                                            \
    SEMIGLOBAL_KERNEL_CALL(a,s,h,DataSource::NONE,b)                                     \
    SEMIGLOBAL_KERNEL_CALL(a,s,h,DataSource::QUERY,b)                                    \
    SEMIGLOBAL_KERNEL_CALL(a,s,h,DataSource::TARGET,b)                                   \
    SEMIGLOBAL_KERNEL_CALL(a,s,h,DataSource::BOTH,b)                                     \
  }                                                                                      \
  break;

#define SWITCH_SEMI_GLOBAL_HEAD(a,s,h,t,b)                                               \
  case s:                                                                                \
  switch(h) {                                                                            \
    SWITCH_SEMI_GLOBAL_TAIL(a,s,DataSource::NONE,t,b)                                    \
    SWITCH_SEMI_GLOBAL_TAIL(a,s,DataSource::QUERY,t,b)                                   \
    SWITCH_SEMI_GLOBAL_TAIL(a,s,DataSource::TARGET,t,b)                                  \
    SWITCH_SEMI_GLOBAL_TAIL(a,s,DataSource::BOTH,t,b)                                    \
  }                                                                                      \
  break;


/*  ####################################################################################
    ALGORITHMS Kernels generation. Allows to have a single line written for all kernels calls. The switch-cases are MACRO-generated.
    ####################################################################################
*/

#define SWITCH_SEMI_GLOBAL(a,s,h,t,b) SWITCH_SEMI_GLOBAL_HEAD(a,s,h,t,b)

#define SWITCH_LOCAL(a,s,h,t,b)                                                      \
  case s: {                                                                          \
	  if(BLOCKDIM!=128) throw std::runtime_error("BLOCKDIM must be 128!");             \
    gasal_local_kernel<s, b><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(            \
      thrust::raw_pointer_cast(gpu_storage.packed_query_batch.data()),               \
      thrust::raw_pointer_cast(gpu_storage.packed_target_batch.data()),              \
      thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()),                 \
      thrust::raw_pointer_cast(gpu_storage.target_batch_lens.data()),                \
      thrust::raw_pointer_cast(gpu_storage.query_batch_offsets.data()),              \
      thrust::raw_pointer_cast(gpu_storage.target_batch_offsets.data()),             \
      gpu_storage.device_res,                                                        \
      gpu_storage.device_res_second,                                                 \
      thrust::raw_pointer_cast(gpu_storage.packed_tb_matrices.data()),               \
      actual_n_alns                                                                  \
    );                                                                               \
    if(s == CompStart::WITH_TB) {                                                    \
      const auto aln_kernel_err = cudaGetLastError();                                \
      if ( cudaSuccess != aln_kernel_err )                                           \
      {                                                                              \
        fprintf(stderr,                                                              \
          "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %sCHEESE", \
          cudaGetErrorString(aln_kernel_err), aln_kernel_err,  __LINE__, __FILE__    \
        );                                                                           \
        exit(EXIT_FAILURE);                                                          \
      }                                                                              \
      gasal_get_tb<algo_type::LOCAL><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(    \
        thrust::raw_pointer_cast(gpu_storage.unpacked_query_batch.data()),           \
        thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()),               \
        thrust::raw_pointer_cast(gpu_storage.target_batch_lens.data()),              \
        thrust::raw_pointer_cast(gpu_storage.query_batch_offsets.data()),            \
        thrust::raw_pointer_cast(gpu_storage.packed_tb_matrices.data()),             \
        gpu_storage.device_res,                                                      \
        gpu_storage.current_n_alns                                                   \
      );                                                                             \
    }                                                                                \
    break;                                                                           \
  }

#define SWITCH_GLOBAL(a,s,h,t,b)                                                     \
  case s: {                                                                          \
    gasal_global_kernel<s><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(              \
      thrust::raw_pointer_cast(gpu_storage.packed_query_batch.data()),               \
      thrust::raw_pointer_cast(gpu_storage.packed_target_batch.data()),              \
      thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()),                 \
      thrust::raw_pointer_cast(gpu_storage.target_batch_lens.data()),                \
      thrust::raw_pointer_cast(gpu_storage.query_batch_offsets.data()),              \
      thrust::raw_pointer_cast(gpu_storage.target_batch_offsets.data()),             \
      gpu_storage.device_res,                                                        \
      thrust::raw_pointer_cast(gpu_storage.packed_tb_matrices.data()),               \
      actual_n_alns                                                                  \
    );                                                                               \
    if(s == CompStart::WITH_TB) {                                                    \
      cudaError_t aln_kernel_err = cudaGetLastError();                               \
      if ( cudaSuccess != aln_kernel_err )                                           \
      {                                                                              \
        fprintf(                                                                     \
          stderr,                                                                    \
          "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %sCHEESE", \
          cudaGetErrorString(aln_kernel_err), aln_kernel_err, __LINE__, __FILE__     \
        );                                                                           \
        exit(EXIT_FAILURE);                                                          \
      }                                                                              \
      gasal_get_tb<algo_type::GLOBAL><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(   \
        thrust::raw_pointer_cast(gpu_storage.unpacked_query_batch.data()),           \
        thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()),               \
        thrust::raw_pointer_cast(gpu_storage.target_batch_lens.data()),              \
        thrust::raw_pointer_cast(gpu_storage.query_batch_offsets.data()),            \
        thrust::raw_pointer_cast(gpu_storage.packed_tb_matrices.data()),             \
        gpu_storage.device_res,                                                      \
        gpu_storage.current_n_alns                                                   \
      );                                                                             \
    }                                                                                \
    break;                                                                           \
  }


#define SWITCH_KSW(a,s,h,t,b)                                                  \
  case s:                                                                      \
    gasal_ksw_kernel<b><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(           \
      thrust::raw_pointer_cast(gpu_storage.packed_query_batch.data()),         \
      thrust::raw_pointer_cast(gpu_storage.packed_target_batch.data()),        \
      thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()),           \
      thrust::raw_pointer_cast(gpu_storage.target_batch_lens.data()),          \
      thrust::raw_pointer_cast(gpu_storage.query_batch_offsets.data()),        \
      thrust::raw_pointer_cast(gpu_storage.target_batch_offsets.data()),       \
      gpu_storage.seed_scores,                                                 \
      gpu_storage.device_res,                                                  \
      gpu_storage.device_res_second, actual_n_alns                             \
    );                                                                         \
    break;

#define SWITCH_BANDED(a,s,h,t,b)                                               \
  case s:                                                                      \
    gasal_banded_tiled_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(     \
      thrust::raw_pointer_cast(gpu_storage.packed_query_batch.data()),         \
      thrust::raw_pointer_cast(gpu_storage.packed_target_batch.data()),        \
      thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()),           \
      thrust::raw_pointer_cast(gpu_storage.target_batch_lens.data()),          \
      thrust::raw_pointer_cast(gpu_storage.query_batch_offsets.data()),        \
      thrust::raw_pointer_cast(gpu_storage.target_batch_offsets.data()),       \
      gpu_storage.device_res,                                                  \
      actual_n_alns,                                                           \
      k_band>>3                                                                \
    );                                                                         \
    break;

/*  ####################################################################################
    RUN PARAMETERS calls : general call (bottom, should be used), and first level TRUE/FALSE calculation for second best,
    then 2nd level WITH / WITHOUT_START switch call (top)
    ####################################################################################
*/

#define SWITCH_START(aname,a,s,h,t,b)                     \
    case b:                                               \
    switch(s){                                            \
        SWITCH_## aname(a,CompStart::WITH_START,h,t,b)    \
        SWITCH_## aname(a,CompStart::WITHOUT_START,h,t,b) \
        SWITCH_## aname(a,CompStart::WITH_TB,h,t,b)       \
    }                                                     \
    break;

#define SWITCH_SECONDBEST(aname,a,s,h,t,b)                \
    switch(b) {                                           \
        SWITCH_START(aname,a,s,h,t,Bool::TRUE)            \
        SWITCH_START(aname,a,s,h,t,Bool::FALSE)           \
    }

#define KERNEL_SWITCH(aname,a,s,h,t,b)                    \
    case a:                                               \
        SWITCH_SECONDBEST(aname,a,s,h,t,b)                \
    break;






inline void gasal_kernel_launcher(
	int32_t N_BLOCKS,
	int32_t BLOCKDIM,
	algo_type algo,
	CompStart start,
	gasal_gpu_storage_t &gpu_storage,
	int32_t actual_n_alns,
	int32_t k_band,
	DataSource semiglobal_skipping_head,
	DataSource semiglobal_skipping_tail,
	Bool secondBest
){
  switch(algo){
    KERNEL_SWITCH(LOCAL,       algo_type::LOCAL,       start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
    KERNEL_SWITCH(SEMI_GLOBAL, algo_type::SEMI_GLOBAL, start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);		// MACRO that expands all 32 semi-global kernels
    KERNEL_SWITCH(GLOBAL,      algo_type::GLOBAL,      start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
    KERNEL_SWITCH(KSW,         algo_type::KSW,         start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
    KERNEL_SWITCH(BANDED,      algo_type::BANDED,      start, semiglobal_skipping_head, semiglobal_skipping_tail, secondBest);
    default:
    break;
  }
}



void check_gasal_aln_async_inputs(
	const gasal_gpu_storage_t &gpu_storage,
	const uint32_t actual_query_batch_bytes,
	const uint32_t actual_target_batch_bytes,
	const uint32_t actual_n_alns,
	const Parameters &params
){
	bool failed = false;

	if (actual_n_alns <= 0) {
		fprintf(stderr, "[GASAL ERROR:] actual_n_alns <= 0\n");
		failed = true;
	}

	if (actual_query_batch_bytes <= 0) {
		fprintf(stderr, "[GASAL ERROR:] actual_query_batch_bytes <= 0\n");
		failed = true;
	}

	if (actual_target_batch_bytes <= 0) {
		fprintf(stderr, "[GASAL ERROR:] actual_target_batch_bytes <= 0\n");
		failed = true;
	}

	if (actual_query_batch_bytes % 8) {
		fprintf(stderr, "[GASAL ERROR:] actual_query_batch_bytes=%d is not a multiple of 8\n", actual_query_batch_bytes);
		failed = true;
	}

	if (actual_target_batch_bytes % 8) {
		fprintf(stderr, "[GASAL ERROR:] actual_target_batch_bytes=%d is not a multiple of 8\n", actual_target_batch_bytes);
		failed = true;
	}

	if (actual_query_batch_bytes > gpu_storage.host_max_query_batch_bytes) {
		fprintf(stderr, "[GASAL ERROR:] actual_query_batch_bytes(%d) > host_max_query_batch_bytes(%d)\n", actual_query_batch_bytes, gpu_storage.host_max_query_batch_bytes);
		failed = true;
	}

	if (actual_target_batch_bytes > gpu_storage.host_max_target_batch_bytes) {
		fprintf(stderr, "[GASAL ERROR:] actual_target_batch_bytes(%d) > host_max_target_batch_bytes(%d)\n", actual_target_batch_bytes, gpu_storage.host_max_target_batch_bytes);
		failed = true;
	}

	if (actual_n_alns > gpu_storage.host_max_n_alns) {
		fprintf(stderr, "[GASAL ERROR:] actual_n_alns(%d) > host_max_n_alns(%d)\n", actual_n_alns, gpu_storage.host_max_n_alns);
		failed = true;
	}

	if(failed){
		exit(EXIT_FAILURE);
	}
}


//GASAL2 asynchronous (a.k.a non-blocking) alignment function
void gasal_aln_async(
	gasal_gpu_storage_t &gpu_storage,
	const uint32_t actual_query_batch_bytes,
	const uint32_t actual_target_batch_bytes,
	const uint32_t actual_n_alns,
	const Parameters &params
){
	thrust::cuda::par.on(gpu_storage.str);

	check_gasal_aln_async_inputs(gpu_storage, actual_query_batch_bytes, actual_target_batch_bytes, actual_n_alns, params);

	//--------------if pre-allocated memory is less, allocate more--------------------------
	if (gpu_storage.gpu_max_query_batch_bytes < actual_query_batch_bytes) {

		int i = 2;
		while ( (gpu_storage.gpu_max_query_batch_bytes * i) < actual_query_batch_bytes) i++;

		fprintf(stderr, "[GASAL WARNING:] actual_query_batch_bytes(%d) > Allocated GPU memory (gpu_max_query_batch_bytes=%d). Therefore, allocating %d bytes on GPU (gpu_max_query_batch_bytes=%d). Performance may be lost if this is repeated many times.\n", actual_query_batch_bytes, gpu_storage.gpu_max_query_batch_bytes, gpu_storage.gpu_max_query_batch_bytes*i, gpu_storage.gpu_max_query_batch_bytes*i);

		gpu_storage.gpu_max_query_batch_bytes = gpu_storage.gpu_max_query_batch_bytes * i;

		//Destroy existing vectors and make new, longer ones
		gpu_storage.unpacked_query_batch.resize(gpu_storage.gpu_max_query_batch_bytes);
		gpu_storage.packed_query_batch.resize(gpu_storage.gpu_max_query_batch_bytes/8);

		if (params.start_pos==CompStart::WITH_TB){
			fprintf(stderr, "[GASAL WARNING:] actual_query_batch_bytes(%d) > Allocated HOST memory for CIGAR (gpu_max_query_batch_bytes=%d). Therefore, allocating %d bytes on the host (gpu_max_query_batch_bytes=%d). Performance may be lost if this is repeated many times.\n", actual_query_batch_bytes, gpu_storage.gpu_max_query_batch_bytes, gpu_storage.gpu_max_query_batch_bytes*i, gpu_storage.gpu_max_query_batch_bytes*i);
			if (gpu_storage.host_res->cigar)CHECKCUDAERROR(cudaFreeHost(gpu_storage.host_res->cigar));
			gpu_storage.host_res->cigar = albp::PageLockedMalloc<uint8_t>(gpu_storage.gpu_max_query_batch_bytes);
		}
	}

	if (gpu_storage.gpu_max_target_batch_bytes < actual_target_batch_bytes) {
		int i = 2;
		while ( (gpu_storage.gpu_max_target_batch_bytes * i) < actual_target_batch_bytes) i++;

		fprintf(stderr, "[GASAL WARNING:] actual_target_batch_bytes(%d) > Allocated GPU memory (gpu_max_target_batch_bytes=%d). Therefore, allocating %d bytes on GPU (gpu_max_target_batch_bytes=%d). Performance may be lost if this is repeated many times.\n", actual_target_batch_bytes, gpu_storage.gpu_max_target_batch_bytes, gpu_storage.gpu_max_target_batch_bytes*i, gpu_storage.gpu_max_target_batch_bytes*i);

		gpu_storage.gpu_max_target_batch_bytes = gpu_storage.gpu_max_target_batch_bytes * i;

		gpu_storage.unpacked_target_batch.resize(gpu_storage.gpu_max_target_batch_bytes);
		gpu_storage.packed_target_batch.resize(gpu_storage.gpu_max_target_batch_bytes/8);
	}

	if (gpu_storage.gpu_max_n_alns < actual_n_alns) {
		int i = 2;
		while ( (gpu_storage.gpu_max_n_alns * i) < actual_n_alns) i++;

		fprintf(stderr, "[GASAL WARNING:] actual_n_alns(%d) > gpu_max_n_alns(%d). Therefore, allocating memory for %d alignments on  GPU (gpu_max_n_alns=%d). Performance may be lost if this is repeated many times.\n", actual_n_alns, gpu_storage.gpu_max_n_alns, gpu_storage.gpu_max_n_alns*i, gpu_storage.gpu_max_n_alns*i);

		gpu_storage.gpu_max_n_alns = gpu_storage.gpu_max_n_alns * i;

		gpu_storage.query_batch_offsets.clear();  gpu_storage.query_batch_offsets.shrink_to_fit();
		gpu_storage.target_batch_offsets.clear(); gpu_storage.target_batch_offsets.shrink_to_fit();
		gpu_storage.query_batch_lens.clear();     gpu_storage.query_batch_lens.shrink_to_fit();
		gpu_storage.target_batch_lens.clear();    gpu_storage.target_batch_lens.shrink_to_fit();

		if (gpu_storage.seed_scores) CHECKCUDAERROR(cudaFree(gpu_storage.seed_scores));

		gpu_storage.query_batch_lens.resize(gpu_storage.gpu_max_n_alns);
		gpu_storage.target_batch_lens.resize(gpu_storage.gpu_max_n_alns);
		gpu_storage.query_batch_offsets.resize(gpu_storage.gpu_max_n_alns);
		gpu_storage.target_batch_offsets.resize(gpu_storage.gpu_max_n_alns);

		gpu_storage.seed_scores = albp::DeviceMalloc<uint32_t>(gpu_storage.gpu_max_n_alns);

		gasal_res_destroy_device(gpu_storage.device_res, gpu_storage.device_cpy);
		gpu_storage.device_cpy = gasal_res_new_device_cpy(gpu_storage.gpu_max_n_alns, params);
		gpu_storage.device_res = gasal_res_new_device(gpu_storage.device_cpy);

		if (params.secondBest==Bool::TRUE)
		{
			gasal_res_destroy_device(gpu_storage.device_res_second, gpu_storage.device_cpy_second);
			gpu_storage.device_cpy_second = gasal_res_new_device_cpy(gpu_storage.gpu_max_n_alns, params);
			gpu_storage.device_res_second = gasal_res_new_device(gpu_storage.device_cpy_second);
		}

	}
	//------------------------------------------

	//------------------------launch copying of sequence batches from CPU to GPU---------------------------

	// here you can track the evolution of your data structure processing with the printer: gasal_host_batch_printall(current);

	const host_batch_t *current = gpu_storage.extensible_host_unpacked_query_batch;
	for(;current;current = current->next){
		gpu_storage.unpacked_query_batch.resize(current->data_size);
		//gasal_host_batch_printall(current);
		CHECKCUDAERROR(cudaMemcpyAsync(
			thrust::raw_pointer_cast(gpu_storage.unpacked_query_batch.data() + (current->offset)),
										current->data,
										current->data_size,
										cudaMemcpyHostToDevice,
										gpu_storage.str ) );
	}

	current = gpu_storage.extensible_host_unpacked_target_batch;
	for(;current;current = current->next){
		CHECKCUDAERROR(cudaMemcpyAsync(
			thrust::raw_pointer_cast(gpu_storage.unpacked_target_batch.data() + (current->offset)),
										current->data,
										current->data_size,
										cudaMemcpyHostToDevice,
										gpu_storage.str ) );
	}

	//-----------------------------------------------------------------------------------------------------------
	// TODO: Adjust the block size depending on the kernel execution.

	const uint32_t BLOCKDIM = 128;
	const uint32_t N_BLOCKS = (actual_n_alns + BLOCKDIM - 1) / BLOCKDIM;

	const int query_batch_tasks_per_thread  = (int)ceil((double)actual_query_batch_bytes/(8*BLOCKDIM*N_BLOCKS));
	const int target_batch_tasks_per_thread = (int)ceil((double)actual_target_batch_bytes/(8*BLOCKDIM*N_BLOCKS));

	//Launch packing kernel
	if(!params.isPacked){
		pack_data<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(
			(const uint32_t*)thrust::raw_pointer_cast(gpu_storage.unpacked_query_batch.data()),
			thrust::raw_pointer_cast(gpu_storage.packed_query_batch.data()),
			actual_query_batch_bytes/4
		);
		const auto pack_query_err = cudaGetLastError();
		if(pack_query_err!=cudaSuccess){
			fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(pack_query_err), pack_query_err,  __LINE__, __FILE__);
		  exit(EXIT_FAILURE);
		}

		pack_data<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(
			(const uint32_t*)thrust::raw_pointer_cast(gpu_storage.unpacked_target_batch.data()),
			thrust::raw_pointer_cast(gpu_storage.packed_target_batch.data()),
			actual_target_batch_bytes/4
		);
		const auto pack_target_err = cudaGetLastError();
		if(pack_target_err!=cudaSuccess){
			fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(pack_target_err), pack_target_err,  __LINE__, __FILE__);
		  exit(EXIT_FAILURE);
		}
	}


	// We could reverse-complement before packing, but we would get 2x more read-writes to memory.

  //----------------------launch copying of sequence offsets and lengths from CPU to GPU--------------------------------------
  CHECKCUDAERROR(cudaMemcpyAsync(thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()),     thrust::raw_pointer_cast(gpu_storage.host_query_batch_lens.data()),     actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage.str));
  CHECKCUDAERROR(cudaMemcpyAsync(thrust::raw_pointer_cast(gpu_storage.target_batch_lens.data()),    thrust::raw_pointer_cast(gpu_storage.host_target_batch_lens.data()),    actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage.str));
  CHECKCUDAERROR(cudaMemcpyAsync(thrust::raw_pointer_cast(gpu_storage.query_batch_offsets.data()),  thrust::raw_pointer_cast(gpu_storage.host_query_batch_offsets.data()),  actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage.str));
	CHECKCUDAERROR(cudaMemcpyAsync(thrust::raw_pointer_cast(gpu_storage.target_batch_offsets.data()), thrust::raw_pointer_cast(gpu_storage.host_target_batch_offsets.data()), actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice,  gpu_storage.str));

	// if needed copy seed scores
	if (params.algo == algo_type::KSW)
	{
		if (gpu_storage.seed_scores == NULL)
		{
			fprintf(stderr, "seed_scores == NULL\n");

		}
		if (gpu_storage.host_seed_scores.empty())
		{
			fprintf(stderr, "host_seed_scores == NULL\n");
		}
		if (gpu_storage.seed_scores == NULL || gpu_storage.host_seed_scores.empty())
			exit(EXIT_FAILURE);

		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.seed_scores, thrust::raw_pointer_cast(gpu_storage.host_seed_scores.data()), actual_n_alns * sizeof(uint32_t), cudaMemcpyHostToDevice, gpu_storage.str));
	}
    //--------------------------------------------------------------------------------------------------------------------------

	//----------------------launch copying of sequence operations (reverse/complement) from CPU to GPU--------------------------
	if (params.isReverseComplement)
	{
		//TODO: Put the copy on the same stream somehow?
		gpu_storage.query_op = gpu_storage.host_query_op;
		gpu_storage.target_op = gpu_storage.host_target_op;
		//--------------------------------------launch reverse-complement kernel------------------------------------------------------
		new_reversecomplement_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(
			thrust::raw_pointer_cast(gpu_storage.packed_query_batch.data()),
			thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()),
			thrust::raw_pointer_cast(gpu_storage.query_batch_offsets.data()),
			thrust::raw_pointer_cast(gpu_storage.query_op.data()),
			actual_n_alns
		);
		const cudaError_t reversecomplement_kernel_err1 = cudaGetLastError();
		if ( cudaSuccess != reversecomplement_kernel_err1 ){
			 fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(reversecomplement_kernel_err1), reversecomplement_kernel_err1,  __LINE__, __FILE__);
			 exit(EXIT_FAILURE);
		}

		new_reversecomplement_kernel<<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(
			thrust::raw_pointer_cast(gpu_storage.packed_target_batch.data()),
			thrust::raw_pointer_cast(gpu_storage.target_batch_lens.data()),
			thrust::raw_pointer_cast(gpu_storage.target_batch_offsets.data()),
			thrust::raw_pointer_cast(gpu_storage.target_op.data()),
			actual_n_alns
		);
		const cudaError_t reversecomplement_kernel_err2 = cudaGetLastError();
		if ( cudaSuccess != reversecomplement_kernel_err2 ){
			 fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(reversecomplement_kernel_err2), reversecomplement_kernel_err2,  __LINE__, __FILE__);
			 exit(EXIT_FAILURE);
		}
	}

    //--------------------------------------launch alignment kernels--------------------------------------------------------------
	gasal_kernel_launcher(N_BLOCKS, BLOCKDIM, params.algo, params.start_pos, gpu_storage, actual_n_alns, params.k_band, params.semiglobal_skipping_head, params.semiglobal_skipping_tail, params.secondBest);

	//if (params.start_pos == WITH_TB) {

		// The output of the kernel: gpu_storage.unpacked_query_batch = cigar, gpu_storage.query_batch_lens = n_cigar_ops
		//gasal_get_tb<Int2Type<params.algo>><<<N_BLOCKS, BLOCKDIM, 0, gpu_storage.str>>>(gpu_storage->unpacked_query_batch, gpu_storage->query_batch_lens, gpu_storage->target_batch_lens, gpu_storage->query_batch_offsets, gpu_storage->packed_tb_matrices, gpu_storage->device_res, gpu_storage->current_n_alns);
	//}

  //-----------------------------------------------------------------------------------------------------------------------
	cudaError_t aln_kernel_err = cudaGetLastError();
	if ( cudaSuccess != aln_kernel_err )
	{
		fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(aln_kernel_err), aln_kernel_err,  __LINE__, __FILE__);
		exit(EXIT_FAILURE);
	}

  //------------------------0launch the copying of alignment results from GPU to CPU--------------------------------------
  if (gpu_storage.host_res->aln_score && gpu_storage.device_cpy->aln_score)
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res->aln_score, gpu_storage.device_cpy->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));

	if (gpu_storage.host_res->query_batch_start && gpu_storage.device_cpy->query_batch_start)
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res->query_batch_start, gpu_storage.device_cpy->query_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));

	if (gpu_storage.host_res->target_batch_start && gpu_storage.device_cpy->target_batch_start)
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res->target_batch_start, gpu_storage.device_cpy->target_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));

	if (gpu_storage.host_res->query_batch_end && gpu_storage.device_cpy->query_batch_end)
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res->query_batch_end, gpu_storage.device_cpy->query_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));

	if (gpu_storage.host_res->target_batch_end && gpu_storage.device_cpy->target_batch_end)
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res->target_batch_end, gpu_storage.device_cpy->target_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));

	if (params.start_pos == CompStart::WITH_TB) {
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res->cigar, thrust::raw_pointer_cast(gpu_storage.unpacked_query_batch.data()), actual_query_batch_bytes * sizeof(uint8_t), cudaMemcpyDeviceToHost, gpu_storage.str));
		CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res->n_cigar_ops, thrust::raw_pointer_cast(gpu_storage.query_batch_lens.data()), actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));
	}
	//-----------------------------------------------------------------------------------------------------------------------


	// not really needed to filter with params.secondBest, since all the pointers will be null and non-initialized.
	if (params.secondBest==Bool::TRUE)
	{
		if (gpu_storage.host_res_second->aln_score && gpu_storage.device_cpy_second->aln_score)
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res_second->aln_score, gpu_storage.device_cpy_second->aln_score, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));

		if (gpu_storage.host_res_second->query_batch_start && gpu_storage.device_cpy_second->query_batch_start)
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res_second->query_batch_start, gpu_storage.device_cpy_second->query_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));

		if (gpu_storage.host_res_second->target_batch_start && gpu_storage.device_cpy_second->target_batch_start)
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res_second->target_batch_start, gpu_storage.device_cpy_second->target_batch_start, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));

		if (gpu_storage.host_res_second->query_batch_end && gpu_storage.device_cpy_second->query_batch_end)
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res_second->query_batch_end, gpu_storage.device_cpy_second->query_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));

		if (gpu_storage.host_res_second->target_batch_end && gpu_storage.device_cpy_second->target_batch_end)
			CHECKCUDAERROR(cudaMemcpyAsync(gpu_storage.host_res_second->target_batch_end, gpu_storage.device_cpy_second->target_batch_end, actual_n_alns * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_storage.str));
	}

    gpu_storage.is_free = 0; //set the availability of current stream to false
}


AlignmentStatus gasal_is_aln_async_done(gasal_gpu_storage_t &gpu_storage){
  if(gpu_storage.is_free == 1)
    return AlignmentStatus::StreamFree;

	//Check to see if the stream is finished
  const cudaError_t err = cudaStreamQuery(gpu_storage.str);
  if(err==cudaErrorNotReady){
    return AlignmentStatus::NotReady;
  } else if(err!=cudaSuccess) {
    fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(err), err,  __LINE__, __FILE__);
    exit(EXIT_FAILURE);
  }

  gasal_host_batch_reset(gpu_storage);
  gpu_storage.is_free = 1;
  gpu_storage.current_n_alns = 0;
  return AlignmentStatus::Finished;
}



void gasal_copy_subst_scores(const gasal_subst_scores &subst){
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapO,          &(subst.gap_open),   sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapExtend,     &(subst.gap_extend), sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	const auto gapoe = subst.gap_open + subst.gap_extend;
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaGapOE,         &(gapoe),            sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMatchScore,    &(subst.match),      sizeof(int32_t), 0, cudaMemcpyHostToDevice));
	CHECKCUDAERROR(cudaMemcpyToSymbol(_cudaMismatchScore, &(subst.mismatch),   sizeof(int32_t), 0, cudaMemcpyHostToDevice));
}
