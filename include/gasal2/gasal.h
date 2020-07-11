#pragma once

#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include <cstdint>
#include <vector>

#ifndef HOST_MALLOC_SAFETY_FACTOR
#define HOST_MALLOC_SAFETY_FACTOR 5
#endif

#define CHECKCUDAERROR(error) \
		do{\
		  const auto err=(error); \
			if (err!=cudaSuccess) { \
				fprintf(stderr, "[GASAL CUDA ERROR:] %s(CUDA error no.=%d). Line no. %d in file %s\n", cudaGetErrorString(err), err,  __LINE__, __FILE__); \
				exit(EXIT_FAILURE);\
			}\
		}while(0)\

namespace thrust {
	template<class T>
	using host_pinned_vector = thrust::host_vector<T, thrust::system::cuda::experimental::pinned_allocator<T>>;
}

enum class CompStart {
	WITHOUT_START,
	WITH_START,
	WITH_TB
};

// Generic enum for ture/false. Using this instead of bool to generalize templates out of Int values for secondBest.
// Can be usd more generically, for example for WITH_/WITHOUT_START.
enum class Bool {
	FALSE,
	TRUE
};

enum class DataSource {
	NONE,
	QUERY,
	TARGET,
	BOTH
};

enum class algo_type {
	UNKNOWN,
	GLOBAL,
	SEMI_GLOBAL,
	LOCAL,
	BANDED,
	KSW
};

enum class operation_on_seq{
	FORWARD_NATURAL,
	REVERSE_NATURAL,
	FORWARD_COMPLEMENT,
	REVERSE_COMPLEMENT,
};



// data structure of linked list to allow extension of memory on host side.
struct host_batch{
	uint8_t *data = nullptr;
	uint32_t page_size;
	uint32_t data_size;
	uint32_t offset;
	int is_locked;
	struct host_batch* next = nullptr;
};
typedef struct host_batch host_batch_t;



// Data structure to hold results. Can be instantiated for host or device memory (see res.cpp)
struct gasal_res_t {
	int32_t *aln_score = nullptr;
	int32_t *query_batch_end = nullptr;
	int32_t *target_batch_end = nullptr;
	int32_t *query_batch_start = nullptr;
	int32_t *target_batch_start = nullptr;
	uint8_t *cigar = nullptr;
	uint32_t *n_cigar_ops = nullptr;
};



//stream data
struct gasal_gpu_storage_t {
	thrust::device_vector<uint8_t> unpacked_query_batch;
	thrust::device_vector<uint8_t> unpacked_target_batch;

	thrust::device_vector<uint32_t> packed_query_batch;
	thrust::device_vector<uint32_t> packed_target_batch;

	thrust::device_vector<uint32_t> query_batch_offsets;
	thrust::device_vector<uint32_t> target_batch_offsets;
	thrust::device_vector<uint32_t> query_batch_lens;
	thrust::device_vector<uint32_t> target_batch_lens;

	thrust::host_pinned_vector<uint32_t> host_seed_scores;
	uint32_t *seed_scores = nullptr;

	host_batch_t *extensible_host_unpacked_query_batch = nullptr;
	host_batch_t *extensible_host_unpacked_target_batch = nullptr;

	thrust::host_pinned_vector<uint8_t> host_query_op;
	thrust::host_pinned_vector<uint8_t> host_target_op;
	thrust::device_vector<uint8_t> query_op;
	thrust::device_vector<uint8_t> target_op;

	std::vector<uint32_t> host_query_batch_offsets;
	std::vector<uint32_t> host_target_batch_offsets;
	std::vector<uint32_t> host_query_batch_lens;
	std::vector<uint32_t> host_target_batch_lens;

	gasal_res_t *host_res = nullptr; // the results that can be read on host - THE STRUCT IS ON HOST SIDE, ITS CONTENT IS ON HOST SIDE.
	gasal_res_t *device_cpy = nullptr; // a struct that contains the pointers to the device side - THE STRUCT IS ON HOST SIDE, but the CONTENT is malloc'd on and points to the DEVICE SIDE
	gasal_res_t *device_res = nullptr; // the results that are written on device - THE STRUCT IS ON DEVICE SIDE, ITS CONTENT POINTS TO THE DEVICE SIDE.

	gasal_res_t *host_res_second = nullptr;
	gasal_res_t *device_res_second = nullptr;
	gasal_res_t *device_cpy_second = nullptr;

	uint32_t gpu_max_query_batch_bytes;
	uint32_t gpu_max_target_batch_bytes;

	uint32_t host_max_query_batch_bytes;
	uint32_t host_max_target_batch_bytes;

	uint32_t gpu_max_n_alns;
	uint32_t host_max_n_alns;
	uint32_t current_n_alns;

	uint64_t packed_tb_matrix_size;
	thrust::device_vector<uint4> packed_tb_matrices;

	cudaStream_t str;
	int is_free;
	int id; //this can be useful in cases where a gasal_gpu_storage only contains PARTS of an alignment (like a seed-extension...), to gather results.
};



//vector of streams
typedef std::vector<gasal_gpu_storage_t> gasal_gpu_storage_v;



//match/mismatch and gap penalties
typedef struct{
	int32_t match;
	int32_t mismatch;
	int32_t gap_open;
	int32_t gap_extend;
} gasal_subst_scores;
