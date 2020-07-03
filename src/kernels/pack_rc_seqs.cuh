#pragma once

#include <cstdint>

__global__ void pack_data(
	const uint32_t *const unpacked,
	uint32_t *const packed,
	const uint64_t len
);

__host__ __device__ uint32_t packed_complement1(uint32_t packed_bases);

__global__ void	gasal_reversecomplement_kernel(
	uint32_t       *const packed_query_batch,
	uint32_t       *const packed_target_batch,
	const uint32_t *const query_batch_lens,
	const uint32_t *const target_batch_lens,
	const uint32_t *const query_batch_offsets,
	const uint32_t *const target_batch_offsets,
	const uint8_t  *const query_op,
	const uint8_t  *const target_op,
	const uint32_t        n_tasks
);