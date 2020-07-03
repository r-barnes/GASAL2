#pragma once

#include <cstdint>

__global__ void pack_data(
	const uint32_t *const unpacked,
	uint32_t *const packed,
	const uint64_t len
);

__global__ void	gasal_reversecomplement_kernel(uint32_t *packed_query_batch,uint32_t *packed_target_batch, uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint8_t *query_op, uint8_t *target_op, uint32_t  n_tasks);