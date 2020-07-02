#pragma once

#include <cstdint>

__global__ void gasal_pack_kernel(
	uint32_t* unpacked_query_batch,
	uint32_t* unpacked_target_batch,
	uint32_t *packed_query_batch,
	uint32_t* packed_target_batch,
	int query_batch_tasks_per_thread,
	int target_batch_tasks_per_thread,
	uint32_t total_query_batch_regs,
	uint32_t total_target_batch_regs
);

__global__ void	gasal_reversecomplement_kernel(uint32_t *packed_query_batch,uint32_t *packed_target_batch, uint32_t *query_batch_lens, uint32_t *target_batch_lens, uint32_t *query_batch_offsets, uint32_t *target_batch_offsets, uint8_t *query_op, uint8_t *target_op, uint32_t  n_tasks);