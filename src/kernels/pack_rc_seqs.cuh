#pragma once

#include <cstdint>

__global__ void pack_data(
	const uint32_t *const unpacked,
	uint32_t *const packed,
	const uint64_t N
);

__host__ __device__ uint32_t complement_word(const uint32_t packed_bases);
__host__ __device__ uint32_t reverse_word(uint32_t word);
__host__ __device__ uint8_t count_word_trailing_n(uint32_t word);

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

__global__ void	new_reversecomplement_kernel(
  uint32_t       *const packed_batch,
  const uint32_t *const batch_lengths,
  const uint32_t *const batch_offsets,
  const uint8_t  *const op,
  const uint32_t        n_tasks
);