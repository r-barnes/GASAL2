#pragma once

#include <gasal2/gasal.h>

#include <cstdint>

__global__ void gasal_banded_tiled_kernel(
	uint32_t *packed_query_batch,
	uint32_t *packed_target_batch,
	uint32_t *query_batch_lens,
	uint32_t *target_batch_lens,
	uint32_t *query_batch_offsets,
	uint32_t *target_batch_offsets,
	gasal_res_t *device_res,
	int n_tasks,
	const int32_t k_band_width
);