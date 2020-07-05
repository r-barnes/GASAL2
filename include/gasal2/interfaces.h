#pragma once

#include <cstdint>
#include <vector>

// Resizer for the whole gpu_storage in terms of number of sequences
void gasal_host_alns_resize(gasal_gpu_storage_t &gpu_storage, int new_max_alns, const Parameters &params);

// operation filler method (field in the gasal_gpu_storage_t field)
void gasal_op_fill(gasal_gpu_storage_t &gpu_storage, const uint8_t *data, uint32_t nbr_seqs_in_stream, DataSource src);

void gasal_set_device(int gpu_select = 0, bool isPrintingProp = true);
