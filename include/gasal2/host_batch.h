#pragma once

#include <cstdint>

// host data structure methods
host_batch_t *gasal_host_batch_new(uint32_t batch_bytes, uint32_t offset);
void gasal_host_batch_destroy(host_batch_t *res); 																		// destructor
host_batch_t *gasal_host_batch_getlast(host_batch_t *arg);
void gasal_host_batch_reset(gasal_gpu_storage_t &gpu_storage);              // get last item of chain
uint32_t gasal_host_batch_fill(gasal_gpu_storage_t &gpu_storage, uint32_t idx, const char* data, uint32_t size, DataSource SRC); 	// fill the data
uint32_t gasal_host_batch_add(gasal_gpu_storage_t &gpu_storage, uint32_t idx, const char *data, uint32_t size, DataSource SRC);
uint32_t gasal_host_batch_addbase(gasal_gpu_storage_t &gpu_storage, uint32_t idx, const char base, DataSource SRC);
void gasal_host_batch_print(const host_batch_t &res); 																		// printer
void gasal_host_batch_printall(const host_batch_t &res);																		// printer for the whole linked list
