#pragma once

gasal_res_t *gasal_res_new_host(uint32_t max_n_alns, const Parameters &params);
gasal_res_t *gasal_res_new_device(gasal_res_t *device_cpy);
gasal_res_t *gasal_res_new_device_cpy(uint32_t max_n_alns, const Parameters &params);

void gasal_res_destroy_host(gasal_res_t *res);
void gasal_res_destroy_device(gasal_res_t *device_res, gasal_res_t *device_cpy);
