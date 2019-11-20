#ifndef __AVA_AOS_WRAPPER_H__
#define __AVA_AOS_WRAPPER_H__

#ifndef __CAVA__
#include "aos.h"

extern "C" {
#endif

struct aos_client_wrapper;

struct aos_client_wrapper *ava_client_new(uint64_t app_id);
void ava_client_free(struct aos_client_wrapper *client_handle);
int ava_cntrlreg_write(struct aos_client_wrapper *client_handle, uint64_t addr, uint64_t value);
int ava_cntrlreg_read(struct aos_client_wrapper *client_handle, uint64_t addr, uint64_t *value);

#ifndef __CAVA__
}
#endif

#endif
