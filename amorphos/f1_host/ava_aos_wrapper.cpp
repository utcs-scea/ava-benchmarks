#include "ava_aos_wrapper.h"

struct aos_client_wrapper *ava_client_new(uint64_t app_id)
{
    aos_client *client_handle = new aos_client(app_id);
    return (struct aos_client_wrapper *)client_handle;
}

void ava_client_free(struct aos_client_wrapper *client_handle)
{
    aos_client *client = (aos_client *)client_handle;
    delete client;
}

int ava_cntrlreg_write(struct aos_client_wrapper *client_handle, uint64_t addr, uint64_t value)
{
    aos_errcode ret;
    aos_client *client = (aos_client *)client_handle;
    ret = client->aos_cntrlreg_write(addr, value);
    return (int)ret;
}


int ava_cntrlreg_read(struct aos_client_wrapper *client_handle, uint64_t addr, uint64_t *value)
{
    aos_errcode ret;
    uint64_t tmp_value;
    aos_client *client = (aos_client *)client_handle;
    ret = client->aos_cntrlreg_read(addr, tmp_value);
    *value = tmp_value;
    return (int)ret;
}
