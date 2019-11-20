#include <stdint.h>
#include <sys/time.h>
#include "ava_aos_wrapper.h"

struct timeval start;
struct timeval end;

int main(int argc, char **argv) {

    uint64_t client_val = 0;
    if (argc > 1) {
        sscanf(argv[1], "%ld", &client_val);
    } else {
        printf("App ID required\n");
        return 0;
    }
    printf("Bitcoin Client app id: %ld\n", client_val);

    gettimeofday(&start, NULL);

    struct aos_client_wrapper *client_handle;
    client_handle = ava_client_new(client_val);

    //// Init Bitcoin
    // example from file, one expected output is 32'h0e33337a or 238,236,538
    // note that real bitcoin data will likely need to have the bytes reversed first
    const char* midstate_s = "228ea4732a3c9ba860c009cda7252b9161a5e75ec8c582a5f106abb3af41f790";
    const char *hash_data_s = "2194261a9395e64dbed17115";

    // convert hex to bits, requires little endian system (e.g. x86)
    uint64_t midstate[4], hash_data[2];
    sscanf(midstate_s, "%16lx%16lx%16lx%16lx", &midstate[3], &midstate[2], &midstate[1], &midstate[0]);
    sscanf(hash_data_s, "%8lx%16lx", &hash_data[1], &hash_data[0]);

    // actually transfer data
    uint32_t addr = (1 << 9);
    for (int i = 0; i < 4; ++i) {
        ava_cntrlreg_write(client_handle, addr, midstate[i]);
        addr += 64;
    }
    for (int i = 0; i < 2; ++i) {
        ava_cntrlreg_write(client_handle, addr, hash_data[i]);
        addr += 64;
    }

    //// Wait on output
    int num_tries = 0;
    while (num_tries++ < 5) {
        uint64_t nonce = ~uint64_t{0};
        while (nonce == ~uint64_t{0}) {
            usleep(10000);
            ava_cntrlreg_read(client_handle, (1 << 9), &nonce);
        }
        printf("Received nonce: %lu (0x%lx)\n", nonce, nonce);
    }

    ava_client_free(client_handle);

    gettimeofday(&end, NULL);
    printf("Total time=%f ms\n", (end.tv_sec * 1000.0 + (float)end.tv_usec / 1000.0)
                               - (start.tv_sec * 1000.0 + (float)start.tv_usec / 1000.0));

    return 0;
}
