#include "aos.h"

int main(int argc, char **argv) {

    uint64_t client_val = 0;
    if (argc > 1) {
        sscanf(argv[1], "%ld", &client_val);
    } else {
        printf("App ID required\n");
        return 0;
    }
    printf("Memdrive Client app id: %ld\n", client_val);

    aos_client client_handle = aos_client(client_val);

    // Program Mem Drive
    uint64_t start_addr0 = 0x0;
    uint64_t total_subs  = 0x15;
    uint64_t mask = 0xFFFFFFFFFFFFFFFF;
    uint64_t mode = 0x1; // write == 1, read = 0
    uint64_t start_addr1 = 0xC000;
    uint64_t addr_delta  = 6;
    uint64_t canary0     = 0xFEEBFEEBBEEFBEEF;
    uint64_t canary1     = 0xDAEDDAEDDEADDEAD;

    client_handle.aos_cntrlreg_write(0x00, start_addr0);
    client_handle.aos_cntrlreg_write(0x08, total_subs);
    client_handle.aos_cntrlreg_write(0x10, mask);
    client_handle.aos_cntrlreg_write(0x18, mode);
    client_handle.aos_cntrlreg_write(0x20, start_addr1);
    client_handle.aos_cntrlreg_write(0x28, addr_delta);
    client_handle.aos_cntrlreg_write(0x30, canary0);
    client_handle.aos_cntrlreg_write(0x38, canary1);

    // Read back runtime
    uint64_t start_cycle;
    uint64_t end_cycle;

    client_handle.aos_cntrlreg_read(0x00ULL, start_cycle);
    client_handle.aos_cntrlreg_read(0x08ULL, end_cycle);

    printf("Memdrive app %ld start cycle: %ld\n", client_val, start_cycle);
    printf("Memdrive app %ld end cycle: %ld\n", client_val, end_cycle);
    printf("Memdrive app %ld runtime: %ld\n"  , client_val, (end_cycle - start_cycle));

    printf("Memdrive client exited normally\n");

    return 0;

}