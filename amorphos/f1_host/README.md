# AmorphOS Host Interface

Table of Contents
- Overview
- Client Interface

1. A daemon runs on the host system that is able to response to multiple clients and controls their access to the FPGA. Currently,
the interface is limited to CntrlReg read/writes and BulkData read/writes.

2. The client interface is very simple to use and requires the following steps.

a) include the aos.h header file in your code

b) Create an aos_client object, currently the only requirement is the app id

c) The object has request/response methods for the two current interfaces and their signatures are as follows.

    // CntrlReg
    aos_errcode aos_cntrlreg_write(uint64_t addr, uint64_t value);
    aos_errcode aos_cntrlreg_read(uint64_t addr, uint64_t & value);
    aos_errcode aos_cntrlreg_read_request(uint64_t addr); // decouples request from response
    aos_errcode aos_cntrlreg_read_response(uint64_t & value); // decouples response from request
    // Bulk Data
    aos_errcode aos_bulkdata_write(uint64_t addr, void * buf, size_t numBytes);
    aos_errcode aos_bulkdata_read_request(uint64_t addr, size_t numBytes);
    aos_errcode aos_bulkdata_read_response(void * buf);

    addr always refers to an address in the application on the FPGA. Currently the cntrlreg and bulkdata address spaces are seperate. The contents of
    DRAM maybe mapped to the BulkData interface at some point. aos_errcode is a status code returned by each API call
    
d) Example of using the host interface to write to app 0 on the FPGA.

#include "aos.h"

int main(int argc, char **argv) {

    aos_client client_handle = aos_client(0); // APP ID 0

    uint64_t valToWrite = 45;  
    uint64_t addrOnFpga = 128; // address must be 8-byte (64 bit aligned)
      
    if (client_handle.aos_cntrlreg_write(addrOnFpga, valToWrite) != aos_errcode::SUCCESS) {
        printf("Failed to write to CntrlReg");
    } else {
        printf("Successfully wrote to CntrlReg");
    }

    return 0;
}

