#include "aos.h"

int main(int argc, char **argv) {

    uint64_t client_val = 0;
    if (argc > 1) {
        sscanf(argv[1], "%ld", &client_val);
    }
    printf("Client val: %ld\n", client_val);

    aos_client client_handle = aos_client(client_val);

    /*struct sockaddr_un name;
    int connection_socket;

    memset(&name, 0, sizeof(struct sockaddr_un));

    name.sun_family = AF_UNIX;
    strncpy(name.sun_path, SOCKET_NAME, sizeof(name.sun_path) - 1);
    

    // Send data
    struct aos_socket_command_packet cmd_pckt;
    cmd_pckt.data64 = client_val;
    */
    int i;
    uint64_t addr;
    for (i = 0 ; i < 16; i++) {

        /*connection_socket = socket(AF_UNIX, SOCK_STREAM, 0);
        if (connection_socket == -1) {
           perror("client socket");
           exit(EXIT_FAILURE);
        }
        

        if (connect(connection_socket, (struct sockaddr *) &name, sizeof(struct sockaddr_un)) == -1) {
            perror("client connection");
        }
        
        if (write(connection_socket, &cmd_pckt, sizeof(struct aos_socket_command_packet)) == -1) {
            printf("Client %ld: Unable to write to socket\n", client_val);
        } else {
            if (close(connection_socket) == -1) {
                perror("close error");
            }
            //printf("Client wrote value");
        }
        */
        //client_handle.aos_cntrlreg_write((64*i), 2*i);
        //printf("Client done with write %d\n", i);
        // sleep(5);
        
        addr = 8*i;
        client_handle.aos_cntrlreg_write(addr, i*client_val);
        //printf("Client done with write to addr %ld\n", addr);
        sleep(10);
        client_handle.aos_cntrlreg_read_request(addr);
        //printf("Client done with read request\n");
        sleep(10);
        uint64_t read_value;
        client_handle.aos_cntrlreg_read_response(read_value);
        printf("Client %ld read back value %ld \n", client_val, read_value);

    }

    printf("Client %ld exited normally\n", client_val);
    return 0;

}