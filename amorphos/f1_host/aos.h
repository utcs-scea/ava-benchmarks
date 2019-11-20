#ifndef aos_h__
#define aos_h__
// Normal includes
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <syslog.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <assert.h>
#include <string>
#include <iostream>
#include <map>
#include <queue>

#define SOCKET_NAME "/tmp/aos_daemon.socket"
#define SOCKET_FAMILY AF_UNIX
#define SOCKET_TYPE SOCK_STREAM

#define BACKLOG 128

enum class aos_socket_command {
    CNTRLREG_READ_REQUEST,
    CNTRLREG_READ_RESPONSE,
    CNTRLREG_WRITE_REQUEST,
    CNTRLREG_WRITE_RESPONSE,
    BULKDATA_READ_REQUEST,
    BULKDATA_READ_RESPONSE,
    BULKDATA_WRITE_REQUEST,
    BULKDATA_WRITE_RESPONSE
};

enum class aos_errcode {
    SUCCESS = 0,
    RETRY, // for reads
    ALIGNMENT_FAILURE,
    PROTECTION_FAILURE,
    APP_DOES_NOT_EXIST,
    TIMEOUT,
    UNKNOWN_FAILURE
};

struct aos_app_handle  {
    uint64_t app_id;
    uint64_t key;
};

struct aos_socket_command_packet {
    aos_socket_command command_type;
    uint64_t app_id;
    uint64_t addr64;
    uint64_t data64; 
};

struct aos_socket_response_packet {
    aos_errcode errorcode;
    uint64_t    data64;
};

class aos_client {
public:

    aos_client(uint64_t new_app_id) :
        app_id(new_app_id),
        connection_socket(0),
        connectionOpen(false),
        intialized(true)
    {
        // Setup the struct needed to connect the aos daemon
        memset(&socket_name, 0, sizeof(struct sockaddr_un));
        socket_name.sun_family = SOCKET_FAMILY;
        strncpy(socket_name.sun_path, SOCKET_NAME, sizeof(socket_name.sun_path) - 1);
    }

    aos_errcode aos_cntrlreg_write(uint64_t addr, uint64_t value) {
        // Open the socket
        openSocket();
        // Create the packet
        aos_socket_command_packet cmd_pckt;
        cmd_pckt.command_type = aos_socket_command::CNTRLREG_WRITE_REQUEST;
        cmd_pckt.app_id = app_id;
        cmd_pckt.addr64 = addr;
        cmd_pckt.data64 = value;
        // Send over the request
        writeCommandPacket(cmd_pckt);
        // close socket
        closeSocket();
        // Return success/error condition
        return aos_errcode::SUCCESS;
    }

    aos_errcode aos_cntrlreg_read(uint64_t addr, uint64_t & value) {
        aos_errcode errorcode = aos_cntrlreg_read_request(addr);
        // do some error checking
        errorcode = aos_cntrlreg_read_response(value);
        return errorcode;
    }

    aos_errcode aos_cntrlreg_read_request(uint64_t addr) {
        // Open the socket
        openSocket();
        // Create the packet
        aos_socket_command_packet cmd_pckt;
        cmd_pckt.command_type = aos_socket_command::CNTRLREG_READ_REQUEST;
        cmd_pckt.app_id = app_id;
        cmd_pckt.addr64 = addr;
        // Send over the request
        writeCommandPacket(cmd_pckt);
        // close socket
        closeSocket();
        // Return success/error condition
        return aos_errcode::SUCCESS;
    }

    aos_errcode aos_cntrlreg_read_response(uint64_t & value) {
        // Open the socket
        openSocket();
        // Create the packet
        aos_socket_command_packet cmd_pckt;
        cmd_pckt.command_type = aos_socket_command::CNTRLREG_READ_RESPONSE;
        cmd_pckt.app_id = app_id;
        // send over the request
        writeCommandPacket(cmd_pckt);
        // read the response packet
        aos_socket_response_packet resp_pckt;
        readResponsePacket(resp_pckt);
        // close the socket
        closeSocket();
        // copy over the data
        value = resp_pckt.data64;

        return aos_errcode::SUCCESS;
    }

    aos_errcode aos_bulkdata_write(uint64_t addr, void * buf, size_t numBytes) {
        return aos_errcode::SUCCESS;
    }

    aos_errcode aos_bulkdata_read_request(uint64_t addr, size_t numBytes) {
        return aos_errcode::SUCCESS;
    }

    aos_errcode aos_bulkdata_read_response(void * buf) {
        return aos_errcode::SUCCESS;
    }

    void printError(std::string errStr) {
        std::cout << errStr << std::endl;
    }

private:
    sockaddr_un socket_name;
    uint64_t app_id;
    int connection_socket;
    bool connectionOpen;
    bool intialized;

    void openSocket() {
        if (connectionOpen)  {
            printError("Can't open already open socket");
        }
        connection_socket = socket(SOCKET_FAMILY, SOCKET_TYPE, 0);
        if (connection_socket == -1) {
           perror("client socket");
           exit(EXIT_FAILURE);
        }

        if (connect(connection_socket, (sockaddr *) &socket_name, sizeof(sockaddr_un)) == -1) {
            perror("client connection");
        }
        connectionOpen = true;
    }

    void closeSocket() {
        if (!connectionOpen) {
            printError("Can't close a socket that isn't open");
        }
        if (close(connection_socket) == -1) {
            perror("close error on client");
        }
        connectionOpen = false;
    }

    int writeCommandPacket(aos_socket_command_packet & cmd_pckt) {
        if (!connectionOpen) {
            printError("Can't write command packet without an open socket");
        }
        if (write(connection_socket, &cmd_pckt, sizeof(aos_socket_command_packet)) == -1) {
            printf("Client %ld: Unable to write to socket\n", app_id);
            perror("client write");
        }
        // return success/error
        return 0;
    }

    int readResponsePacket(aos_socket_response_packet & resp_pckt) {
        if (!connectionOpen) {
            printError("Can't close a socket that isn't open"); 
        }
        if (read(connection_socket, &resp_pckt, sizeof(aos_socket_response_packet)) == -1) {
            perror("Unable to read respone packet from daemon");
        }
        return 0;
    }

};

#endif // end aos_h__