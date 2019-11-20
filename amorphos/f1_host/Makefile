VPATH = src:include:$(HDK_DIR)/common/software/src:$(HDK_DIR)/common/software/include

INCLUDES = -I$(SDK_DIR)/userspace/include
INCLUDES += -I $(HDK_DIR)/common/software/include
INCLUDES += -I ./include

CC = g++
CFLAGS = -std=c++11 -fpermissive -DCONFIG_LOGLEVEL=4 -g -Wall $(INCLUDES)
CLIENT_CFLAGS = -std=c++11 -fpermissive -DCONFIG_LOGLEVEL=4 -g -Wall

LDLIBS = -lfpga_mgmt -lrt -lpthread
CLIENT_LDLIBS = -lrt -lpthread

SRC = ${SDK_DIR}/userspace/utils/sh_dpi_tasks.c

all: client_build memdrive_build bitcoin_build ava_wrapper ava_bitcoin_build

daemon_build: aos_daemon.cpp aos.h aos_scheduler.cpp aos_host_common.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) $(SRC) aos_host_common.cpp aos_daemon.cpp aos_scheduler.cpp -o daemon

client_build: aos.h client.c
	$(CC) $(CLIENT_CFLAGS) $(CLIENT_LDLIBS) client.c -o client

memdrive_build: aos.h memdrive_client.c
	$(CC) $(CLIENT_CFLAGS) $(CLIENT_LDLIBS) memdrive_client.c -o memdrive_client

bitcoin_build: aos.h bitcoin_client.cpp
	$(CC) $(CLIENT_CFLAGS) $(CLIENT_LDLIBS) bitcoin_client.cpp -o bitcoin_client

ava_wrapper: aos.h ava_aos_wrapper.cpp
	$(CC) $(CLIENT_CFLAGS) $(CLIENT_LDLIBS) -fPIC -shared ava_aos_wrapper.cpp -o libava_aos_wrapper.so

ava_bitcoin_build: aos.h ava_bitcoin_client.cpp libava_aos_wrapper.so
	$(CC) $(CLIENT_CFLAGS) $(CLIENT_LDLIBS) -L. ava_bitcoin_client.cpp -o ava_bitcoin_client -lava_aos_wrapper

clean: daemon client libava_aos_wrapper.so
	rm -f daemon client memdrive_client bitcoin_client libava_aos_wrapper.so
	rm -f /tmp/aos_daemon.socket
