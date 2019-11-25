include ../util/make.config

LOCAL_CC = gcc -g -O3 -Wall
CC := $(CUDA_DIR)/bin/nvcc

all : nn hurricane_gen

clean :
	rm -rf *.o nn hurricane_gen

nn : nn_cuda.cu
	$(CC) -cuda nn_cuda.cu --cudart shared
	$(CC) -o nn nn_cuda.cu --cudart shared

clang: $(SRC)
	clang++ nn_cuda.cu -o nn -I../util --cuda-gpu-arch=sm_30 \
		-I$(CUDA_INC) -L$(CUDA_LIB_DIR) $(CUDA_LIB) -ldl -lrt -pthread -DTIMING

hurricane_gen : hurricane_gen.c
	$(LOCAL_CC) -o $@ $< -lm

#data :
#	mkdir data
#	./gen_dataset.sh
