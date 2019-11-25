include ../util/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_INC)

SRC = gaussian.cu
EXE = gaussian

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) $(CUDA_LIB) --cudart shared

clang: $(SRC)
	clang++ $(SRC) -o $(EXE) -I../util --cuda-gpu-arch=sm_30 \
		-L$(CUDA_INC) $(CUDA_LIB) -ldl -lrt -pthread -DTIMING

clean:
	rm gaussian
