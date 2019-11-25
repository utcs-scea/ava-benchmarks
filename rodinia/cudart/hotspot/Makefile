include ../util/make.config

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = hotspot.cu

EXE = hotspot

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(CUDA_INC) -L$(CUDA_LIB_DIR) $(CUDA_LIB) --cudart shared

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(CUDA_INC) -L$(CUDA_LIB_DIR)

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(CUDA_INC) -L$(CUDA_LIB_DIR)

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(CUDA_INC) -L$(CUDA_LIB_DIR)

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt
