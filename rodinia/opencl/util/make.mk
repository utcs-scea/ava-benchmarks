OPENCL_INC = /usr/include

OPENCL_LIB = /usr/local/cuda-10.0/lib64
LIB_NAME = OpenCL

# Comment out to use native library on the host
OPENCL_LIB = $(AVA_ROOT)/cava/cl_nw
LIB_NAME = guestlib

OCL_LIB = -l$(LIB_NAME)
