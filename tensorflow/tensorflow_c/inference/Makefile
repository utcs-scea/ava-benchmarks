TF_LIB = $(AVA_ROOT)/cava/tf_nw
LIB_NAME = guestlib

## Uncomment these two lines when running on host
#TF_LIB = /usr/local/lib
#LIB_NAME = tensorflow

all:
	gcc inference_c.c -o inference.out \
		-L$(TF_LIB) -l$(LIB_NAME) \
		-Wl,-rpath,$(TF_LIB)

clean:
	rm -f inference.out
