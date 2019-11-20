TF_LIB = $(AVA_ROOT)/cava/tf_nw
LIB_NAME = guestlib

## Uncomment these two lines when running on host
#TF_LIB = /usr/local/lib
#LIB_NAME = tensorflow

ifdef num_itr
DEFINE_ITR_NUM = -DNUM_ITR=$(num_itr)
else
DEFINE_ITR_NUM =
endif

all:
	gcc demo.c nanojpeg.c -o demo.out \
		$(DEFINE_ITR_NUM) \
		-L$(TF_LIB) -l$(LIB_NAME) \
		-Wl,-rpath,$(TF_LIB)

hello:
	gcc hello_tf.c -L$(TF_LIB) -l$(LIB_NAME) -o hello_tf \
		-Wl,-rpath,$(TF_LIB)

clean:
	rm -f demo.out hello_tf
