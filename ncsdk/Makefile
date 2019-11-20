MVNC_LIB = ${AVA_ROOT}/cava/mvnc_nw

all: ava

ava:
	gcc demo_movidius.c nanojpeg.c -o demo_movidius \
		-Wl,-rpath,$(MVNC_LIB) \
		-I../../nwcc/files -L$(MVNC_LIB) -lguestlib

movidius:
	gcc demo_movidius.c nanojpeg.c -o demo_movidius \
		-lmvnc
