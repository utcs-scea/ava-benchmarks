
# Setup gti build environment.

CPU_ARCH ?=$(shell uname -m)
OS_TYPE  ?=$(shell uname)
CROSS_COMPILE ?= 0

export LIBFTDI_PATH = $(GTISDKROOT)/Lib/$(OS_TYPE)/$(CPU_ARCH)
export GTILIB_PATH = $(GTISDKROOT)/Lib/$(OS_TYPE)/$(CPU_ARCH)
export GTIINC_PATH = $(GTISDKROOT)/Include

export LIBOPENCV_PATH = $(GTISDKROOT)/Lib/$(OS_TYPE)/OpenCV/$(CPU_ARCH)
export INCOPENCV_PATH = $(GTISDKROOT)/Lib/$(OS_TYPE)/OpenCV

export GTILIB_AVA_PATH = ${AVA_ROOT}/nwcc/gti_nw

# Makefile

ifeq ($(CROSS_COMPILE), 1)
$(info CROSS_COMPILE is $(CROSS_COMPILE))
FTDLIB = -l:libftd3xx.so -Wl,-rpath=$(LIBFTDI_PATH)
EXTRA_SYS_LIB =
else
CROSS_COMPILE=
# anticipate the library on an embedded platform is older
ifneq ($(CPU_ARCH),x86_64)
FTDLIB = -l:libftd3xx.so -Wl,-rpath=$(LIBFTDI_PATH)
EXTRA_SYS_LIB = -l:libstdc++.so.6
else
FTDLIB = -lftd3xx-static
EXTRA_SYS_LIB =
endif
endif

GTI_CC ?= g++
TARGET = pipelineDemo

all: $(wildcard *.cpp) $(wildcard *.hpp)
	$(GTI_CC)  -o $(TARGET) -std=c++11 pipelineDemo.cpp -DLINUX -I $(GTIINC_PATH) \
	-I $(INCOPENCV_PATH) -L $(LIBOPENCV_PATH) -l:libopencv_world.so.3.2 \
	-L $(LIBFTDI_PATH) -lGTILibrary-static $(FTDLIB) -lgomp\
	$(EXTRA_SYS_LIB) -pthread -ldl 

guest: $(wildcard *.cpp) $(wildcard *.hpp)
	$(GTI_CC) -o $(TARGET) -std=c++11 pipelineDemo.cpp -DLINUX \
	-L$(GTILIB_AVA_PATH) -lguestlib -Wl,-rpath,$(GTILIB_AVA_PATH) \
	$(EXTRA_SYS_LIB) -pthread -ldl

clean:
	rm $(TARGET)

