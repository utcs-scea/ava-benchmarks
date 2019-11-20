
# Setup gti build environment.

CPU_ARCH ?=$(shell uname -m)
OS_TYPE  ?=$(shell uname)
CROSS_COMPILE ?= 0

export LIBFTDI_PATH = $(GTISDKROOT)/Lib/$(OS_TYPE)/$(CPU_ARCH)
export GTILIB_PATH = $(GTISDKROOT)/Lib/$(OS_TYPE)/$(CPU_ARCH)
export GTIINC_PATH = $(GTISDKROOT)/Include

export GTILIB_AVA_PATH = ${AVA_ROOT}/cava/gti_nw

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

all: $(wildcard *.cpp) $(wildcard *.hpp)
	$(GTI_CC) -DLINUX -g -o liteDemo -std=c++11 *.cpp -I $(GTIINC_PATH) \
	-L $(LIBFTDI_PATH) -lGTILibrary-static $(FTDLIB) \
	$(EXTRA_SYS_LIB) -pthread -ldl -lgomp

guest: $(wildcard *.cpp) $(wildcard *.hpp)
	$(GTI_CC) -DLINUX -g -o liteDemo -std=c++11 *.cpp \
	-L$(GTILIB_AVA_PATH) -lguestlib -Wl,-rpath,$(GTILIB_AVA_PATH) \
	$(EXTRA_SYS_LIB) -pthread -ldl

clean:
	rm liteDemo

