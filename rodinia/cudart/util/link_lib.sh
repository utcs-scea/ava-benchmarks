#!/bin/bash

if [ -z "$AVA_ROOT" ]
then
    echo "AVA_ROOT is not set"
    return
fi

ln -s $AVA_ROOT/cava/cudart_nw/libguestlib.so libcuda.so.1
ln -s $AVA_ROOT/cava/cudart_nw/libguestlib.so libcudart.so.10.0
