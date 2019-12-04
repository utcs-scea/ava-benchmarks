#!/bin/bash

if [ -z "$AVA_ROOT" ]
then
    echo "AVA_ROOT is not set"
    exit
fi

BASEDIR=$(dirname "$0")

ln -s $AVA_ROOT/cava/cudart_nw/libguestlib.so $BASEDIR/libcuda.so.1
ln -s $AVA_ROOT/cava/cudart_nw/libguestlib.so $BASEDIR/libcudart.so.10.0
ln -s $AVA_ROOT/cava/cudart_nw/libguestlib.so $BASEDIR/libcudnn.so.7
ln -s $AVA_ROOT/cava/cudart_nw/libguestlib.so $BASEDIR/libcublas.so.10.0
