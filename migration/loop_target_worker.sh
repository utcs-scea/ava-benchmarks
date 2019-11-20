#!/bin/bash

# run it on target server

cd $AVA_ROOT/cava/cl_nw

for idx in `seq 4001 4150`; do
    sudo ./worker migrate $idx
done
