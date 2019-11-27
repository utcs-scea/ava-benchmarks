#!/bin/bash

if [ "$HOSTNAME" == "dev" ]; then
    cd opencl/$1
    echo "Execute $1"
elif [ "$HOSTNAME" == "dev1" ]; then
    cd opencl/$2
    echo "Execute $2"
else
    echo "Unknown hostname"
    exit
fi

make clean &>/dev/null
make >/dev/null
./run -p 0 -d 0
