#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 < benchmark-1 > < benchmark-2 >"
    exit
fi

GUEST_AVA_ROOT=/home/hyu/ava
GUEST_DATA_DIR=/home/hyu/gpu-rodinia/data

pssh -h pssh-hosts -l hyu -i " \
    cd ${GUEST_AVA_ROOT}/benchmark/rodinia/micro/rate-limit; \
    AVA_ROOT=${GUEST_AVA_ROOT} AVA_CHANNEL=SHM AVA_WPOOL=TRUE DATA_DIR=${GUEST_DATA_DIR} \
    ./run.sh $1 $2 \
    "
