#!/bin/bash

source environments

bm="backprop  bfs        b+tree dwt2d      gaussian \
    heartwall hybridsort kmeans lavaMD     lud  \
    myocyte   nn         nw     pathfinder srad"

CL_DIR=$DIR/../opencl
OUTDIR=$DIR/cl_results_${AVA_CHANNEL}_${AVA_WPOOL}

mkdir $OUTDIR &>/dev/null

exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    "$@" |& tee -a $OUTDIR/$b.txt ; }

for b in $bm; do
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $CL_DIR/$b
    make clean && make
    for idx in `seq 1 ${WARMUP_TIMES}`; do
        ./run -p 0 -d 0
    done
    for idx in `seq 1 ${EVAL_TIMES}`; do
        exe /usr/bin/time -f "real %e seconds" ./run -p 0 -d 0
        exe echo
    done
    exe echo
    echo
done
