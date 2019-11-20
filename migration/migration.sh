#!/bin/bash

# run it on source guest VM

bm=(backprop  bfs        b+tree dwt2d      gaussian \
    heartwall hybridsort kmeans lavaMD     lud  \
    myocyte   nn         nw     pathfinder srad)
calls=(r53    r208       r107   r128       r24595 \
       r634   r238       r271   r38        r2314 \
       r215097 r27       r305   r93        r479)
num=(50 50 50 50 150 \
     50 50 50 50 150 \
     200 50 50 50 50)
WARMUP_TIMES=3

exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    "$@" |& tee -a $OUTDIR/$b.txt ; }

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CL_DIR=$DIR
WORKER_DIR=$DIR/../../worker
OUTDIR=$DIR/migration_results

mkdir $OUTDIR &>/dev/null

for i in "${!bm[@]}"; do
    b=${bm[i]}
    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"
    cd $CL_DIR/$b
    make clean && make

    #unset AVA_MIGRATION_CALL_ID
    #for idx in `seq 1 ${WARMUP_TIMES}`; do
    #    sleep 1
    #    ./run -p 0 -d 0
    #done

    export AVA_MIGRATION_CALL_ID=${calls[i]}
    echo > $WORKER_DIR/migration.log

    for idx in `seq 1 ${num[i]}`; do
        sleep 1
        exe /usr/bin/time -f "real %e seconds" ./run -p 0 -d 0
        exe echo
    done
    exe echo
    echo

    cp $WORKER_DIR/migration.log $OUTDIR/${b}_worker.log
    sleep 1
done
