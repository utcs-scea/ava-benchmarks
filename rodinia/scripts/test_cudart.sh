#!/bin/bash

source environments

bm="backprop   bfs      b+tree dwt2d gaussian hotspot \
    lavaMD     lud     myocyte nn    nw       particlefilter \
    pathfinder srad_v1 srad_v2 streamcluster"

CUDART_DIR=$DIR/../cudart
OUTDIR=$DIR/cudart_results_${AVA_CHANNEL}_${AVA_WPOOL}

mkdir $OUTDIR &>/dev/null

exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    "$@" |& tee -a $OUTDIR/$b.txt ; }

# Save execution time in an array
declare -a time_array
b_idx=0

for b in $bm; do
    i=0
    e2etime=0

    echo -n > $OUTDIR/$b.txt # clean output file

    cd $CUDART_DIR/$b
    echo "$(date) # compiling $b"
    make clean &>/dev/null ; make &>/dev/null

    # warm up
    echo "$(date) # warming $b"
    for idx in `seq 1 ${WARMUP_TIMES}`; do
        ./run
        sleep 0.1
    done

    # test
    echo "$(date) # running $b"
    for idx in `seq 1 ${EVAL_TIMES}`; do
        tstart=$(date +%s%N)

        exe ./run

        tend=$((($(date +%s%N) - $tstart)/1000000))
        e2etime=$(( $tend + $e2etime ))
        i=$(( $i + 1 ))
        exe echo "$(date) # end2end elapsed $tend ms"

        exe echo
        sleep 0.1
    done

    et=$( echo "scale=3; $e2etime / $i " | bc )
    exe echo "${b}: Average ${et} ms per run"

    time_array[$b_idx]=${et}
    b_idx=$((b_idx+1))

    exe echo
    echo
done

b_idx=0
for b in $bm; do
    echo "${b}: Average ${time_array[$b_idx]} ms per run"
    b_idx=$((b_idx+1))
done
