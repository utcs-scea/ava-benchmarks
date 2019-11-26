#!/usr/bin/env bash

# Run input_data.py to download datasets.

bm="demo  kmeans         nearest_neighbor      \
    linear_regression    random_forest"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTDIR=$DIR/results

mkdir $OUTDIR &>/dev/null

exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

# Save execution time in an array
declare -a time_array
b_idx=0

for b in $bm; do
    i=0
    e2etime=0

    echo -n > $OUTDIR/$b.txt # clean output file
    echo "$(date) # running $b"

    # warm up
    for idx in `seq 1 3`; do
        exe python $b.py
    done

    for idx in `seq 1 7`; do
        tstart=$(date +%s%N)

        exe python $b.py

        tend=$((($(date +%s%N) - $tstart)/1000000))
        e2etime=$(( $tend + $e2etime ))
        i=$(( $i + 1 ))
        exe echo "$(date) # end2end elapsed $tend ms"

        exe echo
        sleep 0.1
    done

    et=$( echo "scale=3; $e2etime / $i " | bc )
    echo "${b}: Average ${et} ms per run"

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
