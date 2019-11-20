#!/usr/bin/env bash

# Run input_data.py to download datasets.

bm="demo  kmeans         nearest_neighbor      \
    linear_regression    random_forest"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTDIR=$DIR/results

mkdir $OUTDIR &>/dev/null

exe() { echo "++ $@" |& tee -a $OUTDIR/$b.txt ; \
    $@ |& tee -a $OUTDIR/$b.txt ; }

tt=0
i=0
e2etime=0

for b in $bm; do
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
    done

    exe echo
    echo
done

et=$( echo "scale=3; $e2etime / $i " | bc )
echo "Average ${et}ms per run"
