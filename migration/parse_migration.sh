#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

bm="backprop  bfs        b+tree dwt2d      gaussian \
    heartwall hybridsort kmeans lavaMD     lud  \
    myocyte   nn         nw     pathfinder srad"

OUTDIR=$DIR/migration_results

for b in $bm; do
    echo -n $b,
    grep "start live migration at call_id" $OUTDIR/$b.txt | \
        sed 's/^.*\(start\slive\smigration\sat\scall_id\s.*\).*$/\1/' | \
        awk '{print $6}' | \
        paste -sd ","

    echo -n ","
    grep migration $OUTDIR/${b}_worker.log | \
        awk '{print $4}' | \
        paste -sd ","
done


