source environments

# OpenCL

col="Init MemAlloc HtoD HtoD_Phy Exec DtoH DtoH_Phy Close API Total real"
OUTDIR=$DIR/cl_results_${AVA_CHANNEL}_${AVA_WPOOL}

if [ ! -d "$OUTDIR" ]; then
    echo "$OUTDIR does not exist."
    echo "Please run test_cl.sh to collect data before parse OpenCL benchmark performance;"
    printf "or update \$OUTDIR\'s value in this script.\n\n"
else
    bm=`ls $OUTDIR`
    TMPFILE=$DIR/.tmp

    cd $OUTDIR
    echo "OpenCL"
    echo -n "GPU,"; echo $col | tr ' ' ','
    for b in $bm; do
        echo -n ${b::-4}
        result=""
        echo -n > $TMPFILE
        for c in $col; do
            grep ^$c $OUTDIR/$b | \
                awk '{ total += $2; count++ } END {
                       if (count > 0)
                           print total/count;
                       else
                           print -1
                     }' \
                >> $TMPFILE
        done
        echo -n ","; cat $TMPFILE | paste -sd "," -
        rm $TMPFILE
    done
fi

# CUDA

col="Init MemAlloc HtoD Exec DtoH Close API Total real"
OUTDIR=$DIR/cu_results_${AVA_CHANNEL}_${AVA_WPOOL}

if [ ! -d "$OUTDIR" ]; then
    echo "$OUTDIR does not exist."
    echo "Please run test_cu.sh to collect data before parse CUDA benchmark performance;"
    printf "or update \$OUTDIR\'s value in this script.\n\n"
else
    bm=`ls $OUTDIR`
    TMPFILE=$DIR/.tmp

    cd $OUTDIR
    echo
    echo "CUDA"
    echo -n "GPU,"; echo $col | tr ' ' ','
    for b in $bm; do
        echo -n ${b::-4}
        result=""
        echo -n > $TMPFILE
        for c in $col; do
            grep ^$c $OUTDIR/$b | \
                awk '{ total += $2; count++ } END {
                       if (count > 0)
                           print total/count;
                       else
                           print -1
                     }' \
                >> $TMPFILE
        done
        echo -n ","; cat $TMPFILE | paste -sd "," -
        rm $TMPFILE
    done
fi
