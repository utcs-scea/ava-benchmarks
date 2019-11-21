source environments

# OpenCL

col="Init MemAlloc HtoD HtoD_Phy Exec DtoH DtoH_Phy Close API Total real"
OUTDIR=$DIR/cl_results_${AVA_CHANNEL}_${AVA_WPOOL}
bm=`ls $OUTDIR`
TMPFILE=$DIR/.tmp

cd $OUTDIR
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

# CUDA

col="Init MemAlloc HtoD Exec DtoH Close API Total real"
OUTDIR=$DIR/cu_results_${AVA_CHANNEL}_${AVA_WPOOL}
bm=`ls $OUTDIR`
TMPFILE=$DIR/.tmp

cd $OUTDIR
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
