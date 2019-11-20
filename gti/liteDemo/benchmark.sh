# liteDemo benchmark

if [ -z $1 ]; then
  echo "$0 <model> <dir> <loop>"
  exit
fi
if [ -z $2 ]; then
  echo "$0 <model> <dir> <loop>"
  exit
fi
if [ -z $3 ]; then
  echo "$0 <model> <dir> <loop>"
  exit
fi
if [ ! -e "liteDemo" ]; then
  echo "Binary \"liteDemo\" does not exist.  Please build before running this script."
  exit
fi

tt=0
i=0
e2etime=0

# warmup
for t in {1..3}; do
    fta=$(./liteDemo $1 $2 | grep FPS | tr -s ' ' | cut -d' ' -f5 )
done

while true; do
    tstart=$(date +%s%N)
    # ./liteDemo ../Models/2803/gti_mnet_fc40_2803.model ../Data/Image_bmp_c1000/beach.bmp
    fta=$(./liteDemo $1 $2 | grep FPS | tr -s ' ' | cut -d' ' -f5 )
    tend=$((($(date +%s%N) - $tstart)/1000000))

    e2etime=$(( $tend + $e2etime ))

    if [ ! -z "$fta" ]; then

        perrun=0
        for ft in $fta ; do
            tt=$( echo "scale=3; $ft + $tt" | bc ) 
            echo "$i ${perrun}: ${ft} ${tt}"
            perrun=$(( $perrun + 1 ))
        done

        i=$(( $i + 1 ))
        if [ $i -ge $3 ]; then 
          break
        fi
    fi
done

at=$( echo "scale=3; $tt / $i / ${perrun} " | bc )
et=$( echo "scale=3; $e2etime / $i / ${perrun} " | bc )

echo "Average ${at}ms per frame"
echo "Average ${et}ms per run"

