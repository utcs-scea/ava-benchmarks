
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
if [ ! -e "pipelineDemo" ]; then
  echo "Binary \"pipelineDemo\" does not exist.  Please build before running this script."
  exit
fi

tt=0
i=0

while true; do

fta=$( ./pipelineDemo $1 $2 | grep FPS | tr -s ' ' | cut -d' ' -f6 )

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

echo "Average ${at}ms per frame"

