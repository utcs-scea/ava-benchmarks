COUNTER=0
while [ $COUNTER -lt 10 ]; do
    ./demo_movidius data/inception_v3_movidius.graph data/imagenet_slim_labels.txt data/grace_hopper.jpg | \
    grep "^Total time" | \
    awk '{ total += $4 } END { print total }'
    let COUNTER+=1
done
echo $COUNTER
