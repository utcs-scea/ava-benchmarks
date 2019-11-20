#!/bin/bash

model=("../Models/2803/gti_mnet_fc40_2803.model" \
       "../Models/2803/gti_resnet18_2803.model")
image=("../Data/Image_bmp_c1000/beach.bmp"  \
       "../Data/Image_bmp_c20/aquarium.jpg" \
       "../Data/Image_bmp_c40/Coffe.jpg")

for m in $(seq 1 ${#model[@]}) ; do
    for i in $(seq 1 ${#image[@]}) ; do
        echo ${model[$m-1]} ${image[$i-1]}
        ./benchmark.sh ${model[$m-1]} ${image[$i-1]} 5
        echo
    done
done
