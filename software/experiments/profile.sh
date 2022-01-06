#!/bin/bash
declare -a arr=("element1" "element2" "element3")

declare -a MODELS=("lenet_ll" "lenet_two_third" "lenet_all" "resnet_ll" "resnet_one_third" "resnet_half" "resnet_two_third" "resnet_all" "vgg_ll" "vgg_one_third" "vgg_half" "vgg_two_third" "vgg_all")
declare -a SAMPLES=(3 4 5 6 7 8 9 10 20 50 100)

profile_path="profiling_$(date +"%d-%m-%y_%T")"
echo $profile_path
for m in ${MODELS[@]}; do
    for s in ${SAMPLES[@]}; do
        echo $m $s
        python3 profile_sample.py --save "./$profile_path" --model_option "$m" --samples "$s" --gpu "$1"
    done
done
