#!/bin/bash

SAMPLES=5
ret_val=""
GPU=$1
function run_experiments {
    declare -a samples_list=()
    for i in $(seq 1 $SAMPLES);
    do 
        local sample=$(python3 $1 --seed $i --gpu $GPU | grep Experiment | cut -d " " -f 4)
        echo $sample
        samples_list+=("../${sample}")
    done
    echo "${samples_list[*]}"
    if [ ! -d "./summary" ]; then
        mkdir summary
    fi 
    cd ./summary/
    result=$(python3 ../$2 --result_paths "${samples_list[*]}" --label $3  | grep Experiment | cut -d " " -f 4)
    ret_val=$result
    cd ../
}

function run_q_experiments {
    local float_samples=(./../default/*"$4"*)
    echo "${float_samples[*]}"
    declare -a samples_list=()
    for i in $(seq 1 $SAMPLES);
    do 
        local index=${i}-1
        echo "${float_samples[$index]}"
        local sample=$(python3 $1 --seed $i --gpu $GPU --load ${float_samples[$index]} | grep Experiment | cut -d " " -f 4)
        echo $sample
        samples_list+=("../${sample}")
    done
    if [ ! -d "./summary" ]; then
        mkdir summary
    fi 
    cd ./summary/
    echo "${samples_list[*]}"
    result=$(python3 ../$2 --result_paths "${samples_list[*]}" --label $3 | grep Experiment | cut -d " " -f 4)
    ret_val=$result
    cd ../
}

cd ./pointwise/
#run_experiments mnist.py ../average_results.py float_pointwise_mnist
#run_experiments cifar.py ../average_results.py float_pointwise_cifar
#run_experiments svhn.py ../average_results.py float_pointwise_svhn

cd ./quantised/
#run_q_experiments mnist.py ../../average_results.py q_pointwise_mnist -mnist-
#run_q_experiments cifar.py ../../average_results.py q_pointwise_cifar -cifar-
#run_q_experiments svhn.py ../../average_results.py q_pointwise_svhn -svhn-
cd ../../

cd ./ll/
#run_experiments mnist.py ../average_results.py float_ll_mnist
#run_experiments cifar.py ../average_results.py float_ll_cifar
#run_experiments svhn.py ../average_results.py float_ll_svhn

cd ./quantised/
#run_q_experiments mnist.py ../../average_results.py q_ll_mnist -mnist-
#run_q_experiments cifar.py ../../average_results.py q_ll_cifar -cifar-
#run_q_experiments svhn.py ../../average_results.py q_ll_svhn -svhn-
cd ../../

cd ./one_third/
#run_experiments cifar.py ../average_results.py float_one_third_cifar
#run_experiments svhn.py ../average_results.py float_one_third_svhn

cd ./quantised/
#run_q_experiments cifar.py ../../average_results.py q_one_third_cifar -cifar-
#run_q_experiments svhn.py ../../average_results.py q_one_third_svhn -svhn-
cd ../../

cd ./half/
#run_experiments cifar.py ../average_results.py float_half_cifar -cifar
#run_experiments svhn.py ../average_results.py float_half_svhn -svhn-

cd ./quantised/
#run_q_experiments cifar.py ../../average_results.py q_half_cifar -cifar-
#run_q_experiments svhn.py ../../average_results.py q_half_svhn -svhn-
cd ../../

cd ./two_third/
#run_experiments mnist.py ../average_results.py float_two_third_mnist
#run_experiments cifar.py ../average_results.py float_two_third_cifar
#run_experiments svhn.py ../average_results.py float_two_third_svhn

cd ./quantised/
#run_q_experiments mnist.py ../../average_results.py q_two_third_mnist -mnist-
#run_q_experiments cifar.py ../../average_results.py q_two_third_cifar -cifar-
#run_q_experiments svhn.py ../../average_results.py q_two_third_svhn -svhn-
cd ../../

cd ./all/
#run_experiments mnist.py ../average_results.py float_all_mnist
#run_experiments cifar.py ../average_results.py float_all_cifar
#run_experiments svhn.py ../average_results.py float_all_svhn

cd ./quantised/
#run_q_experiments mnist.py ../../average_results.py q_all_mnist -mnist-
#run_q_experiments cifar.py ../../average_results.py q_all_cifar -cifar-
#run_q_experiments svhn.py ../../average_results.py q_all_svhn -svhn-
cd ../../

