#!/bin/bash

SAMPLES=5
ret_val=""
GPU=$1
EVAL_SAMPLES=$2
function run_experiments {
    declare -a samples_list=()
    local pretrained_models=(./5_samples/*"$4"*)
    for i in $(seq 1 $SAMPLES);
    do 
        local index=${i}-1
        local sample=$(python3 $1 --seed $i --gpu $GPU --samples $EVAL_SAMPLES --save ${pretrained_models[$index]} | grep Experiment | cut -d " " -f 4)
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


cd ./ll/
#run_experiments mnist.py ../average_results.py float_ll_mnist -mnist-
#run_experiments cifar.py ../average_results.py float_ll_cifar -cifar-
#run_experiments svhn.py ../average_results.py float_ll_svhn -svhn-

cd ./quantised/
#run_experiments mnist.py ../../average_results.py q_ll_mnist -mnist-
#run_experiments cifar.py ../../average_results.py q_ll_cifar -cifar-
#run_experiments svhn.py ../../average_results.py q_ll_svhn -svhn-
cd ../../

cd ./one_third/
#run_experiments cifar.py ../average_results.py float_one_third_cifar -cifar-
#run_experiments svhn.py ../average_results.py float_one_third_svhn -svhn-

cd ./quantised/
#run_experiments cifar.py ../../average_results.py q_one_third_cifar -cifar-
#run_experiments svhn.py ../../average_results.py q_one_third_svhn -svhn-
cd ../../

cd ./half/
#run_experiments cifar.py ../average_results.py float_half_cifar -cifar-
#run_experiments svhn.py ../average_results.py float_half_svhn -svhn-

cd ./quantised/
#run_experiments cifar.py ../../average_results.py q_half_cifar -cifar-
#run_experiments svhn.py ../../average_results.py q_half_svhn -svhn-
cd ../../

cd ./two_third/
#run_experiments mnist.py ../average_results.py float_two_third_mnist -mnist-
#run_experiments cifar.py ../average_results.py float_two_third_cifar -cifar-
#run_experiments svhn.py ../average_results.py float_two_third_svhn -svhn-

cd ./quantised/
#run_experiments mnist.py ../../average_results.py q_two_third_mnist -mnist-
#run_experiments cifar.py ../../average_results.py q_two_third_cifar -cifar-
#run_experiments svhn.py ../../average_results.py q_two_third_svhn -svhn-
cd ../../

cd ./all/
#run_experiments mnist.py ../average_results.py float_all_mnist -mnist-
#run_experiments cifar.py ../average_results.py float_all_cifar -cifar-
#run_experiments svhn.py ../average_results.py float_all_svhn -svhn-

cd ./quantised/
#run_experiments mnist.py ../../average_results.py q_all_mnist -mnist-
#run_experiments cifar.py ../../average_results.py q_all_cifar -cifar-
#run_experiments svhn.py ../../average_results.py q_all_svhn -svhn-
cd ../../
