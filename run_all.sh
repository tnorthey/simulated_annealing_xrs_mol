#!/bin/bash

starting_xyz="xyz/nmm_opt.xyz"
excitation_ratio=0.057
tuning_ratio=0.9

for i in {1..1}
do
    k=0
    for i in {18..48}
    do
        #./run_parallel.sh $i $starting_xyz ${excitation_factors[k]}
        ./run_parallel.sh $i $starting_xyz $excitation_ratio $tuning_ratio
        ((k++))
        sleep 0.1 # For sequential output
        echo "Waiting for processes to finish" 
        wait $(jobs -p)
        echo "All processes finished"
    done

    #for i in {30..43}
    #do
    #    ./run_parallel.sh $i $starting_xyz 0.628
    #    sleep 0.1 # For sequential output
    #    echo "Waiting for processes to finish" 
    #    wait $(jobs -p)
    #    echo "All processes finished"
    #done
done
