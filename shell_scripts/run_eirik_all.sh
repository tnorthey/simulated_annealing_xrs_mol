#!/bin/bash

excitation_factors=(0.140 0.471 0.492 0.515 0.538 0.559 0.577 0.592 0.604 0.612 0.618 0.622 0.625)

path="eirik_data/sampled_frames"

for starting_xyz in "$path/00014_frame340.xyz" "$path/00068_frame962.xyz" "$path/00078_frame614.xyz" "$path/00086_frame484.xyz" "$path/00138_frame022.xyz" "$path/00164_frame298.xyz" "$path/00243_frame337.xyz" "$path/00248_frame432.xyz" "$path/00251_frame526.xyz" "$path/00252_frame395.xyz"
do
    k=0
    for i in {17..29}
    do
        ./run_parallel.sh $i $starting_xyz ${excitation_factors[k]}
        ((k++))
        sleep 0.1 # For sequential output
        echo "Waiting for processes to finish" 
        wait $(jobs -p)
        echo "All processes finished"
    done

    for i in {30..99}
    do
        ./run_parallel.sh $i $starting_xyz 0.628
        sleep 0.1 # For sequential output
        echo "Waiting for processes to finish" 
        wait $(jobs -p)
        echo "All processes finished"
    done
done

