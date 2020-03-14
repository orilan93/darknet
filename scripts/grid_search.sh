#!/bin/bash

# bash grid_search.sh NAME
# NAME: The name of the trained network configuration.

# Runs a grid search on the different hyperparameters specified bellow
# on a specific network configuration.
# Note that the data, and cfg file needs to be located in the data folder
# and have the same name as the weights folder. This should be the f.ex. 'yolov3_tree'.

: ${1?Missing a required argument.}

FOLDER=${1}

thresh_list=(0.4 0.5)
hier_list=(0.2 0.5 0.7 0.9)
nms_list=(0.6 0.45)

for t in ${thresh_list[*]}; do
    for h in ${hier_list[*]}; do
        for n in ${nms_list[*]}; do
            ./validate_weights.sh $FOLDER $t $h $n
        done
    done
done