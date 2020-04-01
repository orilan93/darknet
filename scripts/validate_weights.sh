#!/bin/bash

# bash validate_weights.sh NAME [THRESH] [HIER] [NMS]
# NAME: The name of the trained network configuration.
# THRESH: The probability threshold to classify as a detected object. (default: 0.5)
# HIER: The probability threshold of a tree node before favoring the parent. (default: 0.5)
# NMS: The non maximum supression IoU threshold. (default: 0.5)

# Calculates the mAP for each wheight for the specified network configuration.
# Note that the data, and cfg file needs to be located in the data folder
# and have the same name as the weights folder. This should be the f.ex. 'yolov3_tree'.

: ${1?Missing a required argument.}

FOLDER=${1}
THRESH=${2:-0.5}
HIER=${3:-0.5}
NMS=${4:-0.5}

cd ../

files=$(ls -1 $1 | sort -V | grep -v backup)

for file in $files; do
    echo $file
done

csv="${FOLDER}_thresh_${THRESH}_hier_${HIER}_nms_${NMS}.csv"
echo "epoch,precision,recall,avg_offset,metric,result" >> grid/$csv

for file in $files; do
    echo $file
    rm results/*
    ./darknet detector valid data/$FOLDER.data data/$FOLDER.cfg $FOLDER/$file -thresh $THRESH -hier $HIER -nms $NMS
    result=$(python3 scripts/calc_validation_darknet.py data/fish_taxonomy.xml results data/test.manifest 1920x1080 data/fish.names | grep -E "precision|recall|avg_offset|metric|result" | cut -f 2- -d ' ' | tr '\n' ' ')
    read prec rec avg_offset metric res <<< $result
    result="$prec,$rec,$avg_offset,$metric,$res"
    echo "$file,$result" >> grid/$csv
done

