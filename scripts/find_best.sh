#!/bin/bash

# bash find_best.sh

# Finds the best mAP for all the validation data in the grid folder.

cd ../

results=()

for file in grid/*.csv; do
    result=$(python3 scripts/validation_curve.py $file | cut -f 2- -d ' ' | tr '\n' ' ')
    results+=("$file $result\n")
done

best=$(echo -e "${results[*]}" | sort -k3 -n)
echo -e "$best"
echo -e "$best" >> grid_results.txt
