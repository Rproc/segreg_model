#!/bin/bash

allFiles=()
path_local=/mnt/hdd/Linux/output/
termino=*.csv
# files=(base 10-50-40)
files=(pred=4 pred=6 pred=8 predrandom=50 predrandom=100)
path_dir=/mnt/hdd/Linux/output/
path_results=results
path_shapes=shapes
path_plots=plots
path_tab=tabelas
path_dict=dictionaries
for i in "${files[@]}"; do
    term=$i
    var_local="$path_local";
    /lab/users/renan.procopio/anaconda3/bin/python test.py $var_local $i

done
