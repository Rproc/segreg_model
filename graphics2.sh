#!/bin/bash

allFiles=()
path_local=/lab/users/renan.procopio/experimentos/newoutput/
termino=*.csv
# files=(base 10-50-40)
readarray -t files < array.txt
# files=(base 10-50-40 30-50-20 consolidation=100 decay=100 density=2 density=8 pred=4 pred=6 pred=8 predrandom=50 predrandom=100 steps=4 steps=8 steps2=4 steps2=8 steps3=4 steps3=8)
path_dir=/lab/users/renan.procopio/experimentos/newoutput/
mkdir /lab/users/renan.procopio/experimentos/newoutput/plots/
path_results=results
path_shapes=shapes
path_plots=plots
path_tab=tabelas
path_dict=dictionaries
for i in "${files[@]}"; do
    term=$i
    var_local="$path_local";
    allFiles+=($term)
done

/lab/users/renan.procopio/anaconda3/bin/python plot.py $var_local $i "${allFiles[@]}"
