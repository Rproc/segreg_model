#!/bin/bash

function pause(){
	read -p "$*"
}
sum=0
count=0
allFiles=()
loop=$(($1-1))
ctrl_dir=1
# echo $loop
path_local=/lab/users/renan.procopio/experimentos/output/
termino=*.txt
# files=(base 10-50-40 30-50-20 consolidation=100 decay=100 density=2 density=8 pred=4 pred=6 pred=8 predrandom=50 predrandom=100 steps=4 steps=8 steps2=4 steps2=8 steps3=4 steps3=8)
readarray -t files < array.txt
path_dir=/lab/users/renan.procopio/experimentos/newoutput/
path_results=results
path_shapes=shapes
path_plots=plots
path_tab=tabelas
path_dict=dictionaries
for i in "${files[@]}"; do

	qsub -v -cwd -shell n mesh2.sh $path_dir $path_local $i $loop
	echo $i
	# pause
done

