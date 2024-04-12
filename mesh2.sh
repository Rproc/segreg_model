#!/bin/bash

function pause(){
	read -p "$*"
}
path_dir=$1
path_local=$2;
term=$3;
count=0
allFiles=()
ctrl_dir=1
termino=*.txt
# files=(base 10-50-40 30-50-20 consolidation=100 decay=100 density=2 density=8 pred=4 pred=6 pred=8 predrandom=50 predrandom=100 steps=4 steps=8 steps2=4 steps2=8 steps3=4 steps3=8)
path_results=results
path_shapes=shapes
path_plots=plots
path_tab=tabelas
path_dict=dictionaries

var_local="$path_local$term/output/";
var_local="${var_local}"/matrix"*.txt";
# echo $var_local;
ctrl_dir=1
allFiles=()


# echo $var_local
# echo $term
# echo $4
# $var_local=$var_local$termino
for f in $var_local; do \
    FILENAME=`basename ${f%%}`;
    FILENAME2=`basename ${f%%.*}`;
    v=$FILENAME
    VAR1=${v::-4}
    # echo $VAR1
    # echo $VAR1
    # pause
    if [ $ctrl_dir == 1 ]
    then
    	mkdir $path_dir/$term
        mkdir $path_dir/$term/$path_shapes
        mkdir $path_dir/$term/$path_results
        mkdir $path_dir/$term/$path_plots
        mkdir $path_dir/$term/$path_results/$path_tab
        mkdir $path_dir/$term/$path_results/$path_dict
    	ctrl_dir=0
    fi
    allFiles+=($VAR1)
    # echo $count
    if [ $count == $4 ]
    then
        # echo $count
		# echo ${allFiles[*]}
        export allFiles
        bash -c 'echo "${allFiles[@]}"'
        # python test.py 2 "${allFiles[@]}"
        /lab/users/renan.procopio/anaconda3/bin/python toCSV_SHP.py $path_dir $path_local $term 250 50 1 1 AGENT_RED AGENT_YELLO AGENT_BLUE AGENT_CYAN "${allFiles[@]}";
        count=0
        allFiles=()
        # pause
        # pause 'Press [Enter] key to continue...'

    fi
    count=$((count + 1))


done

