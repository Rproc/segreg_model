path_dir=$1
path_local=$2
term=$3
echo $4
echo $path_dir
echo $path_local
echo $term
# path_dir=/mnt/hdd/Linux/test/newoutput/
path_results=results
path_shapes=shapes
path_plots=plots
path_tab=tabelas
path_dict=dictionaries
# loop=$4
# echo $term
var_local="$path_local$term/output/"
var_local="${var_local}"/matrix"*.txt"
# echo $var_local;
ctrl_dir=1
allFiles=()
count=0
for f in $var_local;
do
    FILENAME=`basename ${f%%}`;
    FILENAME2=`basename ${f%%.*}`;
    v=$FILENAME2
    VAR1=${v::-3}
    # echo $VAR1
    # echo $FILENAME2
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
    count=$((count + 1))
    allFiles+=($FILENAME2)
    echo $count
    if [ $count == $4 ]
    then
        # echo $count
        # echo ${allFiles[*]}
        export allFiles
        bash -c 'echo "${allFiles[@]}"'
        # python test.py 2 "${allFiles[@]}"
        /lab/users/renan.procopio/anaconda3/bin/python toCSV_SHP.py $path_dir $path_local $term 250 50 1 1 AGENT_RED AGENT_YELLO AGENT_BLUE AGENT_CYAN "${allFiles[@]}"
        count=0
        allFiles=()
        # pause 'Press [Enter] key to continue...
    fi
done