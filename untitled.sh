#!/bin/bash
function pause(){
	read -p "$*"
}
count=0
sum=1
allFiles=()
loop=$(($1-1))
ctrl_dir=1
path_local=$PWD
termino=*.txt
path_dir=$PWD/output/
path_results=results
j=0
# value=(2 4 6 8)
# valuepred=(8 7 6 5 4)
# files=(density=2 density=4 density=6 density=8)
# f2=(steps=2 steps=4 steps=6 steps=8)
# f3=(pred=8 pred=7 pred=6 pred=5 pred=4)

value=(2)
valuepred=(8 7)
files=(density=2)
f2=(steps=2)
f3=(pred=8 pred=7)

echo $path_local
ind=0
for i in "${files[@]}"; do
	ind2=0
	for j in "${f2[@]}"; do
		ind3=0
		for k in "${f3[@]}"; do
		    term=$i$j$k
		    # echo $term
		    var_local="$path_local/$i$j$k";
		    var2="$path_dir$i$j$k"
		    echo $var2;
		    # ctrl_dir=1
		    # allFiles+=($term)
		    count=0
		    # mkdir $var2
		    # mkdir $var2/output
		    # cp model_6.jar $var2
		    # cp old.xml $var2
		    cd $var2
		    for count in $(seq 0 $loop); do
				# cd $path_local
				name=$term"_"$count
		  #   	echo $name
				# /lab/users/renan.procopio/usr/bin/xmlstarlet ed -P -u '//Repast:Param[@name="D"]/@value' -v "${value[ind]}"  \
				# -u '//Repast:Param[@name="FindspaceSteps"]/@value' -v "${value[ind2]}" \
				# -u '//Repast:Param[@name="Pred"]/@value' -v "${valuepred[ind3]}" \
				# -u '//Repast:Param[@name="Log"]/@value' -v $count \
				# <old.xml >$name.xml
				# cd $var2
				# echo $var2
				qsub -V -cwd -shell n runJava.sh $name
				# pause
			done
			# cd $var2
			# rm model_02_02.jar
			cd $path_local
			ind3=$((ind3+sum))
		done
		ind2=$((ind2+sum))
	done
	ind=$((ind+sum))

done

# printf "%s\n" "${allFiles[@]}" > array.txt

