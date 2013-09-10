#!/bin/sh

for i in `cat diseases.txt`;
do 
  echo $i; 
  for j in passive active bronze bronze_weighting random_weighting;
  do
    #echo "Looking at file: outputs/$1/"$i"_"$j".txt"
    echo "\t$j\t\t" `perl scripts/getALC.pl outputs/$1/$i"_"$j.txt 25`
  done
done

