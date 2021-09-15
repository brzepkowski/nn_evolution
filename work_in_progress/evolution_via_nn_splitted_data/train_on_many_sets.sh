#!/bin/bash

activation=$1
N=$2
timestep=$3
t_total=$4
total_number_of_sets=$5

cd "data"
dir_name="N=${N}_t=${timestep}_t_total=${t_total}"
cd "$dir_name"

for (( set_number=0; set_number<$total_number_of_sets ; set_number++ ))
do
  echo "#######################################################################"
  echo "# TRAINING ON SET: $set_number"
  echo "#######################################################################"
  python3 ../../neural_networks/train.py $activation $N $timestep $t_total $set_number
done
