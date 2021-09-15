#!/bin/bash

N=$1
J=$2
hz=$3
dt=$4
t_max=$5
number_of_datasets=$6

NAME=`echo "generate_data_SD_${N}_${J}_${hz}_${dt}_${t_max}_${number_of_datasets}"`

PBS="#!/bin/bash\n\
#PBS -N ${NAME}\n\
#PBS -l walltime=168:00:00\n\
#PBS -l select=1:ncpus=8:mem=64000MB\n\
#PBS -l software=generate_data_SD.py\n\
#PBS -m n\n\
cd \$PBS_O_WORKDIR\n\
python3 -W ignore generate_data_SD.py ${N} ${J} ${hz} ${dt} ${t_max} ${number_of_datasets}"

# Echo the string PBS to the function qsub, which submits it as a cluster job for you
# A small delay is included to avoid overloading the submission process

echo -e ${PBS} | qsub
#echo %{$PBS}
sleep 0.5
echo "done."
