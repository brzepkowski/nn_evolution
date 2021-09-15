#!/bin/bash

N=$1
J=$2
hz=$3
dt=$4
t_max=$5

NAME=`echo "generate_data_${N}_${J}_${hz}_${dt}_${t_max}"`

PBS="#!/bin/bash\n\
#PBS -N ${NAME}\n\
#PBS -l walltime=1:00:00\n\
#PBS -l select=1:ncpus=8:mem=8000MB\n\
#PBS -l software=generate_data.py\n\
#PBS -m n\n\
cd \$PBS_O_WORKDIR\n\
python3 -W ignore generate_data.py ${N} ${J} ${hz} ${dt} ${t_max}"

# Echo the string PBS to the function qsub, which submits it as a cluster job for you
# A small delay is included to avoid overloading the submission process

echo -e ${PBS} | qsub
#echo %{$PBS}
sleep 0.5
echo "done."
