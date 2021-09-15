#!/bin/bash

N=$1
dt=$2
t_max=$3

NAME=`echo "tn_evol_${N}_${dt}_${t_max}"`

PBS="#!/bin/bash\n\
#PBS -N ${NAME}\n\
#PBS -l walltime=1:00:00\n\
#PBS -l select=1:ncpus=8:mem=8000MB\n\
#PBS -l software=test_tn_evolution.py\n\
#PBS -m n\n\
cd \$PBS_O_WORKDIR\n\
python3 -W ignore test_tn_evolution.py ${N} ${dt} ${t_max}"

# Echo the string PBS to the function qsub, which submits it as a cluster job for you
# A small delay is included to avoid overloading the submission process

echo -e ${PBS} | qsub
#echo %{$PBS}
sleep 0.5
echo "done."

#######################################################

NAME=`echo "tn_input_${N}_${dt}_${t_max}"`

PBS="#!/bin/bash\n\
#PBS -N ${NAME}\n\
#PBS -l walltime=1:00:00\n\
#PBS -l select=1:ncpus=8:mem=8000MB\n\
#PBS -l software=test_tn_input_recreation.py\n\
#PBS -m n\n\
cd \$PBS_O_WORKDIR\n\
python3 -W ignore test_tn_input_recreation.py ${N} ${dt} ${t_max}"

# Echo the string PBS to the function qsub, which submits it as a cluster job for you
# A small delay is included to avoid overloading the submission process

echo -e ${PBS} | qsub
#echo %{$PBS}
sleep 0.5
echo "done."
