#!/bin/bash

N=$1
timestep=$2
t_total=$3
total_number_of_sets=$4

# Go to the "data_generation" catalog
cd data_generation

# Clean working space
rm *.o

# Compile all Fortran programs
source /opt/intel/parallel_studio_xe_2020.0.088/psxevars.sh
./compil.sh generate_data_before_quench
./compil.sh generate_data_after_quench
# make # This compiles the "generate_evolution_operator.f90" file
make # This compiles the "generate_sets.f90" file
gfortran generate_correlation_operators.f90 -o generate_correlation_operators

# Come back to the main directory
cd ..

# If "data" catalog hasn't been created yet, do it
mkdir "data"
cd "data"

# Create directory, in which we will store all the data generated
dir_name="N=${N}_t=${timestep}_t_total=${t_total}"
mkdir "$dir_name"
cd "$dir_name"

# Diagonalize Hamiltonians H and HQ. Also generate evolution operator U_quenched
echo "Generating H and its eigenvectors..."
./../../data_generation/generate_data_before_quench $N
echo "Done!"
echo "Generating HQ and its eigenvectors..."
./../../data_generation/generate_data_after_quench $N
echo "Done!"
echo "Generating target training, validation and testing sets ..."
for (( set_number=0; set_number<$total_number_of_sets ; set_number++ ))
do
./../../data_generation/generate_sets $N $timestep $t_total $set_number
done
echo "Done!"
echo "Generating correlation operators..."
./../../data_generation/generate_correlation_operators $N
echo "Done!"

# Generate all data necessary for training and testing neural networks
python3 ../../data_generation/cast_correlation_operators.py $N
