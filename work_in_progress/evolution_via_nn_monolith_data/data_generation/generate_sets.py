import misc
from misc import load_array_from_file, load_vector_from_file, generate_random_vector, generate_linearly_combined_vector
import numpy as np
from scipy.linalg import expm
from math import floor
import os.path
from os import path
import sys
import time
import random
import copy

def combine_two_matrices(filename_real, filename_imag, size):
    real = load_array_from_file(filename_real, size)
    imag = load_array_from_file(filename_imag, size)
    combined = np.zeros((size,size), dtype=complex)
    for i in range(size):
        for j in range(size):
            combined[i,j] = complex(real[i,j], imag[i,j])
    return combined

if len(sys.argv) != 4:
    print("Wrong number of arguments provided. Please give them in the following fashion:")
    print("- N,")
    print("- timestep,")
    print("- total_evolution_time.")
    sys.exit()

N = int(sys.argv[1])
t = float(sys.argv[2])
t_total = float(sys.argv[3])
COMBINED_SETS_MULTIPLIER = 15

print("#"*40)
print("# Launching generate_sets.py")
print("#"*40)

################################################################################
# Load evolution operator
################################################################################

# Load evolution operator U_t_quenched
U_real_filename = 'U_real.dat'
U_imag_filename = 'U_imag.dat'
if not (path.exists(U_real_filename) and path.exists(U_imag_filename)):
    print("Evolution operator U_quenched hasn't been generated yet. Please run the Fortran program 'generate_evolution_operator' with appropriate arguments")
    sys.exit()
else:
    print("Loading evolution operator... ", end="", flush=True)
    U_t_quenched = combine_two_matrices('U_real.dat', 'U_imag.dat', 2**N)
    print("Done!")

################################################################################
# Load eigenvectors of H
################################################################################

eigenvecs_H_filename = 'eigenvectors_H.dat'
if not path.exists(eigenvecs_H_filename):
    print("Hamilotnian H hasn't been diagonalized yet. Please run the Fortran program 'generate_data_before_quench' with appropriate arguments")
    sys.exit()
else:
    print("Loading eigenvectors of H... ", end="", flush=True)
    eigenvectors_H = load_array_from_file(eigenvecs_H_filename, 2**N).astype(np.complex128) # Here each eigenvector is represented by a column in a matrix
    eigenvectors_H = eigenvectors_H.T # Transposition on an ndarray is "for free" in O(1)
    print("Done!")

################################################################################
# Load eigenvalues of H
################################################################################

eigenvals_H_filename = 'eigenvalues_H.dat'
if not path.exists(eigenvals_H_filename):
    print("Hamilotnian H hasn't been diagonalized yet. Please run the Fortran program 'generate_data_before_quench' with appropriate arguments")
    sys.exit()
else:
    print("Loading eigenvalues of H... ", end="", flush=True)
    test_eigenvalues = list(load_vector_from_file(eigenvals_H_filename, 2**N).astype(np.float64))
    print("Done!")

print("#"*40)

################################################################################
# Prepare initial training, validation and testing sets
################################################################################
combined_sets_size = COMBINED_SETS_MULTIPLIER*np.shape(eigenvectors_H)[0] # We set the fixed size of training + validation sets to be COMBINED_SETS_MULTIPLIER times larger than the number of eigenvectors in H
validation_set_size = floor(combined_sets_size/4)
training_set_size = combined_sets_size - validation_set_size
eigenvectors_number = np.shape(eigenvectors_H)[0]

# We want to include some of the "bare" eigenvectors in the training and validation sets.
# The remaining "bare" eigenvectors will end up in the testing set. We use the following division:
# - trainig set - 1/2 of eigenvectors + linearly combined vectors;
# - validation set - 1/4 of eigenvectors + linearly combined vectors;
# - testing set - 1/4 of eigenvectors.
testing_set = list(copy.deepcopy(eigenvectors_H))
training_eigenvectors_number = int(eigenvectors_number/2)
validation_eigenvectors_number = int(eigenvectors_number/4)
training_set = []
validation_set = []

for i in range(training_eigenvectors_number): # Pick at random, which eigenvectors should end up in training set
    random_index = random.randint(0, len(testing_set)-1)
    training_set.append(testing_set[random_index])
    del testing_set[random_index]
    del test_eigenvalues[random_index]
for i in range(validation_eigenvectors_number): # Pick at random, which eigenvectors should end up in training set
    random_index = random.randint(0, len(testing_set)-1)
    validation_set.append(testing_set[random_index])
    del testing_set[random_index]
    del test_eigenvalues[random_index]

################################################################################
# Evolve, save and delete TRAINING set
################################################################################

print("Generating training set...", end="", flush=True)
for i in range(training_set_size-training_eigenvectors_number):
    training_set.append(generate_linearly_combined_vector(eigenvectors_H))
print("Done!")

print("Evolving training set...", end="", flush=True)
(training_input_vectors, training_output_vectors) = misc.evolve_whole_set_many_evolution_steps(training_set, U_t_quenched, t, t_total)
print("Done!")

del training_set

print("Reshaping training_input set... ", end="", flush=True)
reshaped_training_input_vectors = misc.reshape_set(training_input_vectors)
print("Done!")
del training_input_vectors
print("Flattening training_input set... ", end="", flush=True)
reshaped_training_input_vectors = misc.flatten_set(reshaped_training_input_vectors)
print("Done!")
print("Saving training_input set... ", end="", flush=True)
with open('training_input.npy', 'wb') as f:
    np.save(f, reshaped_training_input_vectors)
print("Done!")
del reshaped_training_input_vectors


print("Reshaping training_output set... ", end="", flush=True)
reshaped_training_output_vectors = misc.reshape_set(training_output_vectors)
print("Done!")
del training_output_vectors
print("Flattening training_output set... ", end="", flush=True)
reshaped_training_output_vectors = misc.flatten_set(reshaped_training_output_vectors)
print("Done!")
print("Saving training_output set... ", end="", flush=True)
with open('training_output.npy', 'wb') as f:
    np.save(f, reshaped_training_output_vectors)
print("Done!")

del reshaped_training_output_vectors

print("#"*40)

################################################################################
# Evolve, save and delete VALIDATION set
################################################################################

print("Generating validation set...", end="", flush=True)
for i in range(validation_set_size-validation_eigenvectors_number):
    validation_set.append(generate_linearly_combined_vector(eigenvectors_H))
print("Done!")

del eigenvectors_H # We used it for the last time

print("Evolving validation set...", end="", flush=True)
(validation_input_vectors, validation_output_vectors) = misc.evolve_whole_set_many_evolution_steps(validation_set, U_t_quenched, t, t_total)
print("Done!")

del validation_set

print("Reshaping validation_input set... ", end="", flush=True)
reshaped_validation_input_vectors = misc.reshape_set(validation_input_vectors)
print("Done!")
del validation_input_vectors
print("Flattening validation_input set... ", end="", flush=True)
reshaped_validation_input_vectors = misc.flatten_set(reshaped_validation_input_vectors)
print("Done!")
print("Saving validation_input set... ", end="", flush=True)
with open('validation_input.npy', 'wb') as f:
    np.save(f, reshaped_validation_input_vectors)
print("Done!")
del reshaped_validation_input_vectors


print("Reshaping validation_output set... ", end="", flush=True)
reshaped_validation_output_vectors = misc.reshape_set(validation_output_vectors)
print("Done!")
del validation_output_vectors
print("Flattening validation_output set... ", end="", flush=True)
reshaped_validation_output_vectors = misc.flatten_set(reshaped_validation_output_vectors)
print("Done!")
print("Saving validation_output set... ", end="", flush=True)
with open('validation_output.npy', 'wb') as f:
    np.save(f, reshaped_validation_output_vectors)
print("Done!")
del reshaped_validation_output_vectors

print("#"*40)

################################################################################
# Evolve and save TESTING set (it consists of evolved eigenvectors of H)
################################################################################

print("Evolving testing set...", end="", flush=True)
# As mentioned earlier, we put 1/4 of "bare" eigenvectors to the testing set:
(test_input_vectors, test_output_vectors) = misc.evolve_whole_set_many_evolution_steps(testing_set, U_t_quenched, t, t_total)
print("Done!")

del U_t_quenched # We won't be using it anymore

# WARNING: In this case we are not deleting the "testing_set", because it will be used later during
# generation of set evolved using simple evolution operator U_t_simple and generation of input for NN

print("Reshaping testing_input set... ", end="", flush=True)
reshaped_test_input_vectors = misc.reshape_set(test_input_vectors)
print("Done!")
del test_input_vectors
print("Flattening testing_input set... ", end="", flush=True)
reshaped_test_input_vectors = misc.flatten_set(reshaped_test_input_vectors)
print("Done!")
print("Saving testing_input set... ", end="", flush=True)
with open('test_input.npy', 'wb') as f:
    np.save(f, reshaped_test_input_vectors)
print("Done!")
del reshaped_test_input_vectors


print("Reshaping testing_output set... ", end="", flush=True)
reshaped_test_output_vectors = misc.reshape_set(test_output_vectors)
print("Done!")
del test_output_vectors
print("Flattening testing_output set... ", end="", flush=True)
reshaped_test_output_vectors = misc.flatten_set(reshaped_test_output_vectors)
print("Done!")
print("Saving testing_output set... ", end="", flush=True)
with open('test_output.npy', 'wb') as f:
    np.save(f, reshaped_test_output_vectors)
print("Done!")
del reshaped_test_output_vectors

print("Saving testing eigenvalues... ", end="", flush=True)
with open('test_eigenvalues.npy', 'wb') as f: # Save remaining eigenvalues, because these correspond to vectors in the testing set
    np.save(f, test_eigenvalues)
print("Done!")
del test_eigenvalues

print("#"*40)

################################################################################
# Reshape and save INPUT FOR THE NN, so that it (NN) can make predictions, which
# we will be comparing with exact evolution
################################################################################

print("Reshaping and flattening input for NN... ", end="", flush=True)
# Input for NN consists of remaining eigenvectors of H (not H_Q, because eigvecs of H_Q would not evolve under exp(-i H_Q t))!
nn_input_for_predictions = misc.reshape_set(testing_set, remove_input_matrix=False)
nn_input_for_predictions = misc.flatten_set(nn_input_for_predictions)
print("Done!")

print("Saving input for NN... ", end="", flush=True)
with open('nn_input_for_predictions.npy', 'wb') as f:
    np.save(f, nn_input_for_predictions)
print("Done!")

del nn_input_for_predictions

print("#"*40)

################################################################################
# Load HQ Hamiltonian, to create simple evolution operator
################################################################################

HQ_filename = 'HQ.dat'
if not path.exists(HQ_filename):
    print("Hamilotnian HQ hasn't been created yet. Please run the Fortran program 'generate_data_after_quench' with appropriate arguments")
    sys.exit()
else:
    print("Loading HQ hamiltonian... ", end="", flush=True)
    HQ = load_array_from_file(HQ_filename, 2**N).astype(np.float64)
    print("Done!")

U_t_simple = np.identity(2**N) - (1j*t*HQ)

################################################################################
# Generate, evolve and save TESTING set using SIMPLE EVOLUTION OPERATOR
################################################################################

print("Evolving testing_simple set...", end="", flush=True)
# As mentioned earlier, we put 1/4 of "bare" eigenvectors to the testing set:
(test_input_vectors_simple_op, test_output_vectors_simple_op) = misc.evolve_whole_set_many_evolution_steps(testing_set, U_t_simple, t, t_total)
print("Done!")

del testing_set
del U_t_simple

print("Reshaping testing_simple_input set... ", end="", flush=True)
reshaped_test_input_vectors_simple_op = misc.reshape_set(test_input_vectors_simple_op)
print("Done!")
del test_input_vectors_simple_op
print("Flattening testing_simple_input set... ", end="", flush=True)
reshaped_test_input_vectors_simple_op = misc.flatten_set(reshaped_test_input_vectors_simple_op)
print("Done!")
print("Saving testing_simple_input set... ", end="", flush=True)
with open('test_input_simple_op.npy', 'wb') as f:
    np.save(f, reshaped_test_input_vectors_simple_op)
print("Done!")
del reshaped_test_input_vectors_simple_op


print("Reshaping testing_simple_output set... ", end="", flush=True)
reshaped_test_output_vectors_simple_op = misc.reshape_set(test_output_vectors_simple_op)
print("Done!")
del test_output_vectors_simple_op
print("Flattening testing_simple_output set... ", end="", flush=True)
reshaped_test_output_vectors_simple_op = misc.flatten_set(reshaped_test_output_vectors_simple_op)
print("Done!")
print("Saving testing_simple_output set... ", end="", flush=True)
with open('test_output_simple_op.npy', 'wb') as f:
    np.save(f, reshaped_test_output_vectors_simple_op)
print("Done!")
del reshaped_test_output_vectors_simple_op


print("#"*40)

################################################################################
# Cast correlation operators to a form understandable for python
################################################################################

print("Casting correlation operators... ", end="", flush=True)
for N0 in range(N-1):
    for N1 in range(N0+1, N):
        # Load spin correlation operator from two files (ontaining real and imaginary parts separately)
        filename = 'S_corr_' + str(N0) + '_' + str(N1)
        S_corr = combine_two_matrices(filename + '_real', filename + '_imag', 2**N)

        with open(filename + '.npy', 'wb') as f:
            np.save(f, S_corr)
        del S_corr
print("Done!")
