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

# Below variables are hardcoded, in order not to make some silly mistake
# J = 1.0
# h1 = 1.0
# h2 = 0.0

################################################################################
# Load hamiltonian H and generate simple evolution operator
################################################################################

# Load evolution operator U_t_quenched
HQ_filename = 'HQ.dat'
if not path.exists(HQ_filename):
    print("Hamiltonian HQ hasn't been generated yet. Please run the Fortran program 'generate_data_after_quench' with appropriate arguments")
    sys.exit()
else:
    print("Loading hamiltonian HQ... ", end="")
    HQ = load_array_from_file(HQ_filename, 2**N).astype(np.float64)
    print("Done!")

U_t_simple = np.identity(2**N) - (1j * HQ * t)

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
    print("Loading evolution operator... ", end="")
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
    print("Loading eigenvectors of H... ", end="")
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
    print("Loading eigenvalues of H... ", end="")
    eigenvalues_H = list(load_vector_from_file(eigenvals_H_filename, 2**N).astype(np.float64))
    print("Done!")

################################################################################
# Generate training and validation sets
################################################################################
combined_sets_size = 9*np.shape(eigenvectors_H)[0] # We set the fixed size of training + validation sets to be 3 times larger than the number of eigenvectors in H
validation_set_size = floor(combined_sets_size/4)
training_set_size = combined_sets_size - validation_set_size
eigenvectors_number = np.shape(eigenvectors_H)[0]

# We want to include some of the "bare" eigenvectors in the training and validation sets.
# The remaining "bare" eigenvectors will end up in the testing set. We use the following division:
# - trainig set - 1/2 of eigenvectors + linearly combined vectors;
# - validation set - 1/4 of eigenvectors + linearly combined vectors;
# - testing set - 1/4 of eigenvectors.
eigenvectors_H = list(eigenvectors_H)
training_eigenvectors_number = int(eigenvectors_number/2)
validation_eigenvectors_number = int(eigenvectors_number/4)
training_set = []
validation_set = []

for i in range(training_eigenvectors_number): # Pick at random, which eigenvectors should end up in training set
    random_index = random.randint(0, len(eigenvectors_H)-1)
    training_set.append(eigenvectors_H[random_index])
    del eigenvectors_H[random_index]
    del eigenvalues_H[random_index]
for i in range(validation_eigenvectors_number): # Pick at random, which eigenvectors should end up in training set
    random_index = random.randint(0, len(eigenvectors_H)-1)
    validation_set.append(eigenvectors_H[random_index])
    del eigenvectors_H[random_index]
    del eigenvalues_H[random_index]

print("Generating training and validation sets...", end="")
for i in range(training_set_size-training_eigenvectors_number):
    training_set.append(generate_linearly_combined_vector(eigenvectors_H))
# for i in range(validation_set_size-validation_eigenvectors_number):
#     validation_set.append(generate_linearly_combined_vector(eigenvectors_H))

(training_input_vectors, training_output_vectors) = misc.evolve_whole_set_many_evolution_steps_rnn(training_set, U_t_quenched, t, t_total)
(validation_input_vectors, validation_output_vectors) = misc.evolve_whole_set_many_evolution_steps_rnn(validation_set, U_t_quenched, t, t_total)
print("Done!")

################################################################################
# Generate testing set (it consists of evolved eigenvectors of H)
################################################################################

print("Generating testing set...", end="")
# As mentioned earlier, we put 1/4 of "bare" eigenvectors to the testing set:
(test_input_vectors, test_output_vectors) = misc.evolve_whole_set_many_evolution_steps_rnn(eigenvectors_H, U_t_quenched, t, t_total)
print("Done!")

################################################################################
# Generate vectors evolved using U_t_simple
################################################################################

print("Generating testing set...", end="")
# As mentioned earlier, we put 1/4 of "bare" eigenvectors to this set:
(test_input_vectors_simple_op, test_output_vectors_simple_op) = misc.evolve_whole_set_many_evolution_steps_rnn(eigenvectors_H, U_t_simple, t, t_total)
print("Done!")

################################################################################
# Reshape generated sets, to extract imaginary parts of numbers as second dimension in a vector
################################################################################

print("Reshaping sets... ", end="")
(reshaped_training_input_vectors, reshaped_training_output_vectors) = misc.reshape_two_sets(training_input_vectors, training_output_vectors)
(reshaped_validation_input_vectors, reshaped_validation_output_vectors) = misc.reshape_two_sets(validation_input_vectors, validation_output_vectors)
(reshaped_test_input_vectors, reshaped_test_output_vectors) = misc.reshape_two_sets(test_input_vectors, test_output_vectors)
(reshaped_test_input_vectors_simple_op, reshaped_test_output_vectors_simple_op) = misc.reshape_two_sets(test_input_vectors_simple_op, test_output_vectors_simple_op)
print("Done!")

################################################################################
# Flatten the data, so that it can be injected into neural network
################################################################################

print("Flattening sets... ", end="")
reshaped_training_input_vectors = misc.flatten_set(reshaped_training_input_vectors)
reshaped_training_output_vectors = misc.flatten_set(reshaped_training_output_vectors)

reshaped_validation_input_vectors = misc.flatten_set(reshaped_validation_input_vectors)
reshaped_validation_output_vectors = misc.flatten_set(reshaped_validation_output_vectors)

reshaped_test_input_vectors = misc.flatten_set(reshaped_test_input_vectors)
reshaped_test_output_vectors = misc.flatten_set(reshaped_test_output_vectors)

reshaped_test_input_vectors_simple_op = misc.flatten_set(reshaped_test_input_vectors_simple_op)
reshaped_test_output_vectors_simple_op = misc.flatten_set(reshaped_test_output_vectors_simple_op)
print("Done!")

################################################################################
# Generate input to the neural netork, so that it can make predictions, which
# we will be comparing with exact evolution
################################################################################

print("Reshaping and flattening eigenvectors of HQ... ", end="")
# Input for NN consists of remaining eigenvectors of H (not H_Q, because eigvecs of H_Q would not evolve under exp(-i H_Q t))!
nn_input_for_predictions = misc.reshape_set(eigenvectors_H)
nn_input_for_predictions = misc.flatten_set(nn_input_for_predictions)
print("Done!")

################################################################################
# Save all the data generated so far
################################################################################

print("Saving sets... ", end="")
with open('training_input.npy', 'wb') as f:
    np.save(f, reshaped_training_input_vectors)
with open('training_output.npy', 'wb') as f:
    np.save(f, reshaped_training_output_vectors)
with open('validation_input.npy', 'wb') as f:
    np.save(f, reshaped_validation_input_vectors)
with open('validation_output.npy', 'wb') as f:
    np.save(f, reshaped_validation_output_vectors)
with open('test_input.npy', 'wb') as f:
    np.save(f, reshaped_test_input_vectors)
with open('test_output.npy', 'wb') as f:
    np.save(f, reshaped_test_output_vectors)
with open('test_input_simple_op.npy', 'wb') as f:
    np.save(f, reshaped_test_input_vectors_simple_op)
with open('test_output_simple_op.npy', 'wb') as f:
    np.save(f, reshaped_test_output_vectors_simple_op)
with open('nn_input_for_predictions.npy', 'wb') as f:
    np.save(f, nn_input_for_predictions)
with open('test_eigenvalues.npy', 'wb') as f: # Save remaining eigenvalues, because these correspond to vectors in the testing set
    np.save(f, eigenvalues_H)

print("Done!")

################################################################################
# Cast correlation operators to a form understandable for python
################################################################################

print("Casting correlation operators... ", end="")
for N0 in range(N-1):
    for N1 in range(N0+1, N):

        # Load spin correlation operator from two files (ontaining real and imaginary parts separately)
        filename = 'S_corr_' + str(N0) + '_' + str(N1)
        S_corr = combine_two_matrices(filename + '_real', filename + '_imag', 2**N)

        with open(filename + '.npy', 'wb') as f:
            np.save(f, S_corr)
print("Done!")
