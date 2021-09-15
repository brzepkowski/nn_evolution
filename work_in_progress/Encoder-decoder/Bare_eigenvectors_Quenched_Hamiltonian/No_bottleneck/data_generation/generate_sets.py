import misc
from misc import load_array_from_file
import numpy as np
# from numpy.linalg import eig
from scipy.linalg import expm
# from itertools import combinations
from math import floor
import os.path
from os import path
import sys
# import struct
import time
# For testing
# from scipy.linalg import expm # It might be changed to "from scipy.sparse.linalg import expm"
# from numpy.linalg import eig

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
# Load evolution operator
################################################################################

# Load evolution operator U_t_quenched
U_real_filename = 'U_real.dat'
U_imag_filename = 'U_imag.dat'
if not (path.exists(U_real_filename) and path.exists(U_imag_filename)):
    print("Evolution operator U_quenched hasn't been generated yet. Please run the Fortran program 'generate_evolution_operator' with appropriate arguments")
    sys.exit()
else:
    U_t_quenched = combine_two_matrices('U_real.dat', 'U_imag.dat', 2**N)
    # print("[FROM FORTRAN] U_t_quenched: ")
    # print(U_t_quenched)

################################################################################
# Load eigenvectors of H; generate the training and validation sets
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

print("Evolving set of eigenvectors of H... ", end="")
(input_eigenvectors_H, output_eigenvectors_H) = misc.evolve_whole_set_many_evolution_steps(eigenvectors_H, U_t_quenched, t, t_total)
eigenvectors_H_train_set_size = floor(np.shape(input_eigenvectors_H)[0]*3/4)

training_input_vectors = input_eigenvectors_H[:eigenvectors_H_train_set_size]
training_output_vectors = output_eigenvectors_H[:eigenvectors_H_train_set_size]

validation_input_vectors = input_eigenvectors_H[eigenvectors_H_train_set_size:]
validation_output_vectors = output_eigenvectors_H[eigenvectors_H_train_set_size:]
print("Done!")

################################################################################
# Load eigenvectors of H_quenched; generate the testing set
################################################################################

eigenvecs_HQ_filename = 'eigenvectors_HQ.dat'
if not path.exists(eigenvecs_HQ_filename):
    print("Hamilotnian H_quenched hasn't been diagonalized yet. Please run the Fortran program 'generate_data_after_quench' with appropriate arguments")
    sys.exit()
else:
    print("Loading eigenvectors of HQ...", end="")
    eigenvectors_HQ = load_array_from_file(eigenvecs_HQ_filename, 2**N).astype(np.complex128) # Here each eigenvector is represented by a column in a matrix
    eigenvectors_HQ = eigenvectors_HQ.T # Transposition on an ndarray is "for free" in O(1)
    print("Done!")

print("Evolving set of eigenvectors of HQ... ", end="")
(input_eigenvectors_HQ, output_eigenvectors_HQ) = misc.evolve_whole_set_many_evolution_steps(eigenvectors_HQ, U_t_quenched, t, t_total)
# print("input_eigenvectors_HQ: ")
# print(input_eigenvectors_HQ)
# print("output_eigenvectors_HQ: ")
# print(output_eigenvectors_HQ)
eigenvectors_HQ_train_set_size = floor(np.shape(input_eigenvectors_HQ)[0]*3/4)

test_input_vectors = input_eigenvectors_HQ
test_output_vectors = output_eigenvectors_HQ
print("Done!")

################################################################################
# Reshape generated sets, to extract imaginary parts of numbers as second dimension in a vector
################################################################################

print("Reshaping sets... ", end="")
(reshaped_training_input_vectors, reshaped_training_output_vectors) = misc.reshape_two_sets(training_input_vectors, training_output_vectors)
(reshaped_validation_input_vectors, reshaped_validation_output_vectors) = misc.reshape_two_sets(validation_input_vectors, validation_output_vectors)
(reshaped_test_input_vectors, reshaped_test_output_vectors) = misc.reshape_two_sets(test_input_vectors, test_output_vectors)
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
print("Done!")

################################################################################
# Generate input to the neural netork, so that it can make predictions, which
# we will be comparing with exact evolution
################################################################################

print("Reshaping and flattening eigenvectors of HQ... ", end="")
nn_input_for_predictions = misc.flatten_set(misc.reshape_set(eigenvectors_HQ))
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
with open('nn_input_for_predictions.npy', 'wb') as f:
    np.save(f, nn_input_for_predictions)
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
