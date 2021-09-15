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

if len(sys.argv) != 2:
    print("Wrong number of arguments provided. Please give them in the following fashion:")
    print("- N.")
    sys.exit()

N = int(sys.argv[1])

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
