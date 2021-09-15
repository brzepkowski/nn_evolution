import numpy as np
from numpy.random import rand
from numpy.linalg import eig, norm
from cmath import sqrt
from scipy.linalg import expm
import sys

Sx = [[0 + 0j, 1 + 0j], [1 + 0j,  0 + 0j]]
Sy = [[0 + 0j, 0 - 1j], [0 + 1j,  0 + 0j]]
Sz = [[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]]

def generate_random_vector(dims):
    vector = np.random.random(dims) + np.random.random(dims) * 1j
    magnitude = sum(x*x.conjugate() for x in vector) ** .5
    return np.array([x/magnitude for x in vector])

def generate_linearly_combined_vector(eigenvectors):
    lin_comb_vector = np.zeros(np.shape(eigenvectors)[1], dtype='complex128')
    # print("[B]: ", lin_comb_vector)
    for vector in eigenvectors:
        lin_comb_vector += rand(1) * vector
    # print("[A]: ", lin_comb_vector)
    # Normalize the vector
    # print("[B] Norm: ", norm(lin_comb_vector))
    lin_comb_vector /= norm(lin_comb_vector)
    # print("[A] Norm: ", norm(lin_comb_vector))
    return lin_comb_vector



def generate_Heisenberg_hamiltonian(N, J, h=0, periodic=False):
    H = np.zeros((2**N, 2**N), dtype='complex64')
    for i in range(N-1):
        # First particle
        left_size = 2**i
        right_size = int((2**N)/(2**(i+1)))
        S_x_1 = np.kron(np.kron(np.identity(left_size), Sx), np.identity(right_size))
        S_y_1 = np.kron(np.kron(np.identity(left_size), Sy), np.identity(right_size))
        S_z_1 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))

        # Second particle
        left_size = 2**(i+1)
        right_size = int((2**N)/(2**(i+2)))
        S_x_2 = np.kron(np.kron(np.identity(left_size), Sx), np.identity(right_size))
        S_y_2 = np.kron(np.kron(np.identity(left_size), Sy), np.identity(right_size))
        S_z_2 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))

        H += -J*(np.dot(S_x_1,S_x_2) + np.dot(S_y_1,S_y_2) + np.dot(S_z_1,S_z_2))
        H += -h*(S_x_1 + S_y_1 + S_z_1)

    if periodic:
        # First particle
        left_size = 1
        right_size = int(2**(N-1))
        S_x_1 = np.kron(np.kron(np.identity(left_size), Sx), np.identity(right_size))
        S_y_1 = np.kron(np.kron(np.identity(left_size), Sy), np.identity(right_size))
        S_z_1 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))

        # Last particle
        left_size = int(2**(N-1))
        right_size = 1
        S_x_2 = np.kron(np.kron(np.identity(left_size), Sx), np.identity(right_size))
        S_y_2 = np.kron(np.kron(np.identity(left_size), Sy), np.identity(right_size))
        S_z_2 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))

        H += -J*(np.dot(S_x_1,S_x_2) + np.dot(S_y_1,S_y_2) + np.dot(S_z_1,S_z_2))

    # Add "h" term for the last particle
    left_size = int(2**(N-1))
    right_size = 1
    S_x_2 = np.kron(np.kron(np.identity(left_size), Sx), np.identity(right_size))
    S_y_2 = np.kron(np.kron(np.identity(left_size), Sy), np.identity(right_size))
    S_z_2 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))
    H += -h*(S_x_2 + S_y_2 + S_z_2)

    return H

# Below N1_index corresponds to the index of the first particle, and N2_index to the second one.
# First particle in the system has 0-th index, second has 1-st index and so on.
# N is the total number of particles
def generate_correlation_operator(N1_index, N2_index, N):
    if N1_index > N2_index:
        print("Wrong arguments. N1_index must be greater than N2_index.")
        sys.exit()
    if N1_index >= N or N2_index >= N:
        print("Wrong arguments. N1_index and N2_index cannot be >= N.")
        sys.exit()
    # First particle
    left_size = 2**N1_index
    right_size = int((2**N)/(2**(N1_index+1)))

    S_x_1 = np.kron(np.kron(np.identity(left_size), Sx), np.identity(right_size))
    S_y_1 = np.kron(np.kron(np.identity(left_size), Sy), np.identity(right_size))
    S_z_1 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))

    # Second particle
    left_size = 2**(N2_index)
    right_size = int((2**N)/(2**(N2_index+1)))
    S_x_2 = np.kron(np.kron(np.identity(left_size), Sx), np.identity(right_size))
    S_y_2 = np.kron(np.kron(np.identity(left_size), Sy), np.identity(right_size))
    S_z_2 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))

    S_corr = np.dot(S_x_1,S_x_2) + np.dot(S_y_1,S_y_2) + np.dot(S_z_1,S_z_2)
    return S_corr

def generate_Sz_operator(N1_index, N2_index, N):
    if N1_index > N2_index:
        print("Wrong arguments. N1_index must be greater than N2_index.")
        sys.exit()
    if N1_index >= N or N2_index >= N:
        print("Wrong arguments. N1_index and N2_index cannot be >= N.")
        sys.exit()
    # First particle
    left_size = 2**N1_index
    right_size = int((2**N)/(2**(N1_index+1)))
    S_z_1 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))

    # Second particle
    left_size = 2**(N2_index)
    right_size = int((2**N)/(2**(N2_index+1)))
    S_z_2 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))

    S_corr = np.dot(S_z_1,S_z_2)
    return S_corr


def generate_training_set_many_evolution_steps(training_set_size, vector_dims, H, t, t_total):
    print("######################################################")
    print("####### Generating training set")
    print("######################################################")
    if vector_dims != np.shape(H)[0]:
        print("Incompatible sizes of vector and Hamiltonian H.")
        sys.exit()
    U = expm(-1j * H * t)
    input_vectors = []
    output_vectors = []
    for i in range(training_set_size):
        if i % 1000 == 0:
            print("[ ", i, " / ", training_set_size, " ]", end="\r")
        psi_t = generate_random_vector(vector_dims)
        for _t in np.arange(0, t_total, t):
            input_vectors.append(psi_t)
            psi_t = U.dot(psi_t)
            output_vectors.append(psi_t)
    print("[ ", training_set_size, " / ", training_set_size, " ]")

    return (np.array(input_vectors), np.array(output_vectors))

def generate_test_set_many_evolution_steps(training_set_size, vector_dims, H, t, t_total):
    return generate_training_set_many_evolution_steps(training_set_size, vector_dims, H, t, t_total)

def generate_training_set(training_set_size, vector_dims, H, t):
    print("######################################################")
    print("####### Generating training set")
    print("######################################################")
    if vector_dims != np.shape(H)[0]:
        print("Incompatible sizes of vector and Hamiltonian H.")
        sys.exit()
    U = expm(-1j * H * t)
    input_vectors = []
    output_vectors = []
    for i in range(training_set_size):
        if i % 1000 == 0:
            print("[ ", i, " / ", training_set_size, " ]", end="\r")
        psi_0 = generate_random_vector(vector_dims)
        psi_t = U.dot(psi_0)
        input_vectors.append(psi_0)
        output_vectors.append(psi_t)
    print("[ ", training_set_size, " / ", training_set_size, " ]")

    return (np.array(input_vectors), np.array(output_vectors))

def generate_test_set(training_set_size, vector_dims, H, t):
    return generate_training_set(training_set_size, vector_dims, H, t)

def reshape_set(vectors, remove_input_matrix=True):
    reshaped_vectors = []
    if not remove_input_matrix:
        for vector in vectors:
            reshaped_vector = []
            for component in vector:
                reshaped_vector.append([component.real, component.imag])
            reshaped_vectors.append(reshaped_vector)
    else:
        for i in range(np.shape(vectors)[0]):
            reshaped_vector = []
            for component in vectors[0]:
                reshaped_vector.append([component.real, component.imag])
            del vectors[0]
            reshaped_vectors.append(reshaped_vector)
    return np.array(reshaped_vectors)

def reshape_two_sets(inputs, outputs):
    reshaped_inputs = reshape_set(inputs)
    reshaped_outputs = reshape_set(outputs)
    return (reshaped_inputs, reshaped_outputs)

def flatten_set(vectors):
    return vectors.reshape((len(vectors), np.prod(vectors.shape[1:])))

def evolve_whole_set(vectors_set, U):
    evolved_set = []
    for i in range(np.shape(vectors_set)[0]):
        evolved_set.append(np.matmul(U, vectors_set[i]))
    return np.array(evolved_set)

def evolve_whole_set_many_evolution_steps(vectors_set, U, t, t_total):
    input_vectors = []
    output_vectors = []

    # Compute the first timestep
    for vector in vectors_set:
        psi_t = U.dot(vector)
        input_vectors.append(vector)
        output_vectors.append(psi_t)

    vector_length = len(input_vectors)
    pointer = 0

    # print("=== Initial sizes === ")
    # print("input: ", np.shape(input_vectors))
    # print("output: ", np.shape(output_vectors))

    for _t in range(int(t_total/t)-1): # We have to subtract 1, because we already did one step of evolution in the above loop
        for i in range(vector_length):
            psi_0 = output_vectors[pointer]
            psi_t = U.dot(psi_0)
            input_vectors.append(psi_0)
            output_vectors.append(psi_t)
            pointer += 1
        # print("_t: ", _t)
        # print("input: ", np.shape(input_vectors))
        # print("output: ", np.shape(output_vectors))

    return (input_vectors, output_vectors)

# This version works in the following way:
# - take input vector,
# - evolve it for given number of steps,
# - go on to the next vector.
# This version will allow us to train a RNN network by looking at many time series,
# each of which being one vector evolved for a given number of timesteps
def evolve_whole_set_many_evolution_steps_rnn(vectors_set, U, t, t_total):
    input_vectors = []
    output_vectors = []

    for vector in vectors_set:
        psi_0 = vector
        for _t in range(int(t_total/t)):
            psi_t = U.dot(psi_0)
            input_vectors.append(psi_0)
            output_vectors.append(psi_t)
            psi_0 = psi_t

    return (input_vectors, output_vectors)

def overlap(vector_0, vector_1):
    overlap = complex(0, 0)
    for i in range(np.shape(vector_0)[0]):
        overlap += np.conj(vector_0[i])*vector_1[i]
        return overlap

# Here vector_0 and vector_1 are reshaped vectors (arrays two times longer than initial
# arrays of complex numbers, containing only real numbers)
def overlap_of_reshaped(vector_0, vector_1):
    overlap = complex(0, 0)
    for i in range(0, np.shape(vector_0)[0]-1, 2):
        first_complex = complex(vector_0[i], vector_0[i+1])
        second_complex = complex(vector_1[i], vector_1[i+1])
        overlap += np.conj(first_complex)*second_complex
    return overlap

def overlap_of_reshaped_sets(vectors_set_0, vectors_set_1):
    overlaps = []
    for i in range(np.shape(vectors_set_0)[0]):
        overlap = overlap_of_reshaped(vectors_set_0[i], vectors_set_1[i])
        overlaps.append(overlap)
    return overlaps

def correlation_in_reshaped_set(reshaped_vectors_set, corr_operator):
    correlations_real = []
    correlations_imag = []
    for reshaped_vector in reshaped_vectors_set:
        # Just for this calculation reshape vector back again to its initial form (with complex numbers
        # instead of two real ones)
        vector = []
        for i in range(0, len(reshaped_vector), 2):
            vector.append(complex(reshaped_vector[i], reshaped_vector[i+1]))
        vector = np.array(vector)
        # print(vector)
        correlation = np.dot(np.dot(np.conjugate(vector.T), corr_operator), vector) # First vector is not only transposed, but also conjugated!!!
        correlations_real.append(correlation.real)
        correlations_imag.append(correlation.imag)
    return (correlations_real, correlations_imag)

################################################################################
# Declare function used to load eigenvectors from binary files generated by
# other Fortran code
################################################################################

def load_array_from_file(filename, size):
    rec_delim = 4  # This value depends on the Fortran compiler
    with open(filename, "rb") as f:
        f.seek(rec_delim, 1)  # begin record
        arr = np.fromfile(file=f,
                            dtype=np.float64).reshape((size, size), order='F')
        f.seek(rec_delim, 1)  # end record
    return arr

def load_vector_from_file(filename, size):
    rec_delim = 4  # This value depends on the Fortran compiler
    with open(filename, "rb") as f:
        f.seek(rec_delim, 1)  # begin record
        arr = np.fromfile(file=f,
                            dtype=np.float64).reshape(size, order='F')
        f.seek(rec_delim, 1)  # end record
    return arr
