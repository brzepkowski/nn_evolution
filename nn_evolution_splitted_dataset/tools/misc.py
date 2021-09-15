# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

import numpy as np
from numpy.random import rand
from numpy.linalg import eig, norm
from cmath import sqrt
from scipy.linalg import logm
import sys

Sx = 0.5*np.array([[0 + 0j, 1 + 0j], [1 + 0j,  0 + 0j]])
Sy = 0.5*np.array([[0 + 0j, 0 - 1j], [0 + 1j,  0 + 0j]])
Sz = 0.5*np.array([[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]])

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

def overlap(vector_0, vector_1):
    # overlap = complex(0, 0)
    # for i in range(np.shape(vector_0)[0]):
    #     overlap += np.conj(vector_0[i])*vector_1[i]
    #     return overlap
    vector_0 = np.array([vector_0]).T
    print("vector_0: ")
    print(vector_0)
    vector_1 = np.array([vector_1]).T
    print("vector_1: ")
    print(vector_1)
    overlap = np.dot(np.conj(vector_0).T, vector_1)[0][0]
    print("overlap: ", overlap)
    sys.exit()
    return overlap

# def overlap(vector_0, vector_1):
#     overlap = complex(0, 0)
#     for i in range(np.shape(vector_0)[0]):
#         overlap += np.conj(vector_0[i])*vector_1[i]
#     return overlap

# Here vector_0 and vector_1 are reshaped vectors (arrays two times longer than initial
# arrays of complex numbers, containing only real numbers)
def overlap_of_reshaped(vector_0, vector_1):
    overlap = complex(0, 0)
    for i in range(0, np.shape(vector_0)[0]-1, 2):
        first_complex = complex(vector_0[i], vector_0[i+1])
        second_complex = complex(vector_1[i], vector_1[i+1])
        overlap += np.conj(first_complex)*second_complex

        #WARNING: We are making square of the overlap!
        overlap_squared = np.conj(overlap)*overlap
    return float(overlap_squared)

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

def compress_vector(vector):
    compressed_vector = []
    for i in range(0, len(vector), 2):
        real_part = vector[i]
        imag_part = vector[i+1]
        compressed_vector.append(complex(real_part, imag_part))

    return compressed_vector

def energies_of_reshaped_set(vectors_set, hamil):
    energies = []
    for vector in vectors_set:
        compressed_vector = np.array([compress_vector(vector)])
        # Below line produces a scalar, which is stored as 2D matrix... Thats why we need [0][0] at the end.
        energy = np.dot(compressed_vector.conj(), np.dot(hamil, compressed_vector.T))[0][0]
        energies.append(float(energy))
    return energies

################################################################################
# ENTANGLEMENT
################################################################################

def entanglement_entropy(state, N):
    state = np.array(state, ndmin=2)
    ket = state
    bra = state.conj().T
    rho_final = np.outer(ket,bra)
    S = []
    for d in range(1, N):
        Ia = np.identity(2**d)
        Ib = np.identity(2**(N-d))
        Tr_a = np.empty([2**d, 2**(N-d), 2**(N-d)], dtype=complex)
        for i in range(2**d):
            ai = np.array(Ia[i], ndmin=2).T
            Tr_a[i] = np.kron(ai.conj().T, Ib).dot(rho_final).dot(np.kron(ai,Ib))
        rho_b = Tr_a.sum(axis=0)
        rho_b_l2 = logm(rho_b)
        S_rho_b = - rho_b.dot(rho_b_l2).trace()
        S.append(S_rho_b)

    return float(np.array(S).mean())

def entanglements_of_reshaped_set(vectors_set, N):
    # print("size(vectors_set): ", np.shape(vectors_set))
    entanglements = []
    # print("Calculating entanglement...")
    counter = 0
    for vector in vectors_set:
        # print(counter)
        compressed_vector = np.array([compress_vector(vector)])
        entanglement = entanglement_entropy(compressed_vector, N)
        # print("entanglement: ", entanglement)
        entanglements.append(entanglement)
        counter += 1
    return np.array(entanglements)
