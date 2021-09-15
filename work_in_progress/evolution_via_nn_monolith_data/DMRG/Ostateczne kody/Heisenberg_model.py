import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm
import sys
import copy

Sx = 0.5*np.array([[0 + 0j, 1 + 0j], [1 + 0j,  0 + 0j]])
Sy = 0.5*np.array([[0 + 0j, 0 - 1j], [0 + 1j,  0 + 0j]])
Sz = 0.5*np.array([[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]])

def generate_Heisenberg_hamiltonian(N, Jx, Jy, Jz, hx=0, hy=0, hz=0, periodic=False):
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

        H += Jx*np.dot(S_x_1,S_x_2)
        H += Jy*np.dot(S_y_1,S_y_2)
        H += Jz*np.dot(S_z_1,S_z_2)

        # H -= hx*S_x_1
        # H -= hy*S_y_1
        # H -= hz*S_z_1
        # Varying external field
        H -= i*hx*S_x_1
        H -= i*hy*S_y_1
        H -= i*hz*S_z_1


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

        H += Jx*np.dot(S_x_1,S_x_2)
        H += Jy*np.dot(S_y_1,S_y_2)
        H += Jz*np.dot(S_z_1,S_z_2)

    # Add "h" term for the last particle
    left_size = int(2**(N-1))
    right_size = 1
    S_x_2 = np.kron(np.kron(np.identity(left_size), Sx), np.identity(right_size))
    S_y_2 = np.kron(np.kron(np.identity(left_size), Sy), np.identity(right_size))
    S_z_2 = np.kron(np.kron(np.identity(left_size), Sz), np.identity(right_size))

    # H -= hx*S_x_2
    # H -= hy*S_y_2
    # H -= hz*S_z_2
    # Varying external field
    H -= (N-1)*hx*S_x_2
    H -= (N-1)*hy*S_y_2
    H -= (N-1)*hz*S_z_2

    return H

############################################################################
# MAIN
############################################################################
if __name__ == "__main__":
    N = int(sys.argv[1])
    np.set_printoptions(linewidth=np.inf)
    ############################################################################
    # Hamiltonian H
    ############################################################################

    H = generate_Heisenberg_hamiltonian(N=N, Jx=1, Jy=1, Jz=1, hx=0, hy=0, hz=2, periodic=False)
    print("#"*100)
    print("###### H #####")
    print("#"*100)

    print(H)
    eigenvalues_H, eigenvectors_H = eig(H)
    print("eigvals: ")
    print(eigenvalues_H)
    print("eigvecs: ")
    print(eigenvectors_H)
    # print("Is Hermitian: ", H == H.conj().T)

    print("#"*100)
    print("###### HQ #####")
    print("#"*100)

    HQ = generate_Heisenberg_hamiltonian(N=N, Jx=1, Jy=1, Jz=1, hx=0, hy=0, hz=0, periodic=False)
    print(HQ)
    eigenvalues_HQ, eigenvectors_HQ = eig(HQ)
    print("eigvals: ")
    print(eigenvalues_HQ)
    print("eigvecs: ")
    print(eigenvectors_HQ)
    # print("Is Hermitian: ", HQ == HQ.conj().T)

    commutator = np.dot(H, HQ) - np.dot(HQ, H)
    print("commutator:")
    print(commutator)
    # sys.exit()

    print("#"*100)
    print("###### EVOLUTION #####")
    print("#"*100)

    # H_random = np.random.random([2**N, 2**N])
    # H_random = H_random + H_random.conj().T
    # print("H_random:")
    # print(H_random)
    # commutator = np.dot(H, H_random) - np.dot(H_random, H)
    # print("commutator:")
    # print(commutator)
    # print("Is Hermitian: ", H_random == H_random.conj().T)

    t = 1
    U = expm(-(1j)*t*HQ)
    # U = expm(-(1j)*t*H_random)

    eigenvectors_H = eigenvectors_H.T
    for eigenvector in eigenvectors_H:

        eigenvector = np.array([eigenvector]).T
        initial_eigenvector = copy.deepcopy(eigenvector)
        energy = np.dot(np.dot(np.conj(eigenvector).T, H), eigenvector)
        # print("E: ", energy)

        for j in range(10):
            # print(eigenvector.T)
            # print("[PRZED]:", eigenvector)
            eigenvector = np.dot(U, eigenvector)
            # print("[PO]:", eigenvector)
            overlap = np.dot(np.conj(eigenvector).T, initial_eigenvector)
            # print("overlap: ", overlap)
            probability = np.conj(overlap)*overlap

            energy_2 = np.dot(np.dot(np.conj(eigenvector).T, H), eigenvector)
            # print("E: ", energy_2, end="\r", flush=True)
            print("overlap: ", overlap, end="\t")
            print(" || probability: ", probability, end="\t")
            print(" || E: ", energy_2, flush=True)
        # print()
        # print("E: ", energy_2)
        print("#"*50)
