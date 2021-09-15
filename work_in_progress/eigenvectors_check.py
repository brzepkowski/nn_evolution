import numpy as np
from scipy.linalg import expm
import generators
import sys, os
from numpy.linalg import eig
from cmath import isclose, exp
import copy

N = 8 # Number of particles in a chain
t = 0.1 # Length of one timestep
periodic = False
print("#################################")
print("### PERIODIC: ", periodic)
print("#################################")

H = generators.generate_Heisenberg_hamiltonian(N, J=1.0, periodic=periodic)
U_t = expm(-1j * H * t)

eigenvalues, eigenvectors = eig(np.array(H)) # Here each eigenvector is represented by a column in a matrix
eigenvectors = eigenvectors.T # By conducting transposition each eigenvector is stored in a separate row

# Checking, if eigenvectors computed by numpy.linalg.eig are in fact eigenvectors of H
print("H...", end="")
for i in range(np.shape(eigenvectors)[0]):
    eigenvector = eigenvectors[i]
    eigenvalue = eigenvalues[i]

    left = np.matmul(H, eigenvector)
    right = np.dot(eigenvalue, eigenvector)
    same = []
    for j in range(len(eigenvector)):
        same.append(isclose(left[j], right[j], abs_tol=1e-12))
    if False in same:
        print("DIFFERENT!")
        break
print("...CHECKED")

# Checking, if eigenvectors computed by numpy.linalg.eig are also eigenvectors of U
print("U...", end="")
for i in range(np.shape(eigenvectors)[0]):
    eigenvector = eigenvectors[i]
    eigenvalue = eigenvalues[i]
    left = np.matmul(U_t, eigenvector)
    right = np.dot(exp(-1j * eigenvalue * t), eigenvector)

    same = []
    for j in range(len(eigenvector)):
        same.append(isclose(left[j], right[j], abs_tol=1e-12))
    if False in same:
        print("DIFFERENT!")
        break
print("...CHECKED")

indexed_eigenvalues = []
for i in range(len(eigenvalues)):
    indexed_eigenvalues.append((eigenvalues[i], i))
eigenvalues_sorted = sorted(indexed_eigenvalues, key=lambda x: x[0])
eigenvalues_sorted = eigenvalues_sorted[:10]

file = open("Correlations.dat", "w+")
# Check the Sz correlation between the first particle and all other particles
for (eigenvalue, i) in eigenvalues_sorted:
    print("-" * 30)
    print("EIGENVALUE: ", eigenvalue)


    file.write("------------------------------------------\n")
    line = "EIGENVALUE: " + str(eigenvalue) + "\n"
    file.write(line)
    # print(i,"-th eigenvector")
    eigenvector = eigenvectors[i]
    # print("|psi> = ", eigenvector)
    for j in range(1, N):
        S_corr = generators.generate_Sz_operator(0, j, N)
        correlation = np.dot(np.dot(np.conjugate(eigenvector.T), S_corr), eigenvector)
        print("Correlation 0 - ", j, ": ", correlation)
        line = "Correlation 0 - " + str(j) + ": " + str(correlation) + "\n"
        file.write(line)

file.close()
