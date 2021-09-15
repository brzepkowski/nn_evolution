import numpy as np
from scipy.linalg import expm
from numpy.linalg import eig

# The size of training set is equivalent to the size of matrix H, because it must
# have at most dim(H) different eigenvectors
def generate_testing_set_quenched(vector_dims, H, H_quenched, t):
    print("######################################################")
    print("####### Generating testing set - quenched Hamiltonian")
    print("######################################################")
    if vector_dims != np.shape(H)[0]:
        print("Incompatible sizes of vector and Hamiltonian H.")
        sys.exit()
    U_quenched = expm(-1j * H_quenched * t)
    print("Diagonalizing Hamilonian...", end="")
    (eigvals, eigvecs) = eig(H)
    print(" Done!")
    # print("output[0]: ", eigvals[0])
    # print("output[1]: ")
    # print(eigvecs[:, 0])

    input_vectors = []
    output_vectors = []
    for i in range(np.shape(H)[0]):
        psi_0 = eigvecs[:, i]
        psi_t = U_quenched.dot(psi_0)
        input_vectors.append(psi_0)
        output_vectors.append(psi_t)

    return (np.array(input_vectors), np.array(output_vectors))
