#
# In this structure normalization is only added at the end of the network (not inbetween hidden layers).
#
# from keras.layers import Input, Dense, Lambda, concatenate
# from keras.models import Model
# from keras.backend import l2_normalize
import numpy as np
# import matplotlib.pyplot as plt
from scipy.linalg import expm
import generators
import sys, os
from numpy.linalg import norm
from itertools import combinations
from numpy.linalg import eig

# if len(sys.argv) < 10:
#     print("Wrong number of arguments provided. Please give them in the following fashion:")
#     print("- optimizer,")
#     print("- loss function,")
#     print("- activaton function,")
#     print("- N_min,")
#     print("- N_max,")
#     print("- timestep,")
#     print("- total_evolution_time,")
#     print("- train_num_samples,")
#     print("- test_num_samples.")
#
# # optimizer = "adam"
# # loss = "mean_squared_error"
# # activation = "tanh"
# optimizer = sys.argv[1]
# loss = sys.argv[2]
# activation = sys.argv[3]
# N_min = int(sys.argv[4])
# N_max = int(sys.argv[5])
# t = float(sys.argv[6])
# t_total = float(sys.argv[7])
# train_num_samples = int(sys.argv[8])
# test_num_samples = int(sys.argv[9])

N_min = N_max = 2
t = t_total = 0.1

# x = list(np.arange(t, t_total+t, t))
# x_max = x[-1]
# y_overlap = []
legend = []
for N in range(N_min, N_max+1):
    y_correlations_real = []
    y_correlations_imag = []
    print("#################################")
    print("###### N: ", N)
    print("#################################")
    legend.append(str(N))

    H = generators.generate_Heisenberg_hamiltonian(N, J=1.0)
    U_t = expm(-1j * H * t)
    print("H:")
    print(H)
    print("U:")
    print(U_t)
    print("=================================================")

    _, eigenvectors = eig(H) # Here each eigenvector is represented by a column in a matrix
    eigenvectors = eigenvectors.T # By conducting transposition each eigenvector is stored in a separate row
    # print("(1) Eigenvecs: ", eigenvectors)
    reshaped_eigenvectors = generators.reshape_set(eigenvectors)
    # print("(2) Eigenvecs: ", reshaped_eigenvectors)
    reshaped_test_input_vectors = generators.flatten_set(reshaped_eigenvectors)
    # print("(3) Eigenvecs: ", reshaped_test_input_vectors)

    evolved_test_vectors_full_op = eigenvectors

    # print("Current timestep: ", round(time_step, 5), " / ", round(x_max, 2), end="\r")
    # Evolution using full operator
    # print("Wektory przed: ")
    # print(evolved_test_vectors_full_op)
    evolved_test_vectors_full_op = generators.evolve_whole_set(evolved_test_vectors_full_op, U_t)
    # print("Wektory po: ")
    # print(evolved_test_vectors_full_op)
    evolved_test_vectors_full_op_reshaped = generators.flatten_set(generators.reshape_set(evolved_test_vectors_full_op))
    # print("Wektory po (reshaped): ")
    # print(evolved_test_vectors_full_op_reshaped)
    sys.exit()
    """
        # Evolution using neural network
        predicted_test_vectors = model.predict(predicted_test_vectors)
        # for predicted_vector in predicted_test_vectors:
        #     print(norm(predicted_vector))

        # Compute overlap between precisely evolved vector and the one returned by neural network
        overlaps = generators.overlap_of_reshaped_sets(evolved_test_vectors_full_op_reshaped, predicted_test_vectors)
        # print("overlaps: ", overlaps)
        # Save mean overlap
        mean_overlap_per_timestep.append(np.mean(overlaps))
        # Save first coefficient of the first vector
        # y_coefficients_real.append([evolved_test_vectors_full_op_reshaped[0][0], evolved_test_vectors_simple_op_reshaped[0][0], predicted_test_vectors[0][0]])
        # y_coefficients_imag.append([evolved_test_vectors_full_op_reshaped[0][1], evolved_test_vectors_simple_op_reshaped[0][1], predicted_test_vectors[0][1]])
        y_coefficients_real.append([evolved_test_vectors_full_op_reshaped[0][0], predicted_test_vectors[0][0]])
        y_coefficients_imag.append([evolved_test_vectors_full_op_reshaped[0][1], predicted_test_vectors[0][1]])


        #
        # All possible correlations
        #

        for pair in pairs:
            (N0, N1) = pair
            S_corr = generators.generate_correlation_operator(N0, N1, N)
            # print("S_corr - ", N0, ": ", N1)
            # print(S_corr)

            # Compute spin correlation between the two particles (on vectors evolved using full operator)
            (correlations_full_op_real, correlations_full_op_imag) = generators.correlation_in_reshaped_set(evolved_test_vectors_full_op_reshaped, S_corr)

            # Compute spin correlation between first two particles (on vectors evolved using neural network)
            (correlations_nn_real, correlations_nn_imag) = generators.correlation_in_reshaped_set(predicted_test_vectors, S_corr)

            # y_mean_correlations_real.append([np.mean(correlations_full_op_real), np.mean(correlations_nn_real)])
            # y_mean_correlations_imag.append([np.mean(correlations_full_op_imag), np.mean(correlations_nn_imag)])

            if (N0,N1) not in y_mean_correlations:
                y_mean_correlations[N0, N1] = {"real":[], "imag":[]}
            y_mean_correlations[N0, N1]["real"].append([np.mean(correlations_full_op_real), np.mean(correlations_nn_real)])
            y_mean_correlations[N0, N1]["imag"].append([np.mean(correlations_full_op_imag), np.mean(correlations_nn_imag)])
    print("Finished! Current timestep: ", t_total, " / ", t_total)

    y_overlap.append(mean_overlap_per_timestep)

    y_coefficients_real = np.array(y_coefficients_real)
    y_coefficients_imag = np.array(y_coefficients_imag)
    """
