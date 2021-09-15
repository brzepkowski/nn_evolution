from keras.layers import Input, Dense, concatenate
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import generators, generators_quench
import sys

N_min = 6
N_max = 7
t = 0.1
t_total = 1
train_num_samples = 50000
test_num_samples = 1000

x = list(np.arange(t, t_total+t, t))
x_max = x[-1]
y_overlap = []
legend = []
for N in range(N_min, N_max+1):
    y_correlations_real = []
    y_correlations_imag = []
    print("#################################")
    print("###### N: ", N)
    print("#################################")
    legend.append(str(N))
    H = generators.generate_Heisenberg_hamiltonian(N, 1.0)
    H_quenched = generators.generate_Heisenberg_hamiltonian(N, -1.0)
    U_t_quenched = expm(-1j * H_quenched * t)
    U_t_quenched_simple = np.identity(2**N) - (1j * H_quenched * t)

    # Spin correlation operator between first two particles
    S_corr = generators.generate_correlation_operator(0, 1, N)

    (training_input_vectors, training_output_vectors) = generators.generate_training_set(train_num_samples, 2**N, H_quenched, t)
    (test_input_vectors, test_output_vectors) = generators.generate_test_set(test_num_samples, 2**N, H_quenched, t)

    # Reshape generated sets, to extract imaginary parts of numbers as second dimension in a vector
    (reshaped_training_input_vectors, reshaped_training_output_vectors) = generators.reshape_two_sets(training_input_vectors, training_output_vectors)
    (reshaped_test_input_vectors, reshaped_test_output_vectors) = generators.reshape_two_sets(test_input_vectors, test_output_vectors)

    # Flatten the data, so that it can be injected into neural network
    reshaped_training_input_vectors = generators.flatten_set(reshaped_training_input_vectors)
    reshaped_training_output_vectors = generators.flatten_set(reshaped_training_output_vectors)

    reshaped_test_input_vectors = generators.flatten_set(reshaped_test_input_vectors)
    reshaped_test_output_vectors = generators.flatten_set(reshaped_test_output_vectors)

    # ==============================================================================
    input = Input(shape=(2*(2**N),))
    # hidden_layer_0 = Dense(2*(2**N), activation='tanh')(input)
    # hidden_layer_1 = Dense(2*(2**N), activation='tanh')(hidden_layer_0)
    output = Dense(2*(2**N), activation='tanh')(input)
    # ==============================================================================

    model = Model(input, output)
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(reshaped_training_input_vectors, reshaped_training_output_vectors,
                                        batch_size=256,
                                        epochs=100,
                                        shuffle=True)
                                        # validation_data=(reshaped_test_input_vectors, reshaped_test_output_vectors))

    score, acc = model.evaluate(reshaped_test_input_vectors, reshaped_test_output_vectors,
                                                batch_size=256)

    print('Test score:', score)
    print('Test accuracy:', acc)

    #
    # Generate new testing set, which finds eigenvectors of H and then performs evoulution
    # with U_quenched = exp(-i H_quenched t)
    #
    (training_input_vectors, training_output_vectors) = generators_quench.generate_testing_set_quenched(2**N, H, H_quenched, t)
    (reshaped_test_input_vectors, reshaped_test_output_vectors) = generators.reshape_two_sets(test_input_vectors, test_output_vectors)
    reshaped_test_input_vectors = generators.flatten_set(reshaped_test_input_vectors)
    reshaped_test_output_vectors = generators.flatten_set(reshaped_test_output_vectors)

    #
    # Perform evolution of input test set using three methods:
    # - full evolution operator U_quenched = exp( -i H_quenched t),
    # - simple evolution operator U_quenched_simple = I - i H_quenched t,
    # - neural network.
    #
    evolved_test_vectors_full_op = test_input_vectors
    evolved_test_vectors_simple_op = test_input_vectors
    predicted_test_vectors = reshaped_test_input_vectors
    mean_overlap_per_timestep = []
    y_coefficients_real = []
    y_coefficients_imag = []
    y_mean_correlations_real = []
    y_mean_correlations_imag = []
    for time_step in np.arange(t, t_total+t, t):
        # Evolution using full operator
        evolved_test_vectors_full_op = generators.evolve_whole_set(evolved_test_vectors_full_op, U_t_quenched)
        evolved_test_vectors_full_op_reshaped = generators.flatten_set(generators.reshape_set(evolved_test_vectors_full_op))
        # Evolution using simple operator
        evolved_test_vectors_simple_op = generators.evolve_whole_set(evolved_test_vectors_simple_op, U_t_quenched_simple)
        evolved_test_vectors_simple_op_reshaped = generators.flatten_set(generators.reshape_set(evolved_test_vectors_simple_op))
        # Evolution using neural network
        predicted_test_vectors = model.predict(predicted_test_vectors)

        # Compute overlap between precisely evolved vector and the one returned by neural network
        overlaps = generators.overlap_of_reshaped_sets(evolved_test_vectors_full_op_reshaped, predicted_test_vectors)
        # Save mean overlap
        mean_overlap_per_timestep.append(np.mean(overlaps))
        # Save first coefficient of the first vector
        y_coefficients_real.append([evolved_test_vectors_full_op_reshaped[0][0], evolved_test_vectors_simple_op_reshaped[0][0], predicted_test_vectors[0][0]])
        y_coefficients_imag.append([evolved_test_vectors_full_op_reshaped[0][1], evolved_test_vectors_simple_op_reshaped[0][1], predicted_test_vectors[0][1]])

        # Compute spin correlation between first two particles (on vectors evolved using full operator)
        (correlations_full_op_real, correlations_full_op_imag) = generators.correlation_in_reshaped_set(evolved_test_vectors_full_op_reshaped, S_corr)

        # Compute spin correlation between first two particles (on vectors evolved using simple operator)
        (correlations_simple_op_real, correlations_simple_op_imag) = generators.correlation_in_reshaped_set(evolved_test_vectors_simple_op_reshaped, S_corr)

        # Compute spin correlation between first two particles (on vectors evolved using neural network)
        (correlations_nn_real, correlations_nn_imag) = generators.correlation_in_reshaped_set(predicted_test_vectors, S_corr)
        y_mean_correlations_real.append([np.mean(correlations_full_op_real), np.mean(correlations_simple_op_real), np.mean(correlations_nn_real)])
        y_mean_correlations_imag.append([np.mean(correlations_full_op_imag), np.mean(correlations_simple_op_imag), np.mean(correlations_nn_imag)])

    y_overlap.append(mean_overlap_per_timestep)

    y_coefficients_real = np.array(y_coefficients_real)
    y_coefficients_imag = np.array(y_coefficients_imag)
    # Plot real part of the coefficient
    plt.title("Coefficient, N: " + str(N) + ", timestep: " + str(t) + " (real) \n QUENCH")
    plt.plot(x, y_coefficients_real, '.-')
    plt.xlabel('Time')
    plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
    plt.grid()
    plt.savefig("Coeff_N_" + str(N) + "_timestep_" + str(t) + "_(real)_quench.png")
    plt.clf()
    # Plot imaginary part of the coefficient
    plt.title("Coefficient, N: " + str(N) + ", timestep: " + str(t) + " (imaginary) \n QUENCH")
    plt.plot(x, y_coefficients_imag, '.-')
    plt.xlabel('Time')
    plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
    plt.grid()
    plt.savefig("Coeff_N_" + str(N) + "_timestep_" + str(t) + "_(imag)_quench.png")
    plt.clf()

    plt.title("Mean correlation, N: " + str(N) + ", timestep: " + str(t) + " (real) \n QUENCH")
    plt.plot(x, y_mean_correlations_real, '.-')
    plt.xlabel('Time')
    plt.ylabel('Mean correlation')
    plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
    plt.xlim(xmin=0, xmax=x_max)
    plt.grid()
    plt.savefig("Corr_N_" + str(N) + "_timestep_" + str(t) + "_(real)_quench.png")
    plt.clf()

    plt.title("Mean correlation, N: " + str(N) + ", timestep: " + str(t) + " (imag) \n QUENCH")
    plt.plot(x, y_mean_correlations_imag, '.-')
    plt.xlabel('Time')
    plt.ylabel('Mean correlation')
    plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
    plt.xlim(xmin=0, xmax=x_max)
    plt.grid()
    plt.savefig("Corr_N_" + str(N) + "_timestep_" + str(t) + "_(imag)_quench.png")
    plt.clf()

# Reshape data so that it has appropriate dimensions
y_overlap = np.array(y_overlap).T.tolist()
plt.title("Overlap, timestep: " + str(t) + " \n QUENCH")
plt.plot(x, y_overlap, '.-')
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.gca().legend(legend)
plt.xlim(xmin=0, xmax=x_max)
plt.grid()
plt.savefig("Overlap_timestep_" + str(t) + "_quench.png")
