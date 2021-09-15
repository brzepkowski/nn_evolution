#
# In this structure normalization is only added at the end of the network (not inbetween hidden layers).
#
from keras.layers import Input, Dense, Lambda, concatenate
from keras.models import Model
from keras.backend import l2_normalize
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import generators
import sys, os
from numpy.linalg import norm
from itertools import combinations
from numpy.linalg import eig

if len(sys.argv) < 10:
    print("Wrong number of arguments provided. Please give them in the following fashion:")
    print("- optimizer,")
    print("- loss function,")
    print("- activaton function,")
    print("- N_min,")
    print("- N_max,")
    print("- timestep,")
    print("- total_evolution_time,")
    print("- train_num_samples,")
    print("- test_num_samples.")

# optimizer = "adam"
# loss = "mean_squared_error"
# activation = "tanh"
optimizer = sys.argv[1]
loss = sys.argv[2]
activation = sys.argv[3]
N_min = int(sys.argv[4])
N_max = int(sys.argv[5])
t = float(sys.argv[6])
t_total = float(sys.argv[7])
train_num_samples = int(sys.argv[8])
test_num_samples = int(sys.argv[9])

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
    H = generators.generate_Heisenberg_hamiltonian(N, J=1.0, periodic=False)
    U_t = expm(-1j * H * t)
    U_t_simple = np.identity(2**N) - (1j * H * t)

    # Spin correlation operator between first two particles
    # S_corr = generators.generate_correlation_operator(0, 1, N)

    # List of all possible pairs in the chain (needed for calculation of correlations)
    pairs = list(combinations(range(N), 2))

    (training_input_vectors, training_output_vectors) = generators.generate_training_set_many_evolution_steps(train_num_samples, 2**N, H, t, t_total)
    (test_input_vectors, test_output_vectors) = generators.generate_test_set_many_evolution_steps(test_num_samples, 2**N, H, t, t_total)

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
    hidden_layer_0 = Dense(2*(2**N), activation=activation)(input)
    hidden_layer_1 = Dense(2*(2**N), activation=activation)(hidden_layer_0)
    hidden_layer_2 = Dense(2*(2**N), activation=activation)(hidden_layer_1)
    hidden_layer_3 = Dense(2*(2**N), activation=activation)(hidden_layer_2)
    hidden_layer_4 = Dense(2*(2**N), activation=activation)(hidden_layer_3)
    hidden_layer_5 = Dense(2*(2**N), activation=activation)(hidden_layer_4)
    output = Lambda(lambda t: l2_normalize(1000*t, axis=1))(hidden_layer_5)
    # ==============================================================================

    model = Model(input, output)
    model.summary()

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = model.fit(reshaped_training_input_vectors, reshaped_training_output_vectors,
                                        batch_size=256,
                                        epochs=100,
                                        shuffle=True)
                                        # shuffle=False) # Changed to check, if network will learn better on more consistent "packages"
                                        # validation_data=(reshaped_test_input_vectors, reshaped_test_output_vectors))

    score, acc = model.evaluate(reshaped_test_input_vectors, reshaped_test_output_vectors,
                                                batch_size=256)

    print('Test score:', score)
    print('Test accuracy:', acc)

    #
    # Check how does this network behave for longer evolution times
    #

    ############################################################################
    x = list(np.arange(t, (1*t_total)+t, t))
    x_max = x[-1]
    y_overlap = []
    ############################################################################


    # WARNING: We are changing these random vectors to eigenvectors of the hamiltonian
    # (test_input_vectors, test_output_vectors) = generators.generate_test_set(test_num_samples, 2**N, H, t)
    # (reshaped_test_input_vectors, _) = generators.reshape_two_sets(test_input_vectors, test_output_vectors)
    # reshaped_test_input_vectors = generators.flatten_set(reshaped_test_input_vectors)

    _, eigenvectors = eig(H) # Here each eigenvector is represented by a column in a matrix
    eigenvectors = eigenvectors.T # By conducting transposition each eigenvector is stored in a separate row
    reshaped_eigenvectors = generators.reshape_set(eigenvectors)
    reshaped_test_input_vectors = generators.flatten_set(reshaped_eigenvectors)
    evolved_test_vectors_full_op = eigenvectors

    # evolved_test_vectors_full_op = test_input_vectors
    # evolved_test_vectors_simple_op = test_input_vectors
    predicted_test_vectors = reshaped_test_input_vectors
    mean_overlap_per_timestep = []
    y_coefficients_real = []
    y_coefficients_imag = []
    # y_mean_correlations_real = []
    # y_mean_correlations_imag = []
    y_mean_correlations = {}

    for time_step in x:
        print("Current timestep: ", round(time_step, 5), " / ", round(x_max, 2), end="\r")
        # Evolution using full operator
        evolved_test_vectors_full_op = generators.evolve_whole_set(evolved_test_vectors_full_op, U_t)
        evolved_test_vectors_full_op_reshaped = generators.flatten_set(generators.reshape_set(evolved_test_vectors_full_op))
        # # Evolution using simple operator
        # evolved_test_vectors_simple_op = generators.evolve_whole_set(evolved_test_vectors_simple_op, U_t_simple)
        # evolved_test_vectors_simple_op_reshaped = generators.flatten_set(generators.reshape_set(evolved_test_vectors_simple_op))
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

        """
        # Compute spin correlation between first two particles (on vectors evolved using full operator)
        (correlations_full_op_real, correlations_full_op_imag) = generators.correlation_in_reshaped_set(evolved_test_vectors_full_op_reshaped, S_corr)

        # Compute spin correlation between first two particles (on vectors evolved using simple operator)
        # (correlations_simple_op_real, correlations_simple_op_imag) = generators.correlation_in_reshaped_set(evolved_test_vectors_simple_op_reshaped, S_corr)

        # Compute spin correlation between first two particles (on vectors evolved using neural network)
        (correlations_nn_real, correlations_nn_imag) = generators.correlation_in_reshaped_set(predicted_test_vectors, S_corr)
        # y_mean_correlations_real.append([np.mean(correlations_full_op_real), np.mean(correlations_simple_op_real), np.mean(correlations_nn_real)])
        # y_mean_correlations_imag.append([np.mean(correlations_full_op_imag), np.mean(correlations_simple_op_imag), np.mean(correlations_nn_imag)])
        y_mean_correlations_real.append([np.mean(correlations_full_op_real), np.mean(correlations_nn_real)])
        y_mean_correlations_imag.append([np.mean(correlations_full_op_imag), np.mean(correlations_nn_imag)])
        """
    print("Finished! Current timestep: ", t_total, " / ", t_total)

    y_overlap.append(mean_overlap_per_timestep)

    y_coefficients_real = np.array(y_coefficients_real)
    y_coefficients_imag = np.array(y_coefficients_imag)

    #
    # Create catalog to save all plots
    #
    dirName = "6_layers_simple_norm_only_at_end_train_samples=" + str(train_num_samples) + "_timestep=" + str(t) + "_t_total=" + str(t_total)  + "_optimizer=" + optimizer + "_loss=" + loss + "_activation=" + activation

    # Create target directory if doesn't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
        os.mkdir(dirName+"/Correlations")
        print("Directory " , dirName+"/Correlations" ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")

    # Save meta information about the accuracy of learning of current network
    file = open(dirName + "/Meta_information_N=" + str(N) + ".txt","w+")
    file.write("Type: 6 layers (simple) normalization ONLY after last layer (NOT ENCODER-DECODER)\n")
    file.write("Optimizer: " + optimizer + "\n")
    file.write("Loss function: " + loss + "\n")
    file.write("Activation: " + activation + "\n")
    file.write("N: " + str(N) + "\n")
    file.write("Timestep: " + str(t) + ", total evolution time: " + str(t_total) + "\n")
    file.write("Size of the training set: " + str(train_num_samples) + "\n")
    file.write("Size of the testing set: " + str(test_num_samples) + "\n")
    file.write("Test score: " + str(score) + "\n")
    file.write("Test accuracy: " + str(acc) + "\n")
    file.close()


    for pair in pairs:
        (N0, N1) = pair
        title = "Mean correlation (real) - 6 layers (simple) normalization ONLY after last layer \n"
        title += "N0 = " + str(N0) + ", N1 = " + str(N1) + "\n"
        title += "N: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation
        plt.title(title)
        plt.plot(x, y_mean_correlations[N0,N1]["real"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dirName + "/Correlations/Corr_N0=" + str(N0) + "_N1=" + str(N1) + "_(real).png")
        plt.clf()

        title = "Mean correlation (imag) - 6 layers (simple) normalization ONLY after last layer \n"
        title += "N0 = " + str(N0) + ", N1 = " + str(N1) + "\n"
        title += "N: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation
        plt.title(title)
        plt.plot(x, y_mean_correlations[N0,N1]["imag"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dirName + "/Correlations/Corr_N0=" + str(N0) + "_N1=" + str(N1) + "_(imag).png")
        plt.clf()

    """
    # Plot real part of the coefficient
    plt.title("Coefficient (real) - 6 layers (simple) normalization ONLY after last layer \nN: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation)
    plt.plot(x, y_coefficients_real, '.-')
    plt.xlabel('Time')
    # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
    plt.gca().legend(["exp(-iHt)", "NN"])
    plt.grid()
    plt.savefig(dirName + "/Coeff_N=" + str(N) + "_(real).png")
    plt.clf()
    # Plot imaginary part of the coefficient
    plt.title("Coefficient (imaginary) - 6 layers (simple) normalization ONLY after last layer \nN: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation)
    plt.plot(x, y_coefficients_imag, '.-')
    plt.xlabel('Time')
    # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
    plt.gca().legend(["exp(-iHt)", "NN"])
    plt.grid()
    plt.savefig(dirName + "/Coeff_N=" + str(N) + "_(imag).png")
    plt.clf()

    plt.title("Mean correlation (real) - 6 layers (simple) normalization ONLY after last layer \nN: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation)
    plt.plot(x, y_mean_correlations_real, '.-')
    plt.xlabel('Time')
    plt.ylabel('Mean correlation')
    # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
    plt.gca().legend(["exp(-iHt)", "NN"])
    plt.xlim(xmin=0, xmax=x_max)
    plt.grid()
    plt.savefig(dirName + "/Corr_N=" + str(N) + "_(real).png")
    plt.clf()

    plt.title("Mean correlation (imaginary) - 6 layers (simple) normalization ONLY after last layer \nN: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation)
    plt.plot(x, y_mean_correlations_imag, '.-')
    plt.xlabel('Time')
    plt.ylabel('Mean correlation')
    # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
    plt.gca().legend(["exp(-iHt)", "NN"])
    plt.xlim(xmin=0, xmax=x_max)
    plt.grid()
    plt.savefig(dirName + "/Corr_N=" + str(N) + "_(imag).png")
    plt.clf()
    """

# Reshape data so that it has appropriate dimensions
y_overlap = np.array(y_overlap).T.tolist()
plt.title("Overlap - 6 layers (simple) normalization ONLY after last layer \ntimestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation)
plt.plot(x, y_overlap, '.-')
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.gca().legend(legend)
plt.xlim(xmin=0, xmax=x_max)
plt.grid()
plt.savefig(dirName + "/Overlap.png")
