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
from math import floor

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

legend = []
for N in range(N_min, N_max+1):
    y_correlations_real = []
    y_correlations_imag = []
    print("#################################")
    print("###### N: ", N)
    print("#################################")
    legend.append(str(N))
    H = generators.generate_Heisenberg_hamiltonian(N, J=1.0, h=0, periodic=False)
    U_t = expm(-1j * H * t)
    # U_t_simple = np.identity(2**N) - (1j * H * t)

    # List of all possible pairs in the chain (needed for calculation of correlations)
    pairs = list(combinations(range(N), 2))

    # Generate random vectors, that will go to the training and testing sets
    (training_input_vectors, training_output_vectors) = generators.generate_training_set_many_evolution_steps(train_num_samples, 2**N, H, t, t_total)
    (test_input_vectors, test_output_vectors) = generators.generate_test_set_many_evolution_steps(test_num_samples, 2**N, H, t, t_total)

    # Generate eigenvectors of H, that will go to the training and testing sets
    _, eigenvectors = eig(H) # Here each eigenvector is represented by a column in a matrix
    eigenvectors = eigenvectors.T # By conducting transposition each eigenvector is stored in a separate row
    (input_eigenvectors, output_eigenvectors) = generators.evolve_whole_set_many_evolution_steps(eigenvectors, U_t, t, t_total)
    eigenvectors_train_set_size = floor(np.shape(input_eigenvectors)[0]*3/4)

    training_input_eigenvectors = input_eigenvectors[:eigenvectors_train_set_size]
    training_output_eigenvectors = output_eigenvectors[:eigenvectors_train_set_size]

    test_input_eigenvectors = input_eigenvectors[eigenvectors_train_set_size:]
    test_output_eigenvectors = output_eigenvectors[eigenvectors_train_set_size:]

    # Concatenate two sets creating as a result training and testing sets
    training_input_vectors = np.concatenate((training_input_vectors, training_input_eigenvectors))
    training_output_vectors = np.concatenate((training_output_vectors,training_output_eigenvectors))
    test_input_vectors = np.concatenate((test_input_vectors, test_input_eigenvectors))
    test_output_vectors = np.concatenate((test_output_vectors, test_output_eigenvectors))

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
    y_overlap_all = []
    y_overlap_eigvecs = []
    ############################################################################

    # WARNING: We are changing these random vectors to eigenvectors of the hamiltonian
    (test_input_random_vectors, test_output_random_vectors) = generators.generate_test_set(test_num_samples, 2**N, H, t)
    (reshaped_test_input_random_vectors, _) = generators.reshape_two_sets(test_input_random_vectors, test_output_random_vectors)
    reshaped_test_input_random_vectors = generators.flatten_set(reshaped_test_input_random_vectors)

    _, eigenvectors = eig(H) # Here each eigenvector is represented by a column in a matrix
    eigenvectors = eigenvectors.T # By conducting transposition each eigenvector is stored in a separate row
    reshaped_test_input_eigenvectors = generators.reshape_set(eigenvectors)
    reshaped_test_input_eigenvectors = generators.flatten_set(reshaped_test_input_eigenvectors)

    # To access egeinvectors of Hamiltonian H in a more readable way we are giving up on the performance,
    # by appending longer list to a shorter one.
    evolved_test_vectors_full_op = list(eigenvectors)
    for i in range(len(test_input_random_vectors)):
        evolved_test_vectors_full_op.append(test_input_random_vectors[i])
    evolved_test_vectors_full_op = np.array(evolved_test_vectors_full_op)

    # We are doing similar thing as in the above block, to initialize vectors, that can
    # be evolved by a neural network
    predicted_test_vectors = list(reshaped_test_input_eigenvectors)
    for i in range(len(reshaped_test_input_random_vectors)):
        predicted_test_vectors.append(reshaped_test_input_random_vectors[i])
    predicted_test_vectors = np.array(predicted_test_vectors)

    mean_overlap_per_timestep_all = []
    mean_overlap_per_timestep_eigvecs = []

    eigenvecs_number = len(eigenvectors)
    y_all_coefficients_eigvec_real = []
    y_all_coefficients_eigvec_imag = []
    y_all_coefficients_random_vect_real = []
    y_all_coefficients_random_vect_imag = []

    reshaped_vector_length = np.shape(reshaped_test_input_eigenvectors[0])[0]
    for i in range(int(reshaped_vector_length/2)):
        y_all_coefficients_eigvec_real.append([])
        y_all_coefficients_eigvec_imag.append([])
        y_all_coefficients_random_vect_real.append([])
        y_all_coefficients_random_vect_imag.append([])

    y_correlations_eigvec = {}
    y_correlations_random_vect = {}

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

        # Save mean overlap with all vectors in testing set
        mean_overlap_per_timestep_all.append(np.mean(overlaps))
        # Save mean overlap with all eigenvectors of H in testing set
        mean_overlap_per_timestep_eigvecs.append(np.mean(overlaps[:eigenvecs_number]))

        # Save real parts of evolved vectors (eigenvector and random one)
        for i in range(0, reshaped_vector_length, 2):
            y_all_coefficients_eigvec_real[int(i/2)].append([evolved_test_vectors_full_op_reshaped[0][i], predicted_test_vectors[0][i]])
            y_all_coefficients_random_vect_real[int(i/2)].append([evolved_test_vectors_full_op_reshaped[eigenvecs_number][i], predicted_test_vectors[eigenvecs_number][i]])

        # Save imaginary parts of evolved vectors (eigenvector and random one)
        for i in range(1, reshaped_vector_length, 2):
            y_all_coefficients_eigvec_imag[floor(i/2)].append([evolved_test_vectors_full_op_reshaped[0][i], predicted_test_vectors[0][i]])
            y_all_coefficients_random_vect_imag[floor(i/2)].append([evolved_test_vectors_full_op_reshaped[eigenvecs_number][i], predicted_test_vectors[eigenvecs_number][i]])

        #
        # All possible correlations
        #

        for pair in pairs:
            (N0, N1) = pair
            S_corr = generators.generate_correlation_operator(N0, N1, N)

            # Compute spin correlation between the two particles (on vectors evolved using full operator)
            (correlations_full_op_real, correlations_full_op_imag) = generators.correlation_in_reshaped_set(evolved_test_vectors_full_op_reshaped, S_corr)

            # Compute spin correlation between first two particles (on vectors evolved using neural network)
            (correlations_nn_real, correlations_nn_imag) = generators.correlation_in_reshaped_set(predicted_test_vectors, S_corr)

            # Save correlations in the evolved eigenvector of H
            if (N0,N1) not in y_correlations_eigvec:
                y_correlations_eigvec[N0, N1] = {"real":[], "imag":[]}
            y_correlations_eigvec[N0, N1]["real"].append([correlations_full_op_real[0], correlations_nn_real[0]])
            y_correlations_eigvec[N0, N1]["imag"].append([correlations_full_op_imag[0], correlations_nn_imag[0]])

            # Save correlations in the evolved random vector
            if (N0,N1) not in y_correlations_random_vect:
                y_correlations_random_vect[N0, N1] = {"real":[], "imag":[]}
            y_correlations_random_vect[N0, N1]["real"].append([correlations_full_op_real[eigenvecs_number], correlations_nn_real[eigenvecs_number]])
            y_correlations_random_vect[N0, N1]["imag"].append([correlations_full_op_imag[eigenvecs_number], correlations_nn_imag[eigenvecs_number]])

    print("Finished! Current timestep: ", t_total, " / ", t_total)

    y_overlap_all.append(mean_overlap_per_timestep_all)
    y_overlap_eigvecs.append(mean_overlap_per_timestep_eigvecs)

    y_all_coefficients_eigvec_real = np.array(y_all_coefficients_eigvec_real)
    y_all_coefficients_eigvec_imag = np.array(y_all_coefficients_eigvec_imag)

    y_all_coefficients_random_vect_real = np.array(y_all_coefficients_random_vect_real)
    y_all_coefficients_random_vect_imag = np.array(y_all_coefficients_random_vect_imag)

    #
    # Create catalog to save all plots
    #
    dirName = "normal_H_rand_H=" + str(train_num_samples) + "_timestep=" + str(t) + "_t_total=" + str(t_total)  + "_optimizer=" + optimizer + "_loss=" + loss + "_activation=" + activation

    # Create target directory if doesn't exist
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
        # Create appropriate folders to store correlations
        os.mkdir(dirName+"/Correlations")
        print("Directory " , dirName+"/Correlations" ,  " Created ")
        os.mkdir(dirName+"/Correlations/Eigenvector")
        print("Directory " , dirName+"/Correlations/Eigenvector" ,  " Created ")
        os.mkdir(dirName+"/Correlations/Random_vector")
        print("Directory " , dirName+"/Correlations/Random_vector" ,  " Created ")
        # Create appropriate folders to store coefficients
        os.mkdir(dirName+"/Coefficients")
        print("Directory " , dirName+"/Coefficients" ,  " Created ")
        os.mkdir(dirName+"/Coefficients/Eigenvector")
        print("Directory " , dirName+"/Coefficients/Eigenvector" ,  " Created ")
        os.mkdir(dirName+"/Coefficients/Random_vector")
        print("Directory " , dirName+"/Coefficients/Random_vector" ,  " Created ")
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

    # Save correlations in eigenvector
    for pair in pairs:
        (N0, N1) = pair
        # title = "Correlation (real) - 6 layers (simple) normalization ONLY after last layer \n"
        title = "EIGENVECTOR\nCORRELATION (REAL), "
        title += "N0 = " + str(N0) + ", N1 = " + str(N1)
        # title += "N: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation
        plt.title(title)
        plt.plot(x, y_correlations_eigvec[N0,N1]["real"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dirName + "/Correlations/Eigenvector/Corr_N0=" + str(N0) + "_N1=" + str(N1) + "_(real).png")
        plt.clf()

        # WARNING: Below plot might be discarded, because it should give values close to 0. It might be left just to check,
        # if correlations are truly real numbers
        # title = "Correlation (imag) - 6 layers (simple) normalization ONLY after last layer \n"
        title = "EIGENVECTOR\nCORRELATION (IMAG), "
        title += "N0 = " + str(N0) + ", N1 = " + str(N1)
        # title += "N: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation
        plt.title(title)
        plt.plot(x, y_correlations_eigvec[N0,N1]["imag"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dirName + "/Correlations/Eigenvector/Corr_N0=" + str(N0) + "_N1=" + str(N1) + "_(imag).png")
        plt.clf()

    # Save correlations in random vector
    for pair in pairs:
        (N0, N1) = pair
        # title = "Correlation (real) - 6 layers (simple) normalization ONLY after last layer \n"
        title = "RANDOM VECTOR\nCORRELATION (REAL), "
        title += "N0 = " + str(N0) + ", N1 = " + str(N1)
        # title += "N: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation
        plt.title(title)
        plt.plot(x, y_correlations_random_vect[N0,N1]["real"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dirName + "/Correlations/Random_vector/Corr_N0=" + str(N0) + "_N1=" + str(N1) + "_(real).png")
        plt.clf()

        # WARNING: Below plot might be discarded, because it should give values close to 0. It might be left just to check,
        # if correlations are truly real numbers
        # title = "Correlation (imag) - 6 layers (simple) normalization ONLY after last layer \n"
        title = "RANDOM VECTOR\nCORRELATION (IMAG), "
        title += "N0 = " + str(N0) + ", N1 = " + str(N1)
        # title += "N: " + str(N) + " timestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation
        plt.title(title)
        plt.plot(x, y_correlations_random_vect[N0,N1]["imag"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dirName + "/Correlations/Random_vector/Corr_N0=" + str(N0) + "_N1=" + str(N1) + "_(imag).png")
        plt.clf()



    for i in range(int(reshaped_vector_length/2)):
        # Plot real part of the coefficients in the eigenvector of H
        plt.title("EIGENVECTOR\nCOEFFICIENT " + str(i) + " (REAL)")
        plt.plot(x, y_all_coefficients_eigvec_real[i], '.-')
        plt.xlabel('Time')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.grid()
        plt.savefig(dirName + "/Coefficients/Eigenvector/Coeff_N=" + str(i) + "_(real).png")
        plt.clf()

        # Plot imaginary part of the coefficients in the eigenvector of H
        plt.title("EIGENVECTOR\nCOEFFICIENT " + str(i) + " (IMAG)")
        plt.plot(x, y_all_coefficients_eigvec_imag[i], '.-')
        plt.xlabel('Time')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.grid()
        plt.savefig(dirName + "/Coefficients/Eigenvector/Coeff_N=" + str(i) + "_(imag).png")
        plt.clf()

        # Plot real part of the coefficients in random vector
        plt.title("RANDOM VECTOR\nCOEFFICIENT " + str(i) + " (REAL)")
        plt.plot(x, y_all_coefficients_random_vect_real[i], '.-')
        plt.xlabel('Time')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.grid()
        plt.savefig(dirName + "/Coefficients/Random_vector/Coeff_N=" + str(i) + "_(real).png")
        plt.clf()

        # Plot real part of the coefficients in random vector
        plt.title("RANDOM VECTOR\nCOEFFICIENT " + str(i) + " (IMAG)")
        plt.plot(x, y_all_coefficients_random_vect_imag[i], '.-')
        plt.xlabel('Time')
        # plt.gca().legend(["exp(-iHt)", "I - iHt", "NN"])
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.grid()
        plt.savefig(dirName + "/Coefficients/Random_vector/Coeff_N=" + str(i) + "_(imag).png")
        plt.clf()

# Reshape data so that it has appropriate dimensions
y_overlap_all = np.array(y_overlap_all).T.tolist()
# plt.title("Overlap - 6 layers (simple) normalization ONLY after last layer \ntimestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation)
plt.title("OVERLAP \nALL VECTORS")
plt.plot(x, y_overlap_all, '.-')
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.gca().legend(legend)
plt.xlim(xmin=0, xmax=x_max)
plt.grid()
plt.savefig(dirName + "/Overlap_all.png")
plt.clf()

# Reshape data so that it has appropriate dimensions
y_overlap_eigvecs = np.array(y_overlap_eigvecs).T.tolist()
# plt.title("Overlap - 6 layers (simple) normalization ONLY after last layer \ntimestep: " + str(t) + " training set size: " + str(train_num_samples) + "\noptimizer: " + optimizer + " loss: " + loss + " activation: " + activation)
plt.title("OVERLAP \nEIGENVECTORS OF H")
plt.plot(x, y_overlap_eigvecs, '.-')
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.gca().legend(legend)
plt.xlim(xmin=0, xmax=x_max)
plt.grid()
plt.savefig(dirName + "/Overlap_H.png")
plt.clf()
