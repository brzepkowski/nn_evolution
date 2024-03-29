# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

################################################################################
# In this version of the network's structure normalization is added only at
# the end of the network (not inbetween hidden layers).
################################################################################
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.backend import l2_normalize
import numpy as np
import matplotlib.pyplot as plt
from ..tools.misc import overlap_of_reshaped_sets, energies_of_reshaped_set
from ..tools.misc import entanglements_of_reshaped_set, correlation_in_reshaped_set
from numpy.linalg import norm
from itertools import combinations
from math import floor
import sys
import os.path
from os import path

def my_cost(y_true, y_pred):
    y_true_real = y_true[:,::2] # These are real values
    y_true_imag = y_true[:,1::2] # These are imaginary numbers

    y_pred_real = y_pred[:,::2] # These are real values
    y_pred_imag = y_pred[:,1::2] # These are imaginary numbers

    first = tf.math.multiply(y_true_real, y_pred_real)
    second = tf.math.multiply(y_true_real, y_pred_imag)
    third = tf.math.multiply(y_true_imag, y_pred_real)
    fourth = tf.math.multiply(y_true_imag, y_pred_imag)

    reals = first + fourth
    imags = third - second

    sum_of_reals = tf.math.reduce_sum(reals, axis=1)
    sum_of_imags = tf.math.reduce_sum(imags, axis=1)

    overlap_squared = (sum_of_reals*sum_of_reals) + (sum_of_imags*sum_of_imags)

    # Optimize only overlap^2
    result = 1.0-tf.reduce_mean(overlap_squared)
    # Optimize both overlap^2 and log_cosh
    # result = 1.0-tf.reduce_mean(overlap_squared) + tf.keras.losses.log_cosh(y_true, y_pred)
    # Optimize just log_cosh
    # result = tf.keras.losses.log_cosh(y_true, y_pred)

    return result


def test_evol_compress(activation, N, dt, t_max, subtracted_neurons):

    ################################################################################
    # Load the training, validation and testing sets and input to the neural network
    ################################################################################

    dir_name = "data_N=" + str(N) + "_dt=" + str(dt) + "_t_max=" + str(t_max)

    with open(dir_name + '/test_input.npy', 'rb') as f:
        reshaped_test_input_vectors = np.load(f)
    with open(dir_name + '/test_output.npy', 'rb') as f:
        reshaped_test_output_vectors = np.load(f)
    with open(dir_name + '/exact_evol_output.npy', 'rb') as f:
        exact_evol_output = np.load(f)
    with open(dir_name + '/test_eigenvalues.npy', 'rb') as f:
        test_eigenvalues = np.load(f)
    with open(dir_name + '/nn_input_for_predictions.npy', 'rb') as f:
        nn_predicted_vectors = np.load(f)

    ################################################################################
    # Load hamiltonian H
    ################################################################################

    with open(dir_name + '/H.npy', 'rb') as f:
        H = np.load(f)

    ################################################################################
    # Load model
    ################################################################################

    parent_dir_name = dir_name
    # Change the directory to the folder, in which we will be storing all results
    dir_name += "/nn_evol_compress_activation=" + activation + "_subtr=" + str(subtracted_neurons)

    best_model_filename = dir_name + '/best_model.h5'

    ################################################################################
    #  Create folder to store all results
    ################################################################################

    # Create target directory if doesn't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name, " created!")

    ################################################################################
    # Load  model
    ################################################################################

    # Check if the model already exists. If not, create appropriate structure
    if path.exists(best_model_filename):
        model = load_model(best_model_filename, custom_objects={"l2_normalize": l2_normalize, "my_cost": my_cost})
        print("\n###############################")
        print("### Model loaded from file! ###")
        print("###############################\n")
    else:
        print("Error: model not trained yet!")
        sys.exit()

    ################################################################################
    # Compare predictions given by model with the exact evolution
    ################################################################################
    x = []
    for i in range(int(t_max/dt)):
        x.append((i+1)*dt)
    x_max = x[-1]
    mean_overlap_per_timestep_eigvecs_H = []

    # Below variables correspond to all coefficients of ONLY first vector in the test set
    y_coefficients_eigvec_H_real = []
    y_coefficients_eigvec_H_imag = []

    reshaped_vector_length = np.shape(reshaped_test_input_vectors[0])[0]
    for i in range(int(reshaped_vector_length/2)):
        y_coefficients_eigvec_H_real.append([])
        y_coefficients_eigvec_H_imag.append([])

    y_correlations_eigvec_H = {}

    # List of all possible pairs in the chain (needed for calculation of correlations)
    pairs = list(combinations(range(N), 2))

    # Set the number of eigenvectors, so that we can cut out sections of appropriate size
    # from the testing set
    batch_size = floor(len(reshaped_test_output_vectors) / len(x))

    # Batch iterator will enable cuting out appropriate slices from the testing set
    batch_iterator = 0

    all_overlaps = []
    all_predicted_energies = []
    all_predicted_entanglements = []

    for time_step in x:
        print("Current timestep: ", round(time_step, 5), " / ", round(x_max, 2), end="\r")

        # Evolution using full operator
        batch_beginning = batch_iterator*batch_size
        batch_ending = (batch_iterator+1)*batch_size

        evolved_test_vectors_full_op_reshaped = reshaped_test_output_vectors[batch_beginning:batch_ending]

        # Evolution using neural network
        nn_predicted_vectors = model.predict(nn_predicted_vectors)

        # print("Cost: ", my_cost(evolved_test_vectors_full_op_reshaped, nn_predicted_vectors))

        # Compute overlap between precisely evolved vector and the one returned by neural network
        overlaps = overlap_of_reshaped_sets(evolved_test_vectors_full_op_reshaped, nn_predicted_vectors)
        all_overlaps.append(overlaps)

        predicted_energies = energies_of_reshaped_set(nn_predicted_vectors, H)

        predicted_entanglements = entanglements_of_reshaped_set(nn_predicted_vectors, N)

        all_predicted_energies.append(predicted_energies)
        all_predicted_entanglements.append(predicted_entanglements)

        # Save mean overlap with all eigenvectors of H_quenched in testing set. Note,
        # that in this version of the code our testing set consists only of eigenvectors of HQ!
        mean_overlap_per_timestep_eigvecs_H.append(abs(np.mean(overlaps)))

        """
        # Save real parts of the first evolved vector
        for i in range(0, reshaped_vector_length, 2):
            y_coefficients_eigvec_H_real[floor(i/2)].append([evolved_test_vectors_full_op_reshaped[0][i], nn_predicted_vectors[0][i]])

        # Save imaginary parts of the first evolved vector
        for i in range(1, reshaped_vector_length, 2):
            y_coefficients_eigvec_H_imag[floor(i/2)].append([evolved_test_vectors_full_op_reshaped[0][i], nn_predicted_vectors[0][i]])
        """
        ############################################################################
        # Save values of correlation between all pairs of particles
        ############################################################################

        """
        # for N0 in range(N-1):
        N0 = 0
        for N1 in range(N0+1, N):

            # Load spin correlation operator from file
            S_corr_filename = parent_dir_name + '/S_corr_' + str(N0) + '_' + str(N1) + '.npy'
            with open(S_corr_filename, 'rb') as f:
                S_corr = np.load(f)

            # Compute spin correlation between the two particles (on vectors evolved using full operator)
            (correlations_full_op_real, correlations_full_op_imag) = correlation_in_reshaped_set(evolved_test_vectors_full_op_reshaped, S_corr)

            # Compute spin correlation between first two particles (on vectors evolved using neural network)
            (correlations_nn_real, correlations_nn_imag) = correlation_in_reshaped_set(nn_predicted_vectors, S_corr)

            # Save correlations in the evolved eigenvector of H_quenched
            if (N0,N1) not in y_correlations_eigvec_H:
                y_correlations_eigvec_H[N0, N1] = {"real":[], "imag":[]}
            y_correlations_eigvec_H[N0, N1]["real"].append([correlations_full_op_real[0], correlations_nn_real[0]])
            y_correlations_eigvec_H[N0, N1]["imag"].append([correlations_full_op_imag[0], correlations_nn_imag[0]])
        """

        # Increase the batch_iterator, so that appropriate slice is cut out from the
        # testing set in the next iteration of the loop
        batch_iterator += 1
    print("\nFinished!")

    all_predicted_energies = np.stack(all_predicted_energies)
    all_predicted_energies = all_predicted_energies.T

    all_predicted_entanglements = np.stack(all_predicted_entanglements)
    all_predicted_entanglements = all_predicted_entanglements.T

    all_overlaps = np.stack(all_overlaps)
    all_overlaps = all_overlaps.T

    y_coefficients_eigvec_H_real = np.array(y_coefficients_eigvec_H_real)
    y_coefficients_eigvec_H_imag = np.array(y_coefficients_eigvec_H_imag)

    ################################################################################
    # Create appropriate catalogs to save all plots
    ################################################################################

    # Create target directory if it doesn't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name,  " created")


    # Create appropriate folders to store correlations
    if not os.path.exists(dir_name+"/Correlations"):
        os.mkdir(dir_name+"/Correlations")
        print("Directory " , dir_name+"/Correlations" ,  " created")

    # Create appropriate folders to store coefficients
    if not os.path.exists(dir_name+"/Coefficients"):
        os.mkdir(dir_name+"/Coefficients")
        print("Directory " , dir_name+"/Coefficients" ,  " created")


    ################################################################################
    # Save precise values of overlaps, energies and entanglements
    ################################################################################

    print("Saving precise values of overlaps...", end="")
    np.savetxt(dir_name + "/overlaps.csv", all_overlaps, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving precise values of energies...", end="")
    np.savetxt(dir_name + "/predicted_energies.csv", all_predicted_energies, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving precise values of entanglement entropies...", end="")
    np.savetxt(dir_name + "/predicted_entanglements.csv", all_predicted_entanglements, delimiter=",", fmt='%f')
    print(" done!")

    ################################################################################
    # Save plots of evolved correlations between particles in the first eigenvector of H_quenched
    ################################################################################

    """
    print("Saving plots of correlations...", end="")
    # for N0 in range(N-1):
    N0 = 0
    for N1 in range(N0+1, N):
        title = "CORRELATION (REAL), "
        title += "N0 = " + str(N0) + ", N1 = " + str(N1)
        plt.title(title)
        plt.plot(x, y_correlations_eigvec_H[N0,N1]["real"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dir_name + "/Correlations/Corr_n0=" + str(N0) + "_n1=" + str(N1) + "_(real).png")
        plt.close()

        # WARNING: Below plot might be discarded, because it should give values close to 0. It might be left just to check,
        # if correlations are truly real numbers
        title = "CORRELATION (IMAG), "
        title += "N0 = " + str(N0) + ", N1 = " + str(N1)
        plt.title(title)
        plt.plot(x, y_correlations_eigvec_H[N0,N1]["imag"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dir_name + "/Correlations/Corr_n0=" + str(N0) + "_n1=" + str(N1) + "_(imag).png")
        plt.close()
    print(" done!")
    """

    ################################################################################
    # Save plots of evolved coefficients in the first eigenvector of H_quenched
    ################################################################################
    """
    print("Saving plots of coefficients...", end="")
    for i in range(int(reshaped_vector_length/2)):
        # Plot real part of the coefficients in the eigenvector of H_quenched
        plt.title("COEFFICIENT " + str(i) + " (REAL)")
        plt.plot(x, y_coefficients_eigvec_H_real[i], '.-')
        plt.xlabel('Time')
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.grid()
        plt.savefig(dir_name + "/Coefficients/Coeff_n=" + str(i) + "_(real).png")
        plt.close()

        # Plot imaginary part of the coefficients in the eigenvector of H_quenched
        plt.title("COEFFICIENT " + str(i) + " (IMAG)")
        plt.plot(x, y_coefficients_eigvec_H_imag[i], '.-')
        plt.xlabel('Time')
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.grid()
        plt.savefig(dir_name + "/Coefficients/Coeff_n=" + str(i) + "_(imag).png")
        plt.close()
    print(" done!")
    """
    ################################################################################
    # Save plots of correlations between particles in the first eigenvector of H_quenched
    ################################################################################

    print("Saving plot of overlaps...", end="")
    plt.title("Mean overlap")
    plt.plot(x, mean_overlap_per_timestep_eigvecs_H, '.-')
    plt.xlabel('Time')
    plt.ylabel('Overlap')
    plt.xlim(xmin=0, xmax=x_max)
    plt.grid()
    plt.savefig(dir_name + "/mean_overlap.png")
    plt.close()
    print(" done!")
