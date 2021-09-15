################################################################################
# In this version of the network's structure normalization is added only at
# the end of the network (not inbetween hidden layers).
################################################################################
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import misc
from misc import load_array_from_file, load_vector_from_file
from numpy.linalg import norm
from itertools import combinations
from math import floor
import sys
import os.path
from os import path
# For testing
from scipy.linalg import expm
from numpy.linalg import eig


if len(sys.argv) != 5:
    print("Wrong number of arguments provided. Please give them in the following fashion:")
    print("- activaton function,")
    print("- N,")
    print("- timestep,")
    print("- total_evolution_time.")
    sys.exit()

activation = sys.argv[1]
N = int(sys.argv[2])
t = float(sys.argv[3])
t_total = float(sys.argv[4])

def my_cost(y_true, y_pred):
    y_true_even = y_true[:,::2] # These are real values
    y_true_odd = y_true[:,1::2] # These are imaginary numbers

    y_pred_even = y_pred[:,::2] # These are real values
    y_pred_odd = y_pred[:,1::2] # These are imaginary numbers

    real_parts = tf.math.multiply(y_true_even,y_pred_even)
    imag_parts = tf.math.multiply(y_true_odd,y_pred_odd)

    sum_of_reals = tf.math.reduce_sum(real_parts, axis=1)
    sum_of_imags = tf.math.reduce_sum(imag_parts, axis=1)

    result = sum_of_reals + sum_of_imags
    # result = tf.square(result)
    result = 1.0-tf.reduce_mean(result)

    return result

################################################################################
# Load the training, validation and test sets. Also load the eigenvectors of H & HQ,
# and input to the neural network, on the basis of which it will make predictions
################################################################################

with open('test_input.npy', 'rb') as f:
    reshaped_test_input_vectors = np.load(f)
with open('test_output.npy', 'rb') as f:
    reshaped_test_output_vectors = np.load(f)
# with open('test_input_simple_op.npy', 'rb') as f:
#     reshaped_test_input_vectors_simple_op = np.load(f)
# with open('test_output_simple_op.npy', 'rb') as f:
#     reshaped_test_output_vectors_simple_op = np.load(f)
with open('test_eigenvalues.npy', 'rb') as f:
    test_eigenvalues = np.load(f)
# Load data for making predictions
with open('nn_input_for_predictions.npy', 'rb') as f:
    nn_predicted_vectors = np.load(f)
# Load eigenvalues of H
eigenvalues_H = load_vector_from_file("eigenvalues_H.dat", 2**N).astype(np.float64)

################################################################################
# Load hamiltonian HQ
################################################################################

HQ_filename = 'HQ.dat'
if not path.exists(HQ_filename):
    print("Hamilotnian HQ hasn't been generated yet!")
    sys.exit()
else:
    print("Loading hamiltonian HQ... ", end="", flush=True)
    HQ = load_array_from_file(HQ_filename, 2**N).astype(np.float64) # Here each eigenvector is represented by a column in a matrix
    print("Done!")

# ******************************************************************************
# Load hamiltonian H
# ******************************************************************************

H_filename = 'H.dat'
if not path.exists(H_filename):
    print("Hamilotnian H hasn't been generated yet!")
    sys.exit()
else:
    print("Loading hamiltonian H... ", end="", flush=True)
    H = load_array_from_file(H_filename, 2**N).astype(np.float64) # Here each eigenvector is represented by a column in a matrix
    print("Done!")

################################################################################
# Load/create the model
################################################################################

# Change the directory to the folder, in which we will be storing all results from
# this network's structure (model itself and all plots)
dir_name = "simple_activation=" + activation + "_N=" + str(N) + "_t=" + str(t) + "_t_total=" + str(t_total)

# model_filename = dir_name + '/model'
best_model_filename = dir_name + '/best_model.h5'

################################################################################
#  Create folder to store all results
################################################################################
# Create target directory if doesn't exist
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    print("Directory ", dir_name, " created!")

################################################################################
# Train model
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
for i in range(int(t_total/t)):
    x.append((i+1)*t)
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

# Under this variable we will be storing all overlaps. Although, the first entry consists of
# ENERGY VALUES (eigenvalues), thanks to which it will be possible to associate further values of overlap
# with specific vectors from the spectrum.
all_overlaps = [test_eigenvalues]
all_exact_energies = []
all_predicted_energies = []

# ******************************************************************************
all_exact_energies_BQ = [] # BQ = before quench
all_predicted_energies_BQ = []
# ******************************************************************************

for time_step in x:
    print("Current timestep: ", round(time_step, 5), " / ", round(x_max, 2), end="\r")

    # Evolution using full operator
    batch_beginning = batch_iterator*batch_size
    batch_ending = (batch_iterator+1)*batch_size

    evolved_test_vectors_full_op_reshaped = reshaped_test_output_vectors[batch_beginning:batch_ending]

    # # Evolution using simple operator
    # evolved_test_vectors_simple_op_reshaped = reshaped_test_output_vectors_simple_op[batch_beginning:batch_ending]
    # for vector in evolved_test_vectors_simple_op_reshaped:
    #     vector /= norm(vector)

    # for i in range(len(evolved_test_vectors_simple_op_reshaped)):
    #     print("Full: \n", evolved_test_vectors_full_op_reshaped[i])
    #     print("Simp: \n", evolved_test_vectors_simple_op_reshaped[i])

    # **************************************************************************
    exact_energies_BQ = misc.energies_of_reshaped_set(reshaped_test_input_vectors[batch_beginning:batch_ending], H)
    predicted_energies_BQ = misc.energies_of_reshaped_set(nn_predicted_vectors, H)

    all_exact_energies_BQ.append(exact_energies_BQ)
    all_predicted_energies_BQ.append(predicted_energies_BQ)
    # **************************************************************************


    # Evolution using neural network
    nn_predicted_vectors = model.predict(nn_predicted_vectors)

    # print("Cost: ", my_cost(evolved_test_vectors_full_op_reshaped, nn_predicted_vectors))

    # # Compute overlap between precisely evolved vector and one evolved using U_t_simple
    # overlaps_simple_op = misc.overlap_of_reshaped_sets(evolved_test_vectors_full_op_reshaped, evolved_test_vectors_simple_op_reshaped)

    # Compute overlap between precisely evolved vector and the one returned by neural network
    overlaps = misc.overlap_of_reshaped_sets(evolved_test_vectors_full_op_reshaped, nn_predicted_vectors)
    all_overlaps.append(overlaps)

    exact_energies = misc.energies_of_reshaped_set(evolved_test_vectors_full_op_reshaped, HQ)
    predicted_energies = misc.energies_of_reshaped_set(nn_predicted_vectors, HQ)

    all_exact_energies.append(exact_energies)
    all_predicted_energies.append(predicted_energies)

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


    # for N0 in range(N-1):
    N0 = 0
    for N1 in range(N0+1, N):

        # Load spin correlation operator from file
        S_corr_filename = 'S_corr_' + str(N0) + '_' + str(N1) + '.npy'
        with open(S_corr_filename, 'rb') as f:
            S_corr = np.load(f)

        # Compute spin correlation between the two particles (on vectors evolved using full operator)
        (correlations_full_op_real, correlations_full_op_imag) = misc.correlation_in_reshaped_set(evolved_test_vectors_full_op_reshaped, S_corr)

        # Compute spin correlation between first two particles (on vectors evolved using neural network)
        (correlations_nn_real, correlations_nn_imag) = misc.correlation_in_reshaped_set(nn_predicted_vectors, S_corr)

        # Save correlations in the evolved eigenvector of H_quenched
        if (N0,N1) not in y_correlations_eigvec_H:
            y_correlations_eigvec_H[N0, N1] = {"real":[], "imag":[]}
        y_correlations_eigvec_H[N0, N1]["real"].append([correlations_full_op_real[0], correlations_nn_real[0]])
        y_correlations_eigvec_H[N0, N1]["imag"].append([correlations_full_op_imag[0], correlations_nn_imag[0]])

    # Increase the batch_iterator, so that appropriate slice is cut out from the
    # testing set in the next iteration of the loop
    batch_iterator += 1
print("\nFinished!")

all_exact_energies = np.stack(all_exact_energies)
all_exact_energies = all_exact_energies.T

all_predicted_energies = np.stack(all_predicted_energies)
all_predicted_energies = all_predicted_energies.T

# ******************************************************************************
all_exact_energies_BQ = np.stack(all_exact_energies_BQ)
all_exact_energies_BQ = all_exact_energies_BQ.T

all_predicted_energies_BQ = np.stack(all_predicted_energies_BQ)
all_predicted_energies_BQ = all_predicted_energies_BQ.T
# ******************************************************************************

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
# if not os.path.exists(dir_name+"/Correlations/Eigenvector_HQ"):
#     os.mkdir(dir_name+"/Correlations/Eigenvector_HQ")
#     print("Directory " , dir_name+"/Correlations/Eigenvector_HQ" ,  " created")

# Create appropriate folders to store coefficients
if not os.path.exists(dir_name+"/Coefficients"):
    os.mkdir(dir_name+"/Coefficients")
    print("Directory " , dir_name+"/Coefficients" ,  " created")
# if not os.path.exists(dir_name+"/Coefficients/Eigenvector_HQ"):
#     os.mkdir(dir_name+"/Coefficients/Eigenvector_HQ")
#     print("Directory " , dir_name+"/Coefficients/Eigenvector_HQ" ,  " created")

################################################################################
# Save precise values of overlaps
################################################################################

print("Saving precise values of overlaps...", end="")
np.savetxt(dir_name + "/Precise_overlaps.csv", all_overlaps, delimiter=",")
print(" done!")

################################################################################
# Save precise values of energies
################################################################################

print("Saving precise values of energies...", end="")
np.savetxt(dir_name + "/Exact_energies.csv", all_exact_energies, delimiter=",")
np.savetxt(dir_name + "/Predicted_energies.csv", all_predicted_energies, delimiter=",")
print(" done!")

# ******************************************************************************
# Save precise values of energies BQ
# ******************************************************************************

print("Saving precise values of energies BQ...", end="")
np.savetxt(dir_name + "/Exact_energies_BQ.csv", all_exact_energies_BQ, delimiter=",")
np.savetxt(dir_name + "/Predicted_energies_BQ.csv", all_predicted_energies_BQ, delimiter=",")
print(" done!")

################################################################################
# Save plots of evolved correlations between particles in the first eigenvector of H_quenched
################################################################################


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
plt.savefig(dir_name + "/Mean_overlap.png")
plt.close()
print(" done!")
