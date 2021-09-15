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


if len(sys.argv) != 6:
    print("Wrong number of arguments provided. Please give them in the following fashion:")
    print("- activaton function,")
    print("- N,")
    print("- timestep,")
    print("- total_evolution_time,")
    print("- # of neurons subtracted from the bottleneck layer.")
    sys.exit()

activation = sys.argv[1]
N = int(sys.argv[2])
t = float(sys.argv[3])
t_total = float(sys.argv[4])
subtracted_neurons = int(sys.argv[5])

total_timesteps = int(t_total / t)
print("total_timesteps: ", total_timesteps)

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
# with open('test_output.npy', 'rb') as f:
with open('test_input.npy', 'rb') as f:
    reshaped_test_output_vectors = np.load(f)
with open('test_eigenvalues.npy', 'rb') as f:
    test_eigenvalues = np.load(f)
# Load data for making predictions
# with open('nn_input_for_predictions.npy', 'rb') as f:
with open('test_input.npy', 'rb') as f:
    nn_predicted_vectors = np.load(f)
# Load eigenvalues of H
eigenvalues_H = load_vector_from_file("eigenvalues_H.dat", 2**N).astype(np.float64)

# Take only first values from each set, because we are not conducting time evolution in this case
test_size = int(np.shape(reshaped_test_input_vectors)[0] / total_timesteps)
print("=================> test_size: ", test_size)
reshaped_test_input_vectors = reshaped_test_input_vectors[:test_size]
reshaped_test_output_vectors = reshaped_test_output_vectors[:test_size]
nn_predicted_vectors = nn_predicted_vectors[:test_size]

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

################################################################################
# Load/create the model
################################################################################

# Change the directory to the folder, in which we will be storing all results from
# this network's structure (model itself and all plots)
dir_name = "recreate_input_bottleneck_activation=" + activation + "_N=" + str(N) + "_subtr_neurons=" + str(subtracted_neurons)

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
# Compare predictions given by model with the exact outputs
# (being the same as inputs)
################################################################################
# x = []
# for i in range(int(t_total/t)):
#     x.append((i+1)*t)
# x_max = x[-1]
# mean_overlap_per_timestep_eigvecs_H = []

# Below variables correspond to all coefficients of ONLY first vector in the test set
y_coefficients_eigvec_H_real = []
y_coefficients_eigvec_H_imag = []

reshaped_vector_length = np.shape(reshaped_test_input_vectors[0])[0]
for i in range(int(reshaped_vector_length/2)):
    y_coefficients_eigvec_H_real.append([])
    y_coefficients_eigvec_H_imag.append([])

y_correlations_eigvec_H = {}

# Under this variable we will be storing all overlaps. Although, the first entry consists of
# ENERGY VALUES (eigenvalues), thanks to which it will be possible to associate further values of overlap
# with specific vectors from the spectrum.
all_overlaps = [test_eigenvalues]
all_exact_energies = []
all_predicted_energies = []

exact_test_vectors_full_op_reshaped = reshaped_test_output_vectors

# Evolution using neural network
nn_predicted_vectors = model.predict(nn_predicted_vectors)

# Compute overlap between precisely evolved vector and the one returned by neural network
overlaps = misc.overlap_of_reshaped_sets(exact_test_vectors_full_op_reshaped, nn_predicted_vectors)
all_overlaps.append(overlaps)


exact_energies = misc.energies_of_reshaped_set(exact_test_vectors_full_op_reshaped, HQ)
predicted_energies = misc.energies_of_reshaped_set(nn_predicted_vectors, HQ)

all_exact_energies.append(exact_energies)
all_predicted_energies.append(predicted_energies)

print("===== ENTANGLEMENT =====")
if not os.path.exists(dir_name + "/Exact_entanglements.csv"):
    print("Generating from scratch")
    exact_entanglements = misc.entanglements_of_reshaped_set(exact_test_vectors_full_op_reshaped, N)
else:
    print("Loading from file")
    with open(dir_name + "/Exact_entanglements.npy", 'rb') as f:
        exact_entanglements = np.load(f)

# print("exact_entanglements: ", exact_entanglements)
differences_in_energy = np.abs(np.array(exact_energies) - np.array(predicted_energies))
absolute_energy_error = np.abs(np.array(exact_energies) - np.array(predicted_energies))
relative_energy_error = np.abs(np.array(exact_energies) - np.array(predicted_energies)) / np.array(exact_energies)

absolute_energy_error = absolute_energy_error.mean()
relative_energy_error = relative_energy_error.mean()

print("ŚREDNI BŁĄD BEZWZGLĘDNY: ", absolute_energy_error)
print("ŚREDNI BŁĄD WZGLĘDNY: ", relative_energy_error)

plt.title("Energy difference vs. entanglement")
plt.plot(exact_entanglements, differences_in_energy, '.')
plt.xlabel('Entanglement')
plt.xlabel('Energy difference')
plt.grid()
plt.savefig(dir_name + "/Energy_diff_vs_entanglement.png")
plt.close()


"""
# Save real parts of the first predicted vector
for i in range(0, reshaped_vector_length, 2):
    y_coefficients_eigvec_H_real[floor(i/2)].append([evolved_test_vectors_full_op_reshaped[0][i], nn_predicted_vectors[0][i]])

# Save imaginary parts of the first predicted vector
for i in range(1, reshaped_vector_length, 2):
    y_coefficients_eigvec_H_imag[floor(i/2)].append([evolved_test_vectors_full_op_reshaped[0][i], nn_predicted_vectors[0][i]])
"""
############################################################################
# Save values of correlation between all pairs of particles
############################################################################

N0 = 0
for N1 in range(N0+1, N):

    # Load spin correlation operator from file
    S_corr_filename = 'S_corr_' + str(N0) + '_' + str(N1) + '.npy'
    with open(S_corr_filename, 'rb') as f:
        S_corr = np.load(f)

    # Compute spin correlation between the two particles (on vectors evolved using full operator)
    (correlations_full_op_real, correlations_full_op_imag) = misc.correlation_in_reshaped_set(exact_test_vectors_full_op_reshaped, S_corr)

    # Compute spin correlation between first two particles (on vectors evolved using neural network)
    (correlations_nn_real, correlations_nn_imag) = misc.correlation_in_reshaped_set(nn_predicted_vectors, S_corr)

    # Save correlations in the evolved eigenvector of H_quenched
    if (N0,N1) not in y_correlations_eigvec_H:
        y_correlations_eigvec_H[N0, N1] = {"real":[], "imag":[]}
    y_correlations_eigvec_H[N0, N1]["real"].append([correlations_full_op_real[0], correlations_nn_real[0]])
    y_correlations_eigvec_H[N0, N1]["imag"].append([correlations_full_op_imag[0], correlations_nn_imag[0]])

print("\nFinished!")

all_exact_energies = np.stack(all_exact_energies)
all_exact_energies = all_exact_energies.T

all_predicted_energies = np.stack(all_predicted_energies)
all_predicted_energies = all_predicted_energies.T

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

################################################################################
# Save precise values of entanglement
################################################################################

if not os.path.exists(dir_name + "/Exact_entanglements.csv"):
    print("Saving precise values of entanglement...", end="")
    np.savetxt(dir_name + "/Exact_entanglements.csv", exact_entanglements, delimiter=",")
    print(" done!")
    with open(dir_name + "/Exact_entanglements.npy", 'wb') as f:
        np.save(f, exact_entanglements)

################################################################################
# Save plots of evolved correlations between particles in the first eigenvector of H_quenched
################################################################################


print("Saving precise values of correlations...", end="")
N0 = 0
for N1 in range(N0+1, N):
    np.savetxt(dir_name + "/Correlations/Corr_n0=" + str(N0) + "_n1=" + str(N1) + "_(real).txt", y_correlations_eigvec_H[N0,N1]["real"], delimiter=",")
    np.savetxt(dir_name + "/Correlations/Corr_n0=" + str(N0) + "_n1=" + str(N1) + "_(imag).txt", y_correlations_eigvec_H[N0,N1]["imag"], delimiter=",")
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
