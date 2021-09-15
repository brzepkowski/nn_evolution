################################################################################
# In this version of the network's structure normalization is added only at
# the end of the network (not inbetween hidden layers).
################################################################################
from keras.layers import Input, Dense, Dropout, Lambda, concatenate
from keras.models import Model, load_model
from keras.backend import l2_normalize
from keras.callbacks import EarlyStopping
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


if len(sys.argv) != 7:
    print("Wrong number of arguments provided. Please give them in the following fashion:")
    print("- optimizer,")
    print("- loss function,")
    print("- activaton function,")
    print("- N,")
    print("- timestep,")
    print("- total_evolution_time.")
    sys.exit()

optimizer = sys.argv[1]
loss = sys.argv[2]
activation = sys.argv[3]
N = int(sys.argv[4])
t = float(sys.argv[5])
t_total = float(sys.argv[6])

# Below variables are hardcoded, in order not to make some silly mistake
# J = 1.0
# h1 = 1.0
# h2 = 0.0

################################################################################
# Check if proper sets and operators were already generated
################################################################################

# data_dir_name = 'data_N=' + str(N) + '_t=' + str(t) + '_t_total=' + str(t_total)
# print("data_dir_name: ", data_dir_name)
#
# if not os.path.exists(data_dir_name):
#     print("Training, validation and test sets as well as operators were not generated yet!")
#     sys.exit()

################################################################################
# Load the training, validation and test sets. Also load the eigenvectors of H & HQ,
# and input to the neural network, on the basis of which it will make predictions
################################################################################

with open('training_input.npy', 'rb') as f:
    reshaped_training_input_vectors = np.load(f)
with open('training_output.npy', 'rb') as f:
    reshaped_training_output_vectors = np.load(f)
with open('validation_input.npy', 'rb') as f:
    reshaped_validation_input_vectors = np.load(f)
with open('validation_output.npy', 'rb') as f:
    reshaped_validation_output_vectors = np.load(f)
with open('test_input.npy', 'rb') as f:
    reshaped_test_input_vectors = np.load(f)
with open('test_output.npy', 'rb') as f:
    reshaped_test_output_vectors = np.load(f)
with open('test_input_simple_op.npy', 'rb') as f:
    reshaped_test_input_vectors_simple_op = np.load(f)
with open('test_output_simple_op.npy', 'rb') as f:
    reshaped_test_output_vectors_simple_op = np.load(f)
with open('test_eigenvalues.npy', 'rb') as f:
    test_eigenvalues = np.load(f)
# Load data for making predictions
with open('nn_input_for_predictions.npy', 'rb') as f:
    nn_predicted_vectors = np.load(f)
# Load eigenvalues of H
eigenvalues_H = load_vector_from_file("eigenvalues_H.dat", 2**N).astype(np.float64)

################################################################################
# Load/create the model
################################################################################

# Change the directory to the folder, in which we will be storing all results from
# this network's structure (model itself and all plots)
dir_name = "optimizer=" + optimizer + "_loss=" + loss + "_activation=" + activation

model_filename = dir_name + '/model_' + str(optimizer) + '_' + str(loss) + '_' + str(activation)

# Check if the model already exists. If not, create appropriate structure
if path.exists(model_filename):
    model = load_model(model_filename, custom_objects={"l2_normalize": l2_normalize})
    print("\n###############################")
    print("### Model loaded from file! ###")
    print("###############################\n")
else:
    ############################################################################
    # Generate structure of the network
    ############################################################################

    input = Input(shape=(2*(2**N),))
    hidden_layer_0 = Dense(2*(2**N), activation=activation)(input)
    dropout_layer_0 = Dropout(.2, input_shape=(2,))(hidden_layer_0)
    hidden_layer_1 = Dense(2*(2**N), activation=activation)(hidden_layer_0)
    dropout_layer_1 = Dropout(.2, input_shape=(2,))(hidden_layer_1)
    hidden_layer_2 = Dense(2*(2**N), activation=activation)(hidden_layer_1)
    dropout_layer_2 = Dropout(.2, input_shape=(2,))(hidden_layer_2)
    hidden_layer_3 = Dense(2*(2**N), activation=activation)(hidden_layer_2)
    dropout_layer_3 = Dropout(.2, input_shape=(2,))(hidden_layer_3)
    output = Lambda(lambda t: l2_normalize(1000*t, axis=1))(hidden_layer_3)

    ############################################################################
    # Train the model
    ############################################################################

    model = Model(input, output)
    model.summary()

    callback = EarlyStopping(monitor='loss', patience=3)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    history = model.fit(reshaped_training_input_vectors, reshaped_training_output_vectors,
                                        batch_size=256,
                                        epochs=2000,
                                        shuffle=True,
                                        callbacks=[callback])
                                        # shuffle=False) # Changed to check, if network will learn better on more consistent "packages"
                                        # validation_data=(reshaped_test_input_vectors, reshaped_test_output_vectors))

    ############################################################################
    # Save the model
    ############################################################################

    # Create target directory if doesn't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name, " created!")

    # 'model_filename' was created above
    model.save(model_filename)

################################################################################
# Test the accuracy of the model on the validation set
################################################################################

score, acc = model.evaluate(reshaped_validation_input_vectors, reshaped_validation_output_vectors,
                                            batch_size=256)

print('Validation score:', score)
print('Validation accuracy:', acc)

################################################################################
# Compare predictions given by model with the exact evolution
################################################################################
x = []
for i in range(int(t_total/t)):
    x.append((i+1)*t)
x_max = x[-1]
mean_overlap_per_timestep_eigvecs_H = []

# Below variables correspond to all coefficients of ONLY first vector in the test set
y_coefficients_eigvec_HQ_real = []
y_coefficients_eigvec_HQ_imag = []

reshaped_vector_length = np.shape(reshaped_test_input_vectors[0])[0]
for i in range(int(reshaped_vector_length/2)):
    y_coefficients_eigvec_HQ_real.append([])
    y_coefficients_eigvec_HQ_imag.append([])

y_correlations_eigvec_HQ = {}

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

for time_step in x:
    print("Current timestep: ", round(time_step, 5), " / ", round(x_max, 2), end="\r")

    # Evolution using full operator
    batch_beginning = batch_iterator*batch_size
    batch_ending = (batch_iterator+1)*batch_size

    evolved_test_vectors_full_op_reshaped = reshaped_test_output_vectors[batch_beginning:batch_ending]

    # Evolution using simple operator
    evolved_test_vectors_simple_op_reshaped = reshaped_test_output_vectors_simple_op[batch_beginning:batch_ending]

    # Evolution using neural network
    nn_predicted_vectors = model.predict(nn_predicted_vectors)

    # Compute overlap between precisely evolved vector and one evolved using U_t_simple
    overlaps_simple_op = misc.overlap_of_reshaped_sets(evolved_test_vectors_full_op_reshaped, evolved_test_vectors_simple_op_reshaped)

    # Compute overlap between precisely evolved vector and the one returned by neural network
    overlaps = misc.overlap_of_reshaped_sets(evolved_test_vectors_full_op_reshaped, nn_predicted_vectors)
    all_overlaps.append(overlaps)

    # Save mean overlap with all eigenvectors of H_quenched in testing set. Note,
    # that in this version of the code our testing set consists only of eigenvectors of HQ!
    mean_overlap_per_timestep_eigvecs_H.append([np.mean(overlaps), np.mean(overlaps_simple_op)])

    # Save real parts of the first evolved vector
    for i in range(0, reshaped_vector_length, 2):
        y_coefficients_eigvec_HQ_real[floor(i/2)].append([evolved_test_vectors_full_op_reshaped[0][i], nn_predicted_vectors[0][i]])

    # Save imaginary parts of the first evolved vector
    for i in range(1, reshaped_vector_length, 2):
        y_coefficients_eigvec_HQ_imag[floor(i/2)].append([evolved_test_vectors_full_op_reshaped[0][i], nn_predicted_vectors[0][i]])

    ############################################################################
    # Save values of correlation between all pairs of particles
    ############################################################################

    for N0 in range(N-1):
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
            if (N0,N1) not in y_correlations_eigvec_HQ:
                y_correlations_eigvec_HQ[N0, N1] = {"real":[], "imag":[]}
            y_correlations_eigvec_HQ[N0, N1]["real"].append([correlations_full_op_real[0], correlations_nn_real[0]])
            y_correlations_eigvec_HQ[N0, N1]["imag"].append([correlations_full_op_imag[0], correlations_nn_imag[0]])

    # Increase the batch_iterator, so that appropriate slice is cut out from the
    # testing set in the next iteration of the loop
    batch_iterator += 1
print("\nFinished!")

all_overlaps = np.stack(all_overlaps)
all_overlaps = all_overlaps.T

y_coefficients_eigvec_HQ_real = np.array(y_coefficients_eigvec_HQ_real)
y_coefficients_eigvec_HQ_imag = np.array(y_coefficients_eigvec_HQ_imag)

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
if not os.path.exists(dir_name+"/Correlations/Eigenvector_HQ"):
    os.mkdir(dir_name+"/Correlations/Eigenvector_HQ")
    print("Directory " , dir_name+"/Correlations/Eigenvector_HQ" ,  " created")

# Create appropriate folders to store coefficients
if not os.path.exists(dir_name+"/Coefficients"):
    os.mkdir(dir_name+"/Coefficients")
    print("Directory " , dir_name+"/Coefficients" ,  " created")
if not os.path.exists(dir_name+"/Coefficients/Eigenvector_HQ"):
    os.mkdir(dir_name+"/Coefficients/Eigenvector_HQ")
    print("Directory " , dir_name+"/Coefficients/Eigenvector_HQ" ,  " created")

################################################################################
# Save meta information about the accuracy of learning of current network
################################################################################

print("Saving meta information...", end="")
file = open(dir_name + "/Meta_information_N=" + str(N) + ".txt","w+")
file.write("Type: 4 simple layers (NOT ENCODER-DECODER); normalization ONLY after last layer\n")
file.write("Optimizer: " + optimizer + "\n")
file.write("Loss function: " + loss + "\n")
file.write("Activation: " + activation + "\n")
file.write("N: " + str(N) + "\n")
file.write("Timestep: " + str(t) + "\n")
file.write("Total evolution time: " + str(t_total) + "\n")
file.write("Score (in validation set): " + str(score) + "\n")
file.write("Accuracy (in validation set): " + str(acc) + "\n")
file.close()
print(" done!")

################################################################################
# Save precise values of overlaps
################################################################################

print("Saving precise values of overlaps...", end="")
np.savetxt(dir_name + "/Precise_overlaps_N=" + str(N) + ".csv", all_overlaps, delimiter=",")
print(" done!")

################################################################################
# Save plots of evolved correlations between particles in the first eigenvector of H_quenched
################################################################################

print("Saving plots of correlations...", end="")
for N0 in range(N-1):
    for N1 in range(N0+1, N):
        title = "EIGENVECTOR H_QUENCHED\nCORRELATION (REAL), "
        title += "N0 = " + str(N0) + ", N1 = " + str(N1)
        plt.title(title)
        plt.plot(x, y_correlations_eigvec_HQ[N0,N1]["real"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dir_name + "/Correlations/Eigenvector_HQ/Corr_n0=" + str(N0) + "_n1=" + str(N1) + "_(real).png")
        plt.clf()

        # WARNING: Below plot might be discarded, because it should give values close to 0. It might be left just to check,
        # if correlations are truly real numbers
        title = "EIGENVECTOR H_QUENCHED\nCORRELATION (IMAG), "
        title += "N0 = " + str(N0) + ", N1 = " + str(N1)
        plt.title(title)
        plt.plot(x, y_correlations_eigvec_HQ[N0,N1]["imag"], '.-')
        plt.xlabel('Time')
        plt.ylabel('Mean correlation')
        plt.gca().legend(["exp(-iHt)", "NN"])
        plt.xlim(xmin=0, xmax=x_max)
        plt.grid()
        plt.savefig(dir_name + "/Correlations/Eigenvector_HQ/Corr_n0=" + str(N0) + "_n1=" + str(N1) + "_(imag).png")
        plt.clf()
print(" done!")

################################################################################
# Save plots of evolved coefficients in the first eigenvector of H_quenched
################################################################################
"""
print("Saving plots of coefficients...", end="")
for i in range(int(reshaped_vector_length/2)):
    # Plot real part of the coefficients in the eigenvector of H_quenched
    plt.title("EIGENVECTOR H_QUENCHED\nCOEFFICIENT " + str(i) + " (REAL)")
    plt.plot(x, y_coefficients_eigvec_HQ_real[i], '.-')
    plt.xlabel('Time')
    plt.gca().legend(["exp(-iHt)", "NN"])
    plt.grid()
    plt.savefig(dir_name + "/Coefficients/Eigenvector_HQ/Coeff_n=" + str(i) + "_(real).png")
    plt.clf()

    # Plot imaginary part of the coefficients in the eigenvector of H_quenched
    plt.title("EIGENVECTOR H_QUENCHED\nCOEFFICIENT " + str(i) + " (IMAG)")
    plt.plot(x, y_coefficients_eigvec_HQ_imag[i], '.-')
    plt.xlabel('Time')
    plt.gca().legend(["exp(-iHt)", "NN"])
    plt.grid()
    plt.savefig(dir_name + "/Coefficients/Eigenvector_HQ/Coeff_n=" + str(i) + "_(imag).png")
    plt.clf()
print(" done!")
"""
################################################################################
# Save plots of correlations between particles in the first eigenvector of H_quenched
################################################################################

print("Saving plot of overlaps...", end="")
plt.title("OVERLAP \nEIGENVECTORS OF H_QUENCHED")
plt.plot(x, mean_overlap_per_timestep_eigvecs_H, '.-')
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.xlim(xmin=0, xmax=x_max)
plt.grid()
plt.savefig(dir_name + "/Overlap_HQ.png")
plt.clf()
print(" done!")
