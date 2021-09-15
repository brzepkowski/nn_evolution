################################################################################
# In this version of the network's structure normalization is added only at
# the end of the network (not inbetween hidden layers).
################################################################################
from keras.layers import Input, Dense, Dropout, Lambda, concatenate, LSTM, Embedding
from keras.models import Model, load_model
from keras.backend import l2_normalize
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import misc
from misc import load_array_from_file, load_vector_from_file, overlap_of_reshaped
from numpy.linalg import norm
from itertools import combinations
from math import floor
import sys
import os.path
from os import path
# For testing
from scipy.linalg import expm
from numpy.linalg import eig
import time


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
window_length = 5

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
    nn_input_for_predictions = np.load(f)
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
    # Generate partial testing set
    ############################################################################
    model = None
    for i in range(20):
        # print("i: ", i)
        # time.sleep(1)
        time_series_number = 0
        time_series_length = int(t_total/t)

        while (time_series_number+1)*time_series_length < len(reshaped_training_input_vectors):
            set_beginning = time_series_number*time_series_length
            set_ending = set_beginning+time_series_length
            temp_input_vectors = reshaped_training_input_vectors[set_beginning:set_ending]
            temp_output_vectors = reshaped_training_output_vectors[set_beginning:set_ending]

            partial_training_input_vectors = []
            partial_training_output_vectors = []
            batch_number = 0
            while batch_number < time_series_length - window_length:
                partial_training_input_vectors.append(temp_input_vectors[batch_number:batch_number+window_length])
                partial_training_output_vectors.append(temp_output_vectors[batch_number:batch_number+window_length])
                batch_number += 1
            partial_training_input_vectors = np.asarray(partial_training_input_vectors)
            partial_training_output_vectors = np.asarray(partial_training_output_vectors)

            ############################################################################
            # Generate structure of the network
            ############################################################################

            # max_features = 20000  # Only consider the top 20k words

            # input = Input(shape=(None,))
            #
            # embedding = Embedding(max_features, 2*(2**N))(input)
            if model == None:
                features = 2*(2**N)
                time_window = window_length
                batch_size = batch_number
                print("batch_size: ", batch_size)
                # time.sleep(4)
                input = Input(shape=(time_window, features))
                lstm = LSTM(features, return_sequences=True)(input)
                # pre_output = Dense(2*(2**N), activation=activation)(input)
                # output = Lambda(lambda t: l2_normalize(1000*t, axis=1))(pre_output)
                # output = Dense(1, activation="sigmoid")(input)
                model = Model(inputs=input, outputs=lstm)

                ############################################################################
                # Train the model
                ############################################################################

                # model = Model(input, output)
                model.summary()
                # sys.exit()

                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


            # callback = EarlyStopping(monitor='accuracy', patience=10)
            history = model.fit(partial_training_input_vectors, partial_training_output_vectors,
                                                # batch_size=256,
                                                verbose=2,
                                                epochs=10,
                                                shuffle=False)
                                                # callbacks=[callback])
                                                # shuffle=False) # Changed to check, if network will learn better on more consistent "packages"
                                                # validation_data=(reshaped_test_input_vectors, reshaped_test_output_vectors))

            # print("PRZESZŁO!")
            # sys.exit()
            ############################################################################
            # Save the model
            ############################################################################
            # # Create target directory if doesn't exist
            # if not os.path.exists(dir_name):
            #     os.mkdir(dir_name)
            #     print("Directory ", dir_name, " created!")
            #
            # # 'model_filename' was created above
            # model.save(model_filename)
            model.reset_states()
            time_series_number += 1

############################################################################
# Generate partial validation set
############################################################################
time_series_number = 0

while (time_series_number+1)*time_series_length < len(reshaped_validation_input_vectors):
    set_beginning = time_series_number*time_series_length
    set_ending = set_beginning+time_series_length
    temp_input_vectors = reshaped_validation_input_vectors[set_beginning:set_ending]
    temp_output_vectors = reshaped_validation_output_vectors[set_beginning:set_ending]

    partial_validation_input_vectors = []
    partial_validation_output_vectors = []
    batch_number = 0
    while batch_number < time_series_length - window_length:
        partial_validation_input_vectors.append(temp_input_vectors[batch_number:batch_number+window_length])
        partial_validation_output_vectors.append(temp_output_vectors[batch_number:batch_number+window_length])
        batch_number += 1
    partial_validation_input_vectors = np.asarray(partial_validation_input_vectors)
    partial_validation_output_vectors = np.asarray(partial_validation_output_vectors)

    ################################################################################
    # Test the accuracy of the model on the validation set
    ################################################################################

    score, acc = model.evaluate(partial_validation_input_vectors, partial_validation_output_vectors)
                                            # batch_size=256)

    print('Validation score:', score)
    print('Validation accuracy:', acc)
    time_series_number += 1


################################################################################
# Compare predictions given by model with the exact evolution
################################################################################
time_series_length = int(t_total/t)
time_series_number = 0

all_overlaps = []

while (time_series_number+1)*time_series_length < len(reshaped_test_input_vectors):

    psi_0_rnn = []
    for i in range(window_length):
        psi_0_rnn.append(nn_input_for_predictions[time_series_number])
    # psi_0_rnn = np.array(psi_0_rnn)
    overlaps = []

    for _t in range(time_series_length):

        set_beginning = time_series_number*time_series_length
        # set_ending = set_beginning+time_series_length

        psi_0_exact = reshaped_test_input_vectors[set_beginning + _t]
        psi_t_exact = reshaped_test_output_vectors[set_beginning + _t]

        psi_0_simple = reshaped_test_input_vectors_simple_op[set_beginning + _t]
        psi_t_simple = reshaped_test_output_vectors_simple_op[set_beginning + _t]

        # print("_t: ", _t)

        # print("[PRZED] psi_0_rnn: ", psi_0_rnn)
        # print("psi_0_exact: ", psi_0_exact)
        # print("psi_0_simple: ", psi_0_simple)

        # make and show prediction
        psi_0_rnn_input = np.array(psi_0_rnn)
        psi_0_rnn_input = np.reshape(psi_0_rnn_input, (1,np.shape(psi_0_rnn_input)[0], np.shape(psi_0_rnn_input)[1]))
        # print("[PO] psi_0_rnn: ", psi_0_rnn)
        psi_t_rnn = model.predict(psi_0_rnn_input)
        # print("psi_t_rnn: ", psi_t_rnn)
        # print("psi_t_rnn[0]: ", psi_t_rnn[0][0])

        # print("(0) psi_0_rnn: ", psi_0_rnn)
        psi_0_rnn.pop(0)
        # print("(1) psi_0_rnn: ", psi_0_rnn)

        psi_t_rnn = psi_t_rnn[0][0] / norm(psi_t_rnn[0][0])
        psi_0_rnn.append(psi_t_rnn)
        # print("(2) psi_0_rnn: ", psi_0_rnn)

        # print("Overlap Exact|SIMPLE: ", overlap_of_reshaped(psi_t_exact, psi_t_simple))
        # print("Overlap Exact|RNN: ", overlap_of_reshaped(psi_t_exact, psi_t_rnn))
        overlaps.append(overlap_of_reshaped(psi_t_exact, psi_t_rnn))
        # sys.exit()

        # window_length = 3
        # partial_validation_input_vectors = []
        # partial_validation_output_vectors = []
        # batch_number = 0
        # while batch_number < time_series_length - window_length:
        #     partial_validation_input_vectors.append(temp_input_vectors[batch_number:batch_number+window_length])
        #     partial_validation_output_vectors.append(temp_output_vectors[batch_number:batch_number+window_length])
        #     batch_number += 1
        # partial_validation_input_vectors = np.asarray(partial_validation_input_vectors)
        # partial_validation_output_vectors = np.asarray(partial_validation_output_vectors)
    all_overlaps.append(overlaps)
    time_series_number += 1

################################################################################
# Save precise values of overlaps
################################################################################

print("Saving precise values of overlaps...", end="")
np.savetxt("Precise_overlaps_N=" + str(N) + ".csv", all_overlaps, delimiter=",")
print(" done!")
