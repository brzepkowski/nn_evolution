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

with open('training_input.npy', 'rb') as f:
    reshaped_training_input_vectors = np.load(f)
with open('training_output.npy', 'rb') as f:
    reshaped_training_output_vectors = np.load(f)
with open('validation_input.npy', 'rb') as f:
    reshaped_validation_input_vectors = np.load(f)
with open('validation_output.npy', 'rb') as f:
    reshaped_validation_output_vectors = np.load(f)

################################################################################
# Load/create the model
################################################################################

# Change the directory to the folder, in which we will be storing all results from
# this network's structure (model itself and all plots)
dir_name = "bottleneck_activation=" + activation + "_N=" + str(N) + "_t=" + str(t) + "_t_total=" + str(t_total) + "_subtr_neurons=" + str(subtracted_neurons)

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
    ############################################################################
    # Generate structure of the network
    ############################################################################

    # Normalization only after last layer
    input = Input(shape=(2*(2**N),))
    hidden_layer_0 = Dense(2*(2**N) - subtracted_neurons, activation=activation)(input)
    hidden_layer_1 = Dense(2*(2**N), activation=activation)(hidden_layer_0)
    # hidden_layer_2 = Dense(2*(2**N), activation=activation)(hidden_layer_1)
    # hidden_layer_3 = Dense(2*(2**N), activation=activation)(hidden_layer_2)
    output = Lambda(lambda t: l2_normalize(100000*t, axis=1))(hidden_layer_1)

    ############################################################################
    # Train the model
    ############################################################################

    model = Model(input, output)
    model.summary()

    BATCH_SIZE = 10000
    EPOCHS = 2000
    PATIENCE = 250

    early_stopping= EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')
    model_checkpoint = ModelCheckpoint(best_model_filename, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    model.compile(optimizer='adam', loss=my_cost, metrics=['accuracy']) # 'accuracy' doesn't change anything during training. It just makes val_my_cost available to the callbacks
    history_callback = model.fit(reshaped_training_input_vectors, reshaped_training_output_vectors,
                                        batch_size=BATCH_SIZE,
                                        epochs=EPOCHS,
                                        shuffle=True,
                                        callbacks=[early_stopping, model_checkpoint],
                                        # callbacks=[model_checkpoint],
                                        # shuffle=False) # Changed to check, if network will learn better on more consistent "packages"
                                        validation_data=(reshaped_validation_input_vectors, reshaped_validation_output_vectors))

    # Load the best model
    # Note: The best model was already saved in the 'best_model.h5' file!
    model = load_model(best_model_filename, custom_objects={"l2_normalize": l2_normalize, "my_cost": my_cost})

    loss_history = np.array(history_callback.history["loss"])
    val_loss_history = np.array(history_callback.history["val_loss"])

    ############################################################################
    # Save history of learning
    ############################################################################

    print("Saving history of learning...", end="")
    np.savetxt(dir_name + "/loss_history.txt", loss_history, delimiter=",")
    np.savetxt(dir_name + "/val_loss_history.txt", val_loss_history, delimiter=",")
    # np.savetxt(dir_name + "/accuracy_history.txt", accuracy_history, delimiter=",")
    # np.savetxt(dir_name + "/val_accuracy_history.txt", val_accuracy_history, delimiter=",")
    print(" done!")

    ################################################################################
    # Test the accuracy of the model on the validation set
    ################################################################################

    score, acc = model.evaluate(reshaped_validation_input_vectors, reshaped_validation_output_vectors, batch_size=256)

    print('Validation score:', score)
    print('Validation accuracy:', acc)

    ############################################################################
    # Save information about metaparameters used during training
    ############################################################################

    print("Saving meta information...", end="")
    file = open(dir_name + "/Meta_information.txt","w+")
    file.write("===== TIME EVOLUTION =====\n")
    file.write("Type: 2 ENCODER-DECODER layers; normalization ONLY after last layer\n")
    file.write("Neurons subtracted in bottleneck: " + str(subtracted_neurons) + "\n")
    file.write("Activation: " + activation + "\n")
    file.write("Batch size: " + str(BATCH_SIZE) + "\n")
    file.write("Epochs (without EarlyStopping): " + str(EPOCHS) + "\n")
    file.write("Patience: " + str(PATIENCE) + "\n")
    file.write("N: " + str(N) + "\n")
    file.write("Timestep: " + str(t) + "\n")
    file.write("Total evolution time: " + str(t_total) + "\n")
    file.write("Training set size: " + str(np.shape(reshaped_training_input_vectors)) + "\n")
    file.write("Validation set size: " + str(np.shape(reshaped_validation_input_vectors)) + "\n")
    file.write("Validation score: " + str(score))
    file.write("Validation accuracy: " + str(acc))
    file.close()
    print(" done!")
