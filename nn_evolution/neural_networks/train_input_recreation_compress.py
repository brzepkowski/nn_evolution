# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

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
from numpy.linalg import norm
from itertools import combinations
from math import floor
import sys
import os.path
from os import path
# For testing
from scipy.linalg import expm
from numpy.linalg import eig

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


def train_input_recreation_compress(activation, N, dt, t_max, subtracted_neurons):

    # Set linewidth for printing arrays to infinity
    np.set_printoptions(linewidth=np.inf)

    ################################################################################
    # Load the training, validation and test sets. Also load the eigenvectors of H & HQ,
    # and input to the neural network, on the basis of which it will make predictions
    ################################################################################

    dir_name = "data_N=" + str(N) + "_dt=" + str(dt) + "_t_max=" + str(t_max)

    with open(dir_name + '/training_input.npy', 'rb') as f:
        reshaped_training_input_vectors = np.load(f)
    # Note: This is the main change. We are putting the input as output to the neural network
    # with open(dir_name + '/training_output.npy', 'rb') as f:
    with open(dir_name + '/training_input.npy', 'rb') as f:
        reshaped_training_output_vectors = np.load(f)
    with open(dir_name + '/validation_input.npy', 'rb') as f:
        reshaped_validation_input_vectors = np.load(f)
    # with open(dir_name + '/validation_output.npy', 'rb') as f:
    with open(dir_name + '/validation_input.npy', 'rb') as f:
        reshaped_validation_output_vectors = np.load(f)

    ################################################################################
    # Load/create the model
    ################################################################################

    # Change the directory to the folder, in which we will be storing all results
    dir_name += "/nn_input_recreation_compress_activation=" + activation + "_subtr=" + str(subtracted_neurons)

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
        # hidden_layer_4 = Dense(2*(2**N), activation=activation)(hidden_layer_3)
        output = Lambda(lambda t: l2_normalize(100000*t, axis=1))(hidden_layer_1)

        ############################################################################
        # Instantiate model
        ############################################################################

        model = Model(input, output)
        model.summary()

        BATCH_SIZE = 10000
        EPOCHS = 2000
        PATIENCE = 250

        ############################################################################
        # Save information about final values of weights
        ############################################################################

        initial_weights_message = ""
        for layer in model.layers:
            initial_weights_message += "===== LAYER: " + layer.name + " =====\n"
            if layer.get_weights() != []:
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                initial_weights_message += "weights: " + str(np.average(np.absolute(weights))) + "\n"
                initial_weights_message += "biases: " + str(np.average(np.absolute(biases))) + "\n"
            else:
                initial_weights_message += "weights: [] \n"

        print("Saving initial weights' information...", end="", flush=True)
        file = open(dir_name + "/init_weights.txt","w+")
        file.write(initial_weights_message)
        file.close()
        print(" done!")

        ############################################################################
        # TRAIN MODEL
        ############################################################################

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

        # Load the best model, which is saved in the 'best_model.h5' file
        model = load_model(best_model_filename, custom_objects={"l2_normalize": l2_normalize, "my_cost": my_cost})

        loss_history = np.array(history_callback.history["loss"])
        val_loss_history = np.array(history_callback.history["val_loss"])

        ############################################################################
        # Save history of learning
        ############################################################################

        print("Saving history of learning...", end="")
        np.savetxt(dir_name + "/loss_history.txt", loss_history, delimiter=",", fmt='%f')
        np.savetxt(dir_name + "/val_loss_history.txt", val_loss_history, delimiter=",", fmt='%f')
        print(" done!")

        ################################################################################
        # Test the accuracy of the model on the validation set
        ################################################################################

        score, acc = model.evaluate(reshaped_validation_input_vectors, reshaped_validation_output_vectors, batch_size=256)

        print('Validation score:', score)
        print('Validation accuracy:', acc)

        ############################################################################
        # Save information about final values of weights
        ############################################################################

        final_weights_message = ""
        for layer in model.layers:
            final_weights_message += "===== LAYER: " + layer.name + " =====\n"
            if layer.get_weights() != []:
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                final_weights_message += "weights: " + str(np.average(np.absolute(weights))) + "\n"
                final_weights_message += "biases: " + str(np.average(np.absolute(biases))) + "\n"
            else:
                final_weights_message += "weights: [] \n"

        print("Saving final weights' information...", end="", flush=True)
        file = open(dir_name + "/final_weights.txt","w+")
        file.write(final_weights_message)
        file.close()
        print(" done!")

        ############################################################################
        # Save information about metaparameters used during training
        ############################################################################

        print("Saving meta information...", end="")
        file = open(dir_name + "/meta_information.txt","w+")
        file.write("===== INPUT RECREATION =====\n")
        file.write("Type: 2 ENCODER-DECODER layers; normalization ONLY after last layer\n")
        file.write("Activation: " + activation + "\n")
        file.write("Subtracted neurons: " + str(subtracted_neurons) + "\n")
        file.write("Batch size: " + str(BATCH_SIZE) + "\n")
        file.write("Epochs (without EarlyStopping): " + str(EPOCHS) + "\n")
        file.write("Patience: " + str(PATIENCE) + "\n")
        file.write("N: " + str(N) + "\n")
        file.write("Timestep: " + str(dt) + "\n")
        file.write("Total evolution time: " + str(t_max) + "\n")
        file.write("Training set size: " + str(np.shape(reshaped_training_input_vectors)) + "\n")
        file.write("Validation set size: " + str(np.shape(reshaped_validation_input_vectors)) + "\n")
        file.write("Validation score: " + str(score))
        file.write("Validation accuracy: " + str(acc))
        file.close()
        print(" done!")
