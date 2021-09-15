################################################################################
# In this version of the network's structure normalization is added only at
# the end of the network (not inbetween hidden layers).
################################################################################
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.backend import eval, l2_normalize, cast, dtype, floatx, sqrt, pow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from misc import load_array_from_file, load_vector_from_file, load_array_from_file_2
from numpy.linalg import norm
from itertools import combinations
from math import floor
import sys
import os.path
from os import path
import pickle

if len(sys.argv) != 6:
    print("Wrong number of arguments provided. Please give them in the following fashion:")
    print("- activaton function,")
    print("- N,")
    print("- timestep,")
    print("- total_evolution_time,")
    print("- set_number.")
    sys.exit()

activation = sys.argv[1]
N = int(sys.argv[2])
t = float(sys.argv[3])
t_total = float(sys.argv[4])
set_number = int(sys.argv[5])

initial_learning_rate = 0.001
learning_rate = 0.001
total_iterations = 0
iterations_per_epoch = 0

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

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        global iterations_per_epoch
        if iterations_per_epoch == 0:
            iterations_per_epoch = eval(self.model.optimizer.iterations)

################################################################################
# Load the training, validation and test sets. Also load the eigenvectors of H & HQ,
# and input to the neural network, on the basis of which it will make predictions
################################################################################
training_input_filename = ("training_input_{set_number:d}.dat")
training_input_filename = training_input_filename.format(
    set_number=set_number,
)
training_output_filename = ("training_output_{set_number:d}.dat")
training_output_filename = training_output_filename.format(
    set_number=set_number,
)
validation_input_filename = ("validation_input_{set_number:d}.dat")
validation_input_filename = validation_input_filename.format(
    set_number=set_number,
)
validation_output_filename = ("validation_output_{set_number:d}.dat")
validation_output_filename = validation_output_filename.format(
    set_number=set_number,
)

reshaped_training_input_vectors = load_array_from_file_2(training_input_filename, 2**(N+1)).astype(np.float64)
reshaped_training_output_vectors = load_array_from_file_2(training_output_filename, 2**(N+1)).astype(np.float64)
reshaped_validation_input_vectors = load_array_from_file_2(validation_input_filename, 2**(N+1)).astype(np.float64)
reshaped_validation_output_vectors = load_array_from_file_2(validation_output_filename, 2**(N+1)).astype(np.float64)

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
    # Loading last learning rate from previous training session
    with open(dir_name + '/initial_learning_rate.pkl', 'rb') as f:
        initial_learning_rate = pickle.load(f)
    with open(dir_name + '/learning_rate.pkl', 'rb') as f:
        learning_rate = pickle.load(f)
    with open(dir_name + '/total_iterations.pkl', 'rb') as f:
        total_iterations = pickle.load(f)

    print("\n###############################")
    print("### Model loaded from file! ###")
    print("###############################\n")
    print("LOADED LEARNING RATE: ", learning_rate)
    print("LOADED INITIAL LEARNING RATE: ", initial_learning_rate)

else:
    ############################################################################
    # Generate structure of the network
    ############################################################################

    # Normalization only after last layer
    input = Input(shape=(2*(2**N),))
    hidden_layer_0 = Dense(2*(2**N), activation=activation)(input)
    hidden_layer_1 = Dense(2*(2**N), activation=activation)(hidden_layer_0)
    # hidden_layer_2 = Dense(2*(2**N), activation=activation)(hidden_layer_1)
    # hidden_layer_3 = Dense(2*(2**N), activation=activation)(hidden_layer_2)
    output = Lambda(lambda t: l2_normalize(100000*t, axis=1))(hidden_layer_1)


    model = Model(input, output)
    model.summary()

################################################################################
# Train the model
################################################################################

BATCH_SIZE = 10000
EPOCHS = 500
PATIENCE = 50
beta_1=0.9
beta_2=0.999

early_stopping= EarlyStopping(monitor='val_loss', patience=PATIENCE, mode='min')
model_checkpoint = ModelCheckpoint(best_model_filename, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
my_callback = MyCallback()
opt = Adam(learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2)
model.compile(optimizer=opt, loss=my_cost, metrics=['accuracy']) # 'accuracy' doesn't change anything during training. It just makes val_my_cost available to the callbacks
history_callback = model.fit(reshaped_training_input_vectors, reshaped_training_output_vectors,
                                    batch_size=BATCH_SIZE,
                                    epochs=EPOCHS,
                                    shuffle=True,
                                    callbacks=[early_stopping, model_checkpoint, my_callback],
                                    # callbacks=[model_checkpoint],
                                    # shuffle=False) # Changed to check, if network will learn better on more consistent "packages"
                                    validation_data=(reshaped_validation_input_vectors, reshaped_validation_output_vectors))

# Load the best model
# Note: The best model was already saved in the 'best_model.h5' file!
model = load_model(best_model_filename, custom_objects={"l2_normalize": l2_normalize, "my_cost": my_cost})

loss_history = np.array(history_callback.history["loss"])
val_loss_history = np.array(history_callback.history["val_loss"])

def calculate_final_learning_rate(initial_learning_rate, iterations, beta_1, beta_2):
    t = cast(iterations, floatx()) + 1
    learning_rate = initial_learning_rate * (sqrt(1. - pow(beta_2, t)) /
                                                        (1. - pow(beta_1, t)))
    return learning_rate

stopped_epoch = early_stopping.stopped_epoch
if stopped_epoch == 0: # This is the case, in which Early Stopping hasn't been applied
    stopped_epoch = EPOCHS
print("[BEFORE] total_iterations: ", total_iterations)
total_iterations += stopped_epoch*iterations_per_epoch
print("[AFTER] total_iterations: ", total_iterations)

print("stopped_epoch: ", stopped_epoch)
print("iterations_per_epoch: ", iterations_per_epoch)
print("total_iterations: ", stopped_epoch*iterations_per_epoch)

learning_rate = calculate_final_learning_rate(initial_learning_rate, total_iterations, beta_1, beta_2)

print("FINAL LEARNING RATE: ", learning_rate)

# Saving initial and final learning rates
with open(dir_name + '/initial_learning_rate.pkl', 'wb') as f:
    pickle.dump(eval(initial_learning_rate), f)
with open(dir_name + '/learning_rate.pkl', 'wb') as f:
    pickle.dump(eval(learning_rate), f)
with open(dir_name + '/total_iterations.pkl', 'wb') as f:
    pickle.dump(eval(total_iterations), f)

################################################################################
# Save history of learning
################################################################################

print("Saving history of learning...", end="")
np.savetxt(dir_name + "/loss_history.txt", loss_history, delimiter=",")
np.savetxt(dir_name + "/val_loss_history.txt", val_loss_history, delimiter=",")
# np.savetxt(dir_name + "/accuracy_history.txt", accuracy_history, delimiter=",")
# np.savetxt(dir_name + "/val_accuracy_history.txt", val_accuracy_history, delimiter=",")
print(" done!")

################################################################################
# Save information about metaparameters used during training
################################################################################

meta_information_filename = ("/Meta_information_{set_number:d}.txt")
meta_information_filename = meta_information_filename.format(
    set_number=set_number,
)

print("Saving meta information...", end="")
file = open(dir_name + meta_information_filename, "w+")
file.write("Type: 2 simple layers (NOT ENCODER-DECODER); normalization ONLY after last layer\n")
file.write("Initial learning rate: " + str(learning_rate) + "\n")
file.write("Activation: " + activation + "\n")
file.write("Batch size: " + str(BATCH_SIZE) + "\n")
file.write("Epochs (without EarlyStopping): " + str(EPOCHS) + "\n")
file.write("Patience: " + str(PATIENCE) + "\n")
file.write("N: " + str(N) + "\n")
file.write("Timestep: " + str(t) + "\n")
file.write("Total evolution time: " + str(t_total) + "\n")
file.write("Training set size: " + str(np.shape(reshaped_training_input_vectors)) + "\n")
file.write("Validation set size: " + str(np.shape(reshaped_validation_input_vectors)) + "\n")
file.write("Loss function: NOT squared")
file.close()
print(" done!")
