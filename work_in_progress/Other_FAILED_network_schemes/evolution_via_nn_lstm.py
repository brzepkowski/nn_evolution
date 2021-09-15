from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from scipy.linalg import expm
import sys

Sx = [[0 + 0j, 1 + 0j], [1 + 0j,  0 + 0j]]
Sy = [[0 + 0j, 0 - 1j], [0 + 1j,  0 + 0j]]
Sz = [[1 + 0j, 0 + 0j], [0 + 0j, -1 + 0j]]

def generate_random_vector(dims):
    vector = np.random.random(dims) + np.random.random(dims) * 1j
    magnitude = sum(x**2 for x in vector) ** .5
    return np.array([x/magnitude for x in vector])

def generate_Heisenberg_hamiltonian(N):
    J = 1.0
    H = np.zeros((2**N, 2**N), dtype='complex128')
    for i in range(N-1):
        # First particle
        left_size = (2**i, 2**i)
        right_size = (int((2**N)/(2**(i+1))), int((2**N)/(2**(i+1))))
        S_x_1 = np.kron(np.kron(np.ones(left_size), Sx), np.ones(right_size))
        S_y_1 = np.kron(np.kron(np.ones(left_size), Sy), np.ones(right_size))
        S_z_1 = np.kron(np.kron(np.ones(left_size), Sz), np.ones(right_size))
        # Second particle
        left_size = (2**(i+1), 2**(i+1))
        right_size = (int((2**N)/(2**(i+2))), int((2**N)/(2**(i+2))))
        # print("left_size: ", left_size, ", right_size: ", right_size)
        S_x_2 = np.kron(np.kron(np.ones(left_size), Sx), np.ones(right_size))
        S_y_2 = np.kron(np.kron(np.ones(left_size), Sy), np.ones(right_size))
        S_z_2 = np.kron(np.kron(np.ones(left_size), Sz), np.ones(right_size))
        H += (S_x_1*S_x_2) + (S_y_1*S_y_2) + (S_z_1*S_z_2)
        # print(S_1)
        # print(S_2)
    return H

def generate_training_set(training_set_size, vector_dims, H, t):
    if vector_dims != np.shape(H)[0]:
        print("Incompatible sizes of vector an Hamiltonian H.")
        sys.exit()
    U = expm(-1j * H * t)
    input_vectors = []
    output_vectors = []
    for i in range(training_set_size):
        psi_0 = generate_random_vector(vector_dims)
        psi_t = U.dot(psi_0)
        input_vectors.append(psi_0)
        output_vectors.append(psi_t)

    return (np.array(input_vectors), np.array(output_vectors))

def generate_test_set(training_set_size, vector_dims, H, t):
    return generate_training_set(training_set_size, vector_dims, H, t)

batch_size = 64  # Batch size for training.
epochs = 10  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 100  # Number of samples to train on.

N = 2
H = generate_Heisenberg_hamiltonian(N)
t = 1
(training_input_vectors, training_output_vectors) = generate_training_set(num_samples, 2**N, H, t)
(test_input_vectors, test_output_vectors) = generate_test_set(10, 2**N, H, t)

# We have to reshape the training and test sets, because an LSTM leyer takes
# as an input only 3D entries. Each dimension out of these 3 represents:
# - samples,
# - times steps,
# - features.
# We are adding one dimension at the end, which represents only one feature,
# that we want our LSTM layer to take into account
traing_input_vecs_shape = np.shape(training_input_vectors)
training_input_vectors = training_input_vectors.reshape(traing_input_vecs_shape[0], traing_input_vecs_shape[1], 1)
traing_output_vecs_shape = np.shape(training_output_vectors)
training_output_vectors = training_output_vectors.reshape(traing_output_vecs_shape[0], traing_output_vecs_shape[1], 1)

test_input_vecs_shape = np.shape(test_input_vectors)
test_input_vectors = test_input_vectors.reshape(test_input_vecs_shape[0], test_input_vecs_shape[1], 1)
test_output_vecs_shape = np.shape(test_output_vectors)
test_output_vectors = test_output_vectors.reshape(test_output_vecs_shape[0], test_output_vecs_shape[1], 1)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(2**N, 1, ))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(2**N, 1, ))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(1, activation='softmax') # Probably we could get rid off this layer
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
# print("training_input_vectors: ", np.shape(training_input_vectors))
# print("training_output_vectors: ", np.shape(training_output_vectors))
model.summary()

model.fit([training_input_vectors, training_input_vectors], training_output_vectors,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# # Save model
# model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
# decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
