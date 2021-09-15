
from keras.layers import Input, Dense, concatenate
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import generators

N_min = 3
N_max = 7
t_total = 10
train_num_samples = 10000
test_num_samples = 1000

x = list(range(1, t_total+1))
y = []
legend = []
for N in range(N_min, N_max+1):
    legend.append(str(N))
    H = generators.generate_Heisenberg_hamiltonian(N)
    t = 1
    U_t = expm(-1j * H * t)

    (training_input_vectors, training_output_vectors) = generators.generate_training_set(train_num_samples, 2**N, H, t)
    (test_input_vectors, test_output_vectors) = generators.generate_test_set(test_num_samples, 2**N, H, t)

    # Reshape generated sets, to extract imaginary parts of numbers as second dimension in a vector
    (reshaped_training_input_vectors, reshaped_training_output_vectors) = generators.reshape_two_sets(training_input_vectors, training_output_vectors)
    (reshaped_test_input_vectors, reshaped_test_output_vectors) = generators.reshape_two_sets(test_input_vectors, test_output_vectors)

    # Flatten the data, so that it can be injected into neural network
    reshaped_training_input_vectors = generators.flatten_set(reshaped_training_input_vectors)
    reshaped_training_output_vectors = generators.flatten_set(reshaped_training_output_vectors)

    reshaped_test_input_vectors = generators.flatten_set(reshaped_test_input_vectors)
    reshaped_test_output_vectors = generators.flatten_set(reshaped_test_output_vectors)

    # ============================= Pierwsza wersja ================================
    input = Input(shape=(2*(2**N),))
    encode_1 = Dense(2*(2**(N-1)), activation='tanh')(input)
    # merge_1 = concatenate([input, encode_1])
    encode_2 = Dense(2*(2**(N-2)), activation='tanh')(encode_1)
    decode_2 = Dense(2*(2**(N-2)), activation='tanh')(encode_2)

    merge_2_1 = concatenate([decode_2, encode_1])
    decode_1 = Dense(2*(2**(N-1)), activation='tanh')(merge_2_1)
    merge_1_0 = concatenate([decode_1, input])
    output = Dense(2*(2**N), activation='tanh')(merge_1_0)
    # ============================= Druga wersja ================================
    # input = Input(shape=(2*(2**N),))
    # # hidden_layer_0 = Dense(2*(2**N), activation='tanh')(input)
    # # hidden_layer_1 = Dense(2*(2**N), activation='tanh')(hidden_layer_0)
    # output = Dense(2*(2**N), activation='tanh')(input)
    # ==============================================================================

    model = Model(input, output)
    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    history = model.fit(reshaped_training_input_vectors, reshaped_training_output_vectors,
                                        batch_size=256,
                                        epochs=100,
                                        shuffle=True)
                                        # validation_data=(reshaped_test_input_vectors, reshaped_test_output_vectors))

    score, acc = model.evaluate(reshaped_test_input_vectors, reshaped_test_output_vectors,
                                                batch_size=256)

    print('Test score:', score)
    print('Test accuracy:', acc)

    evolved_test_vectors = test_input_vectors
    predicted_test_vectors = reshaped_test_input_vectors
    mean_overlap_per_timestep = []
    for time_step in range(t_total):
        evolved_test_vectors = generators.evolve_whole_set(evolved_test_vectors, U_t)
        evolved_test_vectors_reshaped = generators.flatten_set(generators.reshape_set(evolved_test_vectors))
        predicted_test_vectors = model.predict(predicted_test_vectors)

        overlaps = generators.overlap_of_reshaped_sets(evolved_test_vectors_reshaped, predicted_test_vectors)
        # print(np.mean(overlaps))
        mean_overlap_per_timestep.append(np.mean(overlaps))
    y.append(mean_overlap_per_timestep)

y = np.array(y).T.tolist()
plt.title("Encoder-decoder (tensors merged)")
plt.plot(x, y, 'o-')
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.gca().legend(legend)
plt.xlim(xmin=1, xmax=t_total)
plt.show()
