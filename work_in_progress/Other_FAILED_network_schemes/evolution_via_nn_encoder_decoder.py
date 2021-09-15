from keras.layers import Input, Dense, concatenate
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import generators

N_min = 6
N_max = 7
t = 0.1
t_total = 1
train_num_samples = 10000
test_num_samples = 1000

x = list(np.arange(t, t_total+t, t))
y = []
legend = []
for N in range(N_min, N_max+1):
    print("#################################")
    print("###### N: ", N)
    print("#################################")
    legend.append(str(N))
    H = generators.generate_Heisenberg_hamiltonian(N)
    U_t = expm(-1j * H * t)
    U_t_simple = np.identity(2**N) - (1j * H * t)

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

    # ==============================================================================
    input = Input(shape=(2*(2**N),))
    encode_1 = Dense(2*(2**(N-1)), activation='tanh')(input)
    encode_2 = Dense(2*(2**(N-2)), activation='tanh')(encode_1)
    decode_2 = Dense(2*(2**(N-2)), activation='tanh')(encode_2)

    merge_2_1 = concatenate([decode_2, encode_1])
    decode_1 = Dense(2*(2**(N-1)), activation='tanh')(merge_2_1)
    merge_1_0 = concatenate([decode_1, input])
    output = Dense(2*(2**N), activation='tanh')(merge_1_0)
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

    evolved_test_vectors_full_op = test_input_vectors
    evolved_test_vectors_simple_op = test_input_vectors
    predicted_test_vectors = reshaped_test_input_vectors
    mean_overlap_per_timestep = []
    for time_step in np.arange(t, t_total+t, t):
        # Evolution using full operator
        evolved_test_vectors_full_op = generators.evolve_whole_set(evolved_test_vectors_full_op, U_t)
        evolved_test_vectors_full_op_reshaped = generators.flatten_set(generators.reshape_set(evolved_test_vectors_full_op))
        # Evolution using simple operator
        evolved_test_vectors_simple_op = generators.evolve_whole_set(evolved_test_vectors_simple_op, U_t)
        evolved_test_vectors_simple_op_reshaped = generators.flatten_set(generators.reshape_set(evolved_test_vectors_simple_op))
        # Evolution using neural network
        predicted_test_vectors = model.predict(predicted_test_vectors)

        overlaps = generators.overlap_of_reshaped_sets(evolved_test_vectors_full_op_reshaped, predicted_test_vectors)
        # print(np.mean(overlaps))
        mean_overlap_per_timestep.append(np.mean(overlaps))
    y.append(mean_overlap_per_timestep)

x_max = x[-1]
y = np.array(y).T.tolist()
plt.title("Sequential - no hidden layers")
plt.plot(x, y, 'o-')
plt.xlabel('Time')
plt.ylabel('Overlap')
plt.gca().legend(legend)
plt.xlim(xmin=0, xmax=x_max)
plt.show()
