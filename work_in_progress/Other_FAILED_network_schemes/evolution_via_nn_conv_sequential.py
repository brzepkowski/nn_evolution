
from keras.layers import Input, Dense, Conv1D, UpSampling1D, ZeroPadding1D, concatenate
from keras.models import Model, Sequential
import numpy as np
import generators
import sys

N = 3
H = generators.generate_Heisenberg_hamiltonian(N)
t = 1
train_num_samples = 10000
test_num_samples = 1000

(training_input_vectors, training_output_vectors) = generators.generate_training_set(train_num_samples, 2**N, H, t)
(test_input_vectors, test_output_vectors) = generators.generate_test_set(test_num_samples, 2**N, H, t)

# Reshape generated sets, to extract imaginary parts of numbers as second dimension in a vector
(reshaped_training_input_vectors, reshaped_training_output_vectors) = generators.reshape(training_input_vectors, training_output_vectors)
(reshaped_test_input_vectors, reshaped_test_output_vectors) = generators.reshape(test_input_vectors, test_output_vectors)

# Flatten the data, so that it can be injected into neural network
reshaped_training_input_vectors = reshaped_training_input_vectors.reshape((len(reshaped_training_input_vectors),
                                                                np.prod(reshaped_training_input_vectors.shape[1:]), 1))
reshaped_training_output_vectors = reshaped_training_output_vectors.reshape((len(reshaped_training_output_vectors),
                                                                np.prod(reshaped_training_output_vectors.shape[1:]), 1))

reshaped_test_input_vectors = reshaped_test_input_vectors.reshape((len(reshaped_test_input_vectors),
                                                                np.prod(reshaped_test_input_vectors.shape[1:]), 1))
reshaped_test_output_vectors = reshaped_test_output_vectors.reshape((len(reshaped_test_output_vectors),
                                                                np.prod(reshaped_test_output_vectors.shape[1:]), 1))


# print(np.shape(reshaped_training_input_vectors))
# sys.exit()
# input = Input(shape=(2*(2**N),1,))

# filters - how many sliding windows will run through the data
model = Sequential()
model.add(Conv1D(filters=1, kernel_size=5, activation='tanh', input_shape=(None, 1)))
model.add(Conv1D(filters=1, kernel_size=5, activation='tanh', input_shape=(None, 1)))

model.add(ZeroPadding1D(padding=2))
model.add(Conv1D(filters=1, kernel_size=5, activation='tanh', padding='same'))

model.add(ZeroPadding1D(padding=2))
model.add(Conv1D(filters=1, kernel_size=5, activation='tanh', padding='same'))

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

predicted_vectors = model.predict(reshaped_training_input_vectors)

print("training_input_vectors[0]: ", reshaped_training_input_vectors[0])
print("training_output_vectors[0]: ", reshaped_training_output_vectors[0])
print("predicted_vectors[0]: ", predicted_vectors[0])
