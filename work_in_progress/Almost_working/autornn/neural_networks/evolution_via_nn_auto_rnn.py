import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import sys
from misc import overlap_of_reshaped
from numpy.linalg import norm
from keras.losses import mean_squared_error

if len(sys.argv) != 4:
    print("Wrong number of arguments provided. Please give them in the following fashion:")
    print("- N,")
    print("- timestep,")
    print("- total_evolution_time.")
    sys.exit()

N = int(sys.argv[1])
t = float(sys.argv[2])
t_total = float(sys.argv[3])

################################################################################
# Load the training, validation and test sets
################################################################################

with open('training_input.npy', 'rb') as f:
    train_df = np.load(f)
# with open('training_output.npy', 'rb') as f:
#     reshaped_training_output_vectors = np.load(f)
with open('validation_input.npy', 'rb') as f:
    val_df = np.load(f)
# with open('validation_output.npy', 'rb') as f:
#     reshaped_validation_output_vectors = np.load(f)
with open('test_input.npy', 'rb') as f:
    test_df = np.load(f)

# with open('test_output.npy', 'rb') as f:
#     reshaped_test_output_vectors = np.load(f)
# with open('test_input_simple_op.npy', 'rb') as f:
#     reshaped_test_input_vectors_simple_op = np.load(f)
# with open('test_output_simple_op.npy', 'rb') as f:
#     reshaped_test_output_vectors_simple_op = np.load(f)
# with open('test_eigenvalues.npy', 'rb') as f:
#     test_eigenvalues = np.load(f)
# Load data for making predictions
# with open('nn_input_for_predictions.npy', 'rb') as f:
#     nn_input_for_predictions = np.load(f)
# Load eigenvalues of H
# eigenvalues_H = load_vector_from_file("eigenvalues_H.dat", 2**N).astype(np.float64)

"""
################################################################################
# Split the data
################################################################################

column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
"""

# np.set_printoptions(threshold=sys.maxsize)

################################################################################
# Data windowing
################################################################################

# 1. Indexes and offsets

class WindowGenerator():
    def __init__(self, input_width, label_width, offset, batch_size, time_series_length,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        #### Additional fields ####
        self.batch_size = batch_size
        self.time_series_length = time_series_length

        # Work out the label column indices.
        # self.label_columns = label_columns
        # if label_columns is not None:
        #   self.label_columns_indices = {name: i for i, name in
        #                                 enumerate(label_columns)}
        # self.column_indices = {name: i for i, name in
        #                        enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.offset = offset

        self.total_window_size = input_width + offset

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    @property
    def train(self):
      return self.make_dataset(self.train_df, self.time_series_length)

    @property
    def val(self):
      return self.make_dataset(self.val_df, self.time_series_length)

    @property
    def test(self):
      return self.make_dataset(self.test_df, self.time_series_length)

    @property
    def example(self):
      """Get and cache an example batch of `inputs, labels` for plotting."""
      result = getattr(self, '_example', None)
      if result is None:
        # No example batch was found, so get one from the `.train` dataset
        # print("(1) ", self.train)
        # print("(2) ", iter(self.train))
        # print("(3) ", next(iter(self.train)))
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
      return result

# 2. Split

    def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      # if self.label_columns is not None:
      #     labels = tf.stack(
      #           [labels[:, :, self.column_indices[name]] for name in self.label_columns],
      #           axis=-1)

      # Slicing doesn't preserve static shape information, so set the shapes
      # manually. This way the `tf.data.Datasets` are easier to inspect.
      inputs.set_shape([None, self.input_width, None])
      labels.set_shape([None, self.label_width, None])

      return inputs, labels

# 3. Plot
    """
    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
      inputs, labels = self.example
      plt.figure(figsize=(12, 8))
      plot_col_index = self.column_indices[plot_col]
      max_n = min(max_subplots, len(inputs))
      for n in range(max_n):
        plt.subplot(3, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
          label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
          label_col_index = plot_col_index

        if label_col_index is None:
          continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
          predictions = model(inputs)
          plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)

        if n == 0:
          plt.legend()

      plt.xlabel('Time [h]')
    """

# 4. Create tf.data.Datasets

    def make_dataset(self, data, time_series_length):
        # print("[0] DATA: \n", data)

        ds = None
        i = 0
        while (i+1)*time_series_length < len(data):
            set_beginning = i*time_series_length
            set_ending = set_beginning+time_series_length

            partial_data = data[set_beginning:set_ending]

            partial_data = np.array(partial_data, dtype=np.float32)
            # print("[1] PARTIAL_DATA: \n", partial_data)
            # Because "targets=None" dataset yields only batches of sequences.
            # ds will hold many batches of data. Each batch contains "batch_size" of time_series,
            # each of length input_width + offset. Each such time_series contains "num_features" of features.
            # Total number of batches is equal to data.length / batch_size.
            partial_ds = tf.keras.preprocessing.timeseries_dataset_from_array(
              data=partial_data,
              targets=None,
              sequence_length=self.total_window_size, # = input_width + offset
              sequence_stride=1,
              # shuffle=True,
              shuffle=False,
              # batch_size=32,)
              batch_size=self.batch_size,)
            # print("[2] PARTIAL DS: \n", partial_ds)
            # print("--- PARTIAL ---")
            # for batch in partial_ds:
            #   print(batch)

            if ds == None:
              ds = partial_ds
            else:
              ds = ds.concatenate(partial_ds)

            # print("----- DS -----")
            # j = 0
            # for batch in ds:
            #   # print(batch)
            #   j += 1

            # print("ds.length = ", j)

            i += 1

        # print("[2] DATA: \n", ds)

        # Below line generates pairs (inputs, labels). Or rather batches of inputs and labels.
        # In the split_window method we are slicing only the "middle" axis of the data provided.
        # This corresponds to the time dimension. The data provided to this function is of shape:
        # (batch_size, time, features),
        # so we are taking first "input_width" timesteps to the inputs sets, and last "label_width"
        # timesteps to the labels sets.
        ds = ds.map(self.split_window)

        # print("[3] DATA: \n", ds)

        return ds

# LSTMCell processes one step within the whole time sequence input, whereas tf.keras.layer.LSTM processes the whole sequence.
class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.units = units
        self.out_steps = out_steps
        self.num_features = num_features
        # self.lstm_cell = tf.keras.layers.LSTMCell(units)
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)
        self.normalization = tf.keras.layers.Lambda(lambda t: tf.keras.backend.l2_normalize(1000*t, axis=1))

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)

        # predictions.shape => (batch, features)
        # not_normalized_prediction = self.dense(x)
        # prediction = self.normalization(not_normalized_prediction)

        prediction = self.dense(x)
        prediction = self.normalization(prediction)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)

        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            prediction = self.normalization(prediction)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions

def my_cost(y_true, y_pred):
    y_true_even = y_true[:,:,::2] # These are real values
    y_true_odd = y_true[:,:,1::2] # These are imaginary numbers
    # print("y_true: ", y_true)
    # print("y_true_even: ", y_true_even)
    # print("y_true_odd: ", y_true_odd)

    # print()
    # print("="*40)
    # print()

    y_pred_even = y_pred[:,:,::2] # These are real values
    y_pred_odd = y_pred[:,:,1::2] # These are imaginary numbers
    # print("y_pred: ", y_pred)
    # print("y_pred_even: ", y_pred_even)
    # print("y_pred_odd: ", y_pred_odd)

    real_parts = tf.math.multiply(y_true_even,y_pred_even)
    imag_parts = tf.math.multiply(y_true_odd,y_pred_odd)

    # print()
    # print("#"*40)
    # print("#"*40)
    # print()

    # print("real_parts: ", real_parts)
    # print("imag_parts: ", imag_parts)

    sum_of_reals = tf.math.reduce_sum(real_parts, axis=2)
    sum_of_imags = tf.math.reduce_sum(imag_parts, axis=2)

    # print("sum_of_reals: ", sum_of_reals)
    # print("sum_of_imags: ", sum_of_imags)

    # result = sum_of_reals - sum_of_imags
    result = sum_of_reals + sum_of_imags
    # print("(0) result: ", result)
    result = tf.square(result)
    # print("(1) result: ", result)
    result = 1.0-tf.reduce_mean(result)
    # print("(2) result: ", result)

    return result

def compile_and_fit(model, window, patience=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                    patience=patience,
                                                    mode='min')

    # model.compile(loss=tf.losses.MeanSquaredError(),
    #             optimizer=tf.optimizers.Adam(),
    #             # metrics=[tf.metrics.MeanAbsoluteError()])
    #             metrics=['accuracy'])

    model.compile(loss=my_cost,
                optimizer=tf.optimizers.Adam())
                # metrics=[tf.metrics.MeanAbsoluteError()])
                # metrics=['val_loss'])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
    return history

MAX_EPOCHS = 2000
INPUT_WIDTH = 10
LABEL_WIDTH = OUT_STEPS = OFFSET = 30
TS_LENGTH = int(t_total/t)
NUM_FEATURES = UNITS = 2*(2**N)
BATCH_SIZE = 1000

# It might be smart to use the same size of label_width and offset, due to the way in which split_window
# in the WindowGenerator works
multi_window = WindowGenerator(input_width=INPUT_WIDTH,
                               label_width=LABEL_WIDTH,
                               offset=OFFSET,
                               batch_size=BATCH_SIZE,
                               time_series_length=TS_LENGTH)

feedback_model = FeedBack(units=UNITS, out_steps=OUT_STEPS, num_features=NUM_FEATURES)

# This is just a warmup, thus it will give just one prediction!
# prediction, state = feedback_model.warmup(multi_window.example[0])

history = compile_and_fit(feedback_model, multi_window)

IPython.display.clear_output()

all_overlaps = []

for batch in iter(multi_window.test):
# for batch in iter(multi_window.train):
    # print(np.shape(batch[1]))

    predictions = feedback_model(batch[0])
    # print("Cost: ", my_cost(batch[1], predictions))

    # print("Exact: ", batch[1])
    # print("Predicted: ", predictions)
    for i in range(len(batch[1][0])):
        print("i: ", i, end="")
        exact = np.array(batch[1][0][i])
        predicted = np.array(predictions[0][i])
        # print("True norm: ", norm(exact))
        # print("Pred norm: ", norm(predicted))
        overlap = overlap_of_reshaped(exact, predicted).real
        print(" - overlap: ", overlap)
        # all_overlaps.append(overlap)

# print("Mean overlap: ", np.mean(all_overlaps))


# predictions_many_steps = feedback_model(multi_window.example[0])
# print("Input: ", multi_window.example[0])
# print("Predictions: ", predictions_many_steps)
