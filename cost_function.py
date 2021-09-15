import tensorflow as tf
import numpy as np
from numpy.random import random
from numpy.linalg import norm

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

    overlap_real = tf.math.reduce_sum(reals, axis=1)
    overlap_imag = tf.math.reduce_sum(imags, axis=1)

    print("overlap_real: ", overlap_real)
    print("overlap_imag: ", overlap_imag)

    result = tf.abs(overlap_real) - tf.abs(overlap_imag)
    print("result: ", result)

    # overlap_squared = (overlap_real*overlap_real) + (overlap_imag*overlap_imag)

    # Optimize only overlap^2
    # result = 1.0-tf.reduce_mean(overlap_squared)
    # Optimize both overlap^2 and log_cosh
    # result = 1.0-tf.reduce_mean(overlap_squared) + tf.keras.losses.log_cosh(y_true, y_pred)
    # Optimize just log_cosh
    # result = tf.keras.losses.log_cosh(y_true, y_pred)
    # Optimize just the real part of the overlap
    # result = 1.0 - tf.reduce_mean(tf.abs(overlap_real - overlap_imag))
    result = 1.0 - tf.reduce_mean(result)

    return result

def main():
    ket_0 = np.array(random(8))
    ket_norm = norm(ket_0)
    ket_0 /= ket_norm

    ket_1 = np.array(random(8))
    ket_norm = norm(ket_1)
    ket_1 /= ket_norm

    ket_2 = np.array(random(8))
    ket_norm = norm(ket_2)
    ket_2 /= ket_norm

    ket_3 = np.array(random(8))
    ket_norm = norm(ket_3)
    ket_3 /= ket_norm

    batch_0 = tf.constant([ket_0, ket_1])
    batch_1 = tf.constant([ket_2, ket_3])

    cost = my_cost(batch_0, batch_1)
    print("cost: ", cost)

    cost_with_self = my_cost(batch_0, batch_0)
    print("cost_with_self: ", cost_with_self)

main()
