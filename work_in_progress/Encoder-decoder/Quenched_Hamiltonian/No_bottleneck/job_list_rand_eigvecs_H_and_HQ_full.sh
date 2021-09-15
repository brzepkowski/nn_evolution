#!/bin/bash

# ==== Adadelta ====
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta cosine_proximity softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta huber_loss softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta logcosh softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta mean_absolute_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta mean_absolute_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta mean_squared_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta mean_squared_logarithmic_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta squared_hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adadelta squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adagrad ====
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad cosine_proximity softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad huber_loss softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad logcosh softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad mean_absolute_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad mean_absolute_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad mean_squared_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad mean_squared_logarithmic_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad squared_hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adam ====
python3 random_vectors_eigvecs_H_and_HQ.py Adam cosine_proximity softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam huber_loss softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam logcosh softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam mean_absolute_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam mean_absolute_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam mean_squared_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam mean_squared_logarithmic_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam squared_hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adamax ====
python3 random_vectors_eigvecs_H_and_HQ.py Adamax cosine_proximity softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax huber_loss softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax logcosh softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax mean_absolute_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax mean_absolute_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax mean_squared_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax mean_squared_logarithmic_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax squared_hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Nadam ====
python3 random_vectors_eigvecs_H_and_HQ.py Nadam cosine_proximity softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam huber_loss softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam logcosh softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam mean_absolute_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam mean_absolute_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam mean_squared_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam mean_squared_logarithmic_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam squared_hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Nadam squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== RMSprop ====
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop cosine_proximity softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop huber_loss softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop logcosh softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop mean_absolute_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop mean_absolute_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop mean_squared_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop mean_squared_logarithmic_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop squared_hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py RMSprop squared_hinge tanh 7 7 0.1 10 4000 1000

# ===== SGD =====
python3 random_vectors_eigvecs_H_and_HQ.py SGD cosine_proximity softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD huber_loss softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD logcosh softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD mean_absolute_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD mean_absolute_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD mean_squared_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD mean_squared_logarithmic_error softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD squared_hinge softsign 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py SGD squared_hinge tanh 7 7 0.1 10 4000 1000
