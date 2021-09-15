#!/bin/bash

# ==== Adadelta ====
python3 evolution_via_nn_simple.py Adadelta cosine_proximity softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta cosine_proximity tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta hinge tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta huber_loss softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta logcosh softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta mean_absolute_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta mean_absolute_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta mean_squared_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta mean_squared_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta mean_squared_logarithmic_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta squared_hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adadelta squared_hinge tanh $1 $2 $3

# ==== Adagrad ====
python3 evolution_via_nn_simple.py Adagrad cosine_proximity softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad cosine_proximity tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad hinge tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad huber_loss softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad logcosh softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad mean_absolute_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad mean_absolute_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad mean_squared_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad mean_squared_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad mean_squared_logarithmic_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad squared_hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad squared_hinge tanh $1 $2 $3

# ==== Adam ====
python3 evolution_via_nn_simple.py Adam cosine_proximity softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adam cosine_proximity tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adam hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adam hinge tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adam huber_loss softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adam huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adam logcosh softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adam logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adam mean_absolute_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adam mean_absolute_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adam mean_squared_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adam mean_squared_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adam mean_squared_logarithmic_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adam squared_hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adam squared_hinge tanh $1 $2 $3

# ==== Adamax ====
python3 evolution_via_nn_simple.py Adamax cosine_proximity softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adamax cosine_proximity tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adamax hinge tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax huber_loss softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adamax huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax logcosh softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adamax logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax mean_absolute_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adamax mean_absolute_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax mean_squared_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adamax mean_squared_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax mean_squared_logarithmic_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adamax squared_hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Adamax squared_hinge tanh $1 $2 $3

# ==== Nadam ====
python3 evolution_via_nn_simple.py Nadam cosine_proximity softsign $1 $2 $3
python3 evolution_via_nn_simple.py Nadam cosine_proximity tanh $1 $2 $3
python3 evolution_via_nn_simple.py Nadam hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Nadam hinge tanh $1 $2 $3
python3 evolution_via_nn_simple.py Nadam huber_loss softsign $1 $2 $3
python3 evolution_via_nn_simple.py Nadam huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py Nadam logcosh softsign $1 $2 $3
python3 evolution_via_nn_simple.py Nadam logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py Nadam mean_absolute_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Nadam mean_absolute_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Nadam mean_squared_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Nadam mean_squared_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Nadam mean_squared_logarithmic_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py Nadam squared_hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py Nadam squared_hinge tanh $1 $2 $3

# ==== RMSprop ====
python3 evolution_via_nn_simple.py RMSprop cosine_proximity softsign $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop cosine_proximity tanh $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop hinge tanh $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop huber_loss softsign $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop logcosh softsign $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop mean_absolute_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop mean_absolute_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop mean_squared_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop mean_squared_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop mean_squared_logarithmic_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop squared_hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py RMSprop squared_hinge tanh $1 $2 $3

# ===== SGD =====
python3 evolution_via_nn_simple.py SGD cosine_proximity softsign $1 $2 $3
python3 evolution_via_nn_simple.py SGD cosine_proximity tanh $1 $2 $3
python3 evolution_via_nn_simple.py SGD hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py SGD hinge tanh $1 $2 $3
python3 evolution_via_nn_simple.py SGD huber_loss softsign $1 $2 $3
python3 evolution_via_nn_simple.py SGD huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py SGD logcosh softsign $1 $2 $3
python3 evolution_via_nn_simple.py SGD logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py SGD mean_absolute_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py SGD mean_absolute_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py SGD mean_squared_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py SGD mean_squared_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py SGD mean_squared_logarithmic_error softsign $1 $2 $3
python3 evolution_via_nn_simple.py SGD squared_hinge softsign $1 $2 $3
python3 evolution_via_nn_simple.py SGD squared_hinge tanh $1 $2 $3
