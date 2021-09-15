#!/bin/bash

# ==== Adadelta ====
python3 evolution_via_nn_sequential_6_layers.py Adadelta cosine_proximity softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta hinge tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta huber_loss softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta huber_loss tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta logcosh softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta logcosh tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta mean_absolute_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta mean_squared_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta mean_squared_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta squared_hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adadelta squared_hinge tanh 7 7 0.1 10 1000 1000

# ==== Adagrad ====
python3 evolution_via_nn_sequential_6_layers.py Adagrad cosine_proximity tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad hinge tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad huber_loss softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad huber_loss tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad logcosh softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad logcosh tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad mean_absolute_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad mean_squared_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad mean_squared_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad mean_squared_logarithmic_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad squared_hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adagrad squared_hinge tanh 7 7 0.1 10 1000 1000

# ==== Adam ====
python3 evolution_via_nn_sequential_6_layers.py Adam cosine_proximity softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam cosine_proximity tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam huber_loss softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam huber_loss tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam logcosh softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam logcosh tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam mean_absolute_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam mean_absolute_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam mean_squared_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam mean_squared_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam mean_squared_logarithmic_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam squared_hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adam squared_hinge tanh 7 7 0.1 10 1000 1000

# ==== Adamax ====
python3 evolution_via_nn_sequential_6_layers.py Adamax cosine_proximity softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax cosine_proximity tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax hinge tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax huber_loss softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax huber_loss tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax logcosh softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax logcosh tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax mean_absolute_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax mean_absolute_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax mean_squared_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax mean_squared_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax mean_squared_logarithmic_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax squared_hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Adamax squared_hinge tanh 7 7 0.1 10 1000 1000

# ==== Nadam ====
python3 evolution_via_nn_sequential_6_layers.py Nadam cosine_proximity softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam cosine_proximity tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam hinge tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam huber_loss softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam huber_loss tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam logcosh softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam logcosh tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam mean_absolute_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam mean_absolute_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam mean_squared_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam mean_squared_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam mean_squared_logarithmic_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py Nadam squared_hinge softsign 7 7 0.1 10 1000 1000

# ==== RMSprop ====
python3 evolution_via_nn_sequential_6_layers.py RMSprop cosine_proximity softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py RMSprop cosine_proximity tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py RMSprop hinge tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py RMSprop mean_absolute_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py RMSprop mean_absolute_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py RMSprop squared_hinge tanh 7 7 0.1 10 1000 1000

# ===== SGD =====
python3 evolution_via_nn_sequential_6_layers.py SGD categorical_hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py SGD categorical_hinge tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py SGD cosine_proximity softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py SGD cosine_proximity tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py SGD huber_loss tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py SGD mean_absolute_error softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py SGD mean_absolute_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py SGD mean_squared_error tanh 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py SGD squared_hinge softsign 7 7 0.1 10 1000 1000
python3 evolution_via_nn_sequential_6_layers.py SGD squared_hinge tanh 7 7 0.1 10 1000 1000
