#!/bin/bash

#
# In this version network have normalization only at the end of the structure
# (not inbetween each hidden layer)
#

# ==== Adagrad ====
python3 evolution_via_nn_sequential_6_layers_v2.py Adagrad cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adagrad hinge tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adagrad huber_loss tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adagrad logcosh tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adagrad mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adagrad squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adam ====
python3 evolution_via_nn_sequential_6_layers_v2.py Adam huber_loss tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adam logcosh tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adam squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adamax ====
python3 evolution_via_nn_sequential_6_layers_v2.py Adamax cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adamax hinge tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adamax huber_loss tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adamax logcosh tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adamax mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 evolution_via_nn_sequential_6_layers_v2.py Adamax squared_hinge tanh 7 7 0.1 10 4000 1000
