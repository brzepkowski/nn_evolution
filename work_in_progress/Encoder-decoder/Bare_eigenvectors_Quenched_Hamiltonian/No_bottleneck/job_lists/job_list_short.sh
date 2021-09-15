#!/bin/bash

#
# In this version network have normalization only at the end of the structure
# (not inbetween each hidden layer)
#

# ==== Adagrad ====
python3 evolution_via_nn_simple.py Adagrad cosine_proximity tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad hinge tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad mean_squared_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adagrad squared_hinge tanh $1 $2 $3

# ==== Adam ====
python3 evolution_via_nn_simple.py Adam huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adam logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adam squared_hinge tanh $1 $2 $3

# ==== Adamax ====
python3 evolution_via_nn_simple.py Adamax cosine_proximity tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax hinge tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax huber_loss tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax logcosh tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax mean_squared_error tanh $1 $2 $3
python3 evolution_via_nn_simple.py Adamax squared_hinge tanh $1 $2 $3
