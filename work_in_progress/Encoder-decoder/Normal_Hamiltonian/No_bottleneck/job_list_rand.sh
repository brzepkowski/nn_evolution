#!/bin/bash

#
# In this version network have normalization only at the end of the structure
# (not inbetween each hidden layer)
#

# ==== Adagrad ====
python3 random_vectors.py Adagrad cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adagrad hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adagrad huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adagrad logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adagrad mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adagrad squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adam ====
python3 random_vectors.py Adam huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adam logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adam squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adamax ====
python3 random_vectors.py Adamax cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adamax hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adamax huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adamax logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adamax mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors.py Adamax squared_hinge tanh 7 7 0.1 10 4000 1000
