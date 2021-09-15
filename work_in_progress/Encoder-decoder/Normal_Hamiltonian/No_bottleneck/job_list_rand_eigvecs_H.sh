#!/bin/bash

#
# In this version network have normalization only at the end of the structure
# (not inbetween each hidden layer)
#

# ==== Adagrad ====
python3 random_vectors_eigvecs_H.py Adagrad cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adagrad hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adagrad huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adagrad logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adagrad mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adagrad squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adam ====
python3 random_vectors_eigvecs_H.py Adam huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adam logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adam squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adamax ====
python3 random_vectors_eigvecs_H.py Adamax cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adamax hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adamax huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adamax logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adamax mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H.py Adamax squared_hinge tanh 7 7 0.1 10 4000 1000
