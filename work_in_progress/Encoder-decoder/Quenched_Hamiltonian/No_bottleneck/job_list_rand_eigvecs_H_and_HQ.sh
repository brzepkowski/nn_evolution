#!/bin/bash

#
# In this version network have normalization only at the end of the structure
# (not inbetween each hidden layer)
#

# ==== Adagrad ====
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adagrad squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adam ====
python3 random_vectors_eigvecs_H_and_HQ.py Adam huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adam squared_hinge tanh 7 7 0.1 10 4000 1000

# ==== Adamax ====
python3 random_vectors_eigvecs_H_and_HQ.py Adamax cosine_proximity tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax hinge tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax huber_loss tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax logcosh tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax mean_squared_error tanh 7 7 0.1 10 4000 1000
python3 random_vectors_eigvecs_H_and_HQ.py Adamax squared_hinge tanh 7 7 0.1 10 4000 1000
