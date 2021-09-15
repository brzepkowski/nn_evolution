#!/bin/bash

conda activate python

# ==== Adadelta ====
python3 evolution_via_nn_simple.py Adam logcosh tanh $1 $2 $3
