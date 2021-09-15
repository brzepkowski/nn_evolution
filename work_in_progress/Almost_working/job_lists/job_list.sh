#!/bin/bash

# ==== Adadelta ====
python3 evolution_via_nn_simple.py softsign $1 $2 $3
python3 evolution_via_nn_simple.py tanh $1 $2 $3
