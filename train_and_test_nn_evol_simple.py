# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

from nn_evolution.neural_networks.train_evol_simple import train_evol_simple
from nn_evolution.neural_networks.test_evol_simple import test_evol_simple
import sys

if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Wrong number of arguments provided. Please give them in the following fashion:")
        print("- activaton function,")
        print("- N,")
        print("- dt,")
        print("- t_max.")
        sys.exit()

    activation = sys.argv[1]
    N = int(sys.argv[2])
    dt = float(sys.argv[3])
    t_max = float(sys.argv[4])

    train_evol_simple(activation, N, dt, t_max)
    test_evol_simple(activation, N, dt, t_max)
