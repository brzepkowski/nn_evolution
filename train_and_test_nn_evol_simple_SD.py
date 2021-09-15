# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

from nn_evolution_splitted_dataset.neural_networks.train_evol_simple import train_evol_simple
from nn_evolution_splitted_dataset.neural_networks.test_evol_simple import test_evol_simple
import sys

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Wrong number of arguments provided. Please give them in the following fashion:")
        print("- activaton function,")
        print("- N,")
        print("- dt,")
        print("- t_max,")
        print("- number_of_datasets.")
        sys.exit()

    activation = sys.argv[1]
    N = int(sys.argv[2])
    dt = float(sys.argv[3])
    t_max = float(sys.argv[4])
    number_of_datasets = int(sys.argv[5])

    for set_number in range(number_of_datasets):
        train_evol_simple(activation, N, dt, t_max, set_number)
    test_evol_simple(activation, N, dt, t_max)
