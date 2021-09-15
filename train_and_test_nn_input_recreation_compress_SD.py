# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

from nn_evolution_splitted_dataset.neural_networks.train_input_recreation_compress import train_input_recreation_compress
from nn_evolution_splitted_dataset.neural_networks.test_input_recreation_compress import test_input_recreation_compress
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


    subtracted_neurons = 2
    while subtracted_neurons <= (2**N):
        for set_number in range(number_of_datasets):
            train_input_recreation_compress(activation, N, dt, t_max, subtracted_neurons, set_number)
        test_input_recreation_compress(activation, N, dt, t_max, subtracted_neurons)
        subtracted_neurons *= 2
