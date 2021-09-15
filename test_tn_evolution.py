# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

from nn_evolution.tensor_networks.MPS_evolution import conduct_tebd_time_evolution
import sys

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- N,")
        # print("- chi_max,")
        print("- dt,")
        print("- t_max.")
        sys.exit()

    N = int(sys.argv[1])
    # chi_max = int(sys.argv[2])
    dt = float(sys.argv[2])
    t_max = float(sys.argv[3])

    # chi_max = int(2**N)
    chi_max = 12
    while chi_max >= 2:
        conduct_tebd_time_evolution(N, chi_max, dt, t_max)
        # chi_max = int(chi_max / 2)
        chi_max -= 1
