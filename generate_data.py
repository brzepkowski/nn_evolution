# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

from nn_evolution.data_generation.exact_diag_and_operators_generation import gen_and_diagonalize_hamil_before_quench
from nn_evolution.data_generation.exact_diag_and_operators_generation import gen_hamil_after_quench_and_evol_operator
from nn_evolution.data_generation.exact_diag_and_operators_generation import gen_correlation_operators

from nn_evolution.data_generation.exact_evol_and_sets_generation import conduct_exact_evol_and_gen_sets

from nn_evolution.data_generation.exact_properties_calculation import calculate_properties

import sys

if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- N,")
        print("- J,")
        print("- hz,")
        print("- dt,")
        print("- t_max.")
        sys.exit()

    N = int(sys.argv[1])
    J = float(sys.argv[2])
    hz = float(sys.argv[3])
    dt = float(sys.argv[4])
    t_max = float(sys.argv[5])

    gen_and_diagonalize_hamil_before_quench(N, J, hz, dt, t_max)
    gen_hamil_after_quench_and_evol_operator(N, J, dt, t_max)
    gen_correlation_operators(N, dt, t_max)

    conduct_exact_evol_and_gen_sets(N, dt, t_max)

    calculate_properties(N, dt, t_max)
