# Copyright Bartosz Rzepkowski, Michał Kupczyński, Wrocław 2020

import tenpy.linalg.np_conserved as npc
from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Chain
from tenpy.models.spins import SpinModel, SpinChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from collections import Counter
import numpy as np
from os import path
import sys, math

def HeisenbergModel(lattice, J, h=0, bcy="ladder", bcx="open"):
    Jx = Jy = Jz = J
    hx = hy = hz = h
    muJ = 0
    D = 0
    E = 0
    model_params = {
        # "S": 1.5,  # Spin 3/2
        # "lattice": "Honeycomb",
        "lattice": lattice,
        # "bc_MPS": "finite",
        # "bc_y": bcy,
        # "bc_x": bcx,
        # # "Lx": Lx,  # defines cylinder circumference
        # # "Ly": Ly,  # defines cylinder circumference
        "Jx": Jx,
        "Jy": Jy,
        "Jz": Jz,
        "hx": hx,
        "hy": hy,
        "hz": hz,
        "muJ": muJ,
        "D": D,
        "E": E,
        # "conserve": "best"   # Heisenberg coupling
    }
    # model = SpinModel(model_params)
    model = SpinChain(model_params)
    return model

# N - liczba wszystkich spinów,
# Sz - domyslna podprzestrzeń.
def prep_initial_state(N, Sz):
    psi = ["0.5"]*N # Zainicjalizuj stan początkowy ze wszystkimi spinami w górę
    current_spin = N-1
    current_Sz = N*0.5

    while current_spin >= 0 and current_Sz != Sz:
        if float(psi[current_spin]) > -0.5:
            psi[current_spin] = str(float(psi[current_spin]) - 1.0)
            current_Sz -= 1.0
        else:
            current_spin -= 1

    psi =  np.random.permutation(psi)
    # print(psi)
    return psi

# n - liczba stanów wzbudzonych
def run_dmrg(model,lattice,s,n=1,chi_max=10000,svd_min=1.e-20,mixer=True,generate_maps=True):
    N_sites = model.lat.N_sites
    results = []
    sites = model.lat.mps_sites()
    excited_states = []
    print("###### Sector Sz: ", s, " ######")
    max_permutations = max_number_of_permutations(prep_initial_state(N_sites ,s))
    # print("max_permutations: ", max_permutations)
    for j in range(n):
        print("# state:", j+1, " / ", n, end="\r")
        State = prep_initial_state(N_sites, s)
        psi = MPS.from_product_state(sites, State, "finite")
        mixer_params = {}
        dmrg_params = {"trunc_params": {"chi_max": chi_max, "svd_min": svd_min}, "mixer": mixer,"mixer_params":mixer_params,
            'combine': True,"orthogonal_to": excited_states, 'verbose':False}
        info = dmrg.run(psi, model, dmrg_params)
        # for j in range(len(psi._B)):
        #     print(psi._B[j])

        # Energy
        energy = info['E']
        # print("\nEnergy: ", energy)
        # bond_energies = model.bond_energies(psi)
        # print("bond_energies: ", bond_energies)
        # print("sum(bond_energies): ", sum(bond_energies))
        # print()

        #  Sz (average and total)
        Sz_per_site = psi.expectation_value("Sz")
        Sz_per_site_str = list_to_string(Sz_per_site)
        average_Sz = np.sum(Sz_per_site) / N_sites
        Sz_total = np.sum(Sz_per_site)

        """
        # Correlations
        nearest_neighbors = get_nearest_neighbors(lattice)
        correlation_Sp_Sm = psi.correlation_function('Sp', 'Sm')
        correlation_Sm_Sp = psi.correlation_function('Sm', 'Sp')
        correlations = (correlation_Sp_Sm + correlation_Sm_Sp) / 2
        first_indices = []
        second_indices = []
        average_correlation = 0
        for (site_1, site_2) in nearest_neighbors:
            average_correlation += correlations[site_1, site_2]
        average_correlation /= len(nearest_neighbors)
        """

        # Max chi
        max_chi = max(info['sweep_statistics']['max_chi'])

        # Calculate the (half-chain) entanglement entropy for all NONTRIVIAL BONDS.
        # Above probably means, that we will not calculate entanglement entropy
        # for bonds connecting only one site to the rest of the system.
        entanglement_entropy =  psi.entanglement_entropy()
        average_entanglement_entropy = np.sum(entanglement_entropy) / len(entanglement_entropy)

        # Save results
        results.append([energy, max_chi, average_entanglement_entropy, Sz_total])
        excited_states.append(psi)
        if j >= max_permutations - 1:
            print("# state:", j+1, " / ", n, " ...break")
            break

    # results.sort(key = lambda results: results[0])

    # print("results: ", results)
    return results

def list_to_string(list):
    string = "("
    for entry in list:
        string += str(entry) + " | "
    string = string[:-3]
    string += ")"
    return string

def max_number_of_permutations(list):
    numbers_of_occurrences = Counter(list).values()
    max_number_of_permutations = math.factorial(len(list))
    for number_of_occur in numbers_of_occurrences:
        max_number_of_permutations /= math.factorial(number_of_occur)
    return max_number_of_permutations

def load_vector_from_file(filename, size):
    rec_delim = 4  # This value depends on the Fortran compiler
    with open(filename, "rb") as f:
        f.seek(rec_delim, 1)  # begin record
        arr = np.fromfile(file=f,
                            dtype=np.float64).reshape(size, order='F')
        f.seek(rec_delim, 1)  # end record
    return arr

def load_array_from_file(filename, size):
    rec_delim = 4  # This value depends on the Fortran compiler
    with open(filename, "rb") as f:
        f.seek(rec_delim, 1)  # begin record
        arr = np.fromfile(file=f,
                            dtype=np.float64).reshape((size, size), order='F')
        f.seek(rec_delim, 1)  # end record
    return arr

def energies(vectors_set, hamil):
    energies = []
    for vector in vectors_set:
        vector = np.array([vector])
        # Below line produces a scalar, which is stored as 2D matrix... Thats why we need [0][0] at the end.
        energy = np.dot(vector, np.dot(hamil, vector.T))[0][0]
        energies.append(energy)
    return energies

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- N.")
        sys.exit()

    N = int(sys.argv[1])
    n = 2**N
    J = 1
    h = 0

    spinSite = SpinSite(S=0.5, conserve='Sz')
    # print("spinSite.onsite_ops: ", spinSite.onsite_ops)
    lattice = Chain(L=N, site=spinSite)

    model = HeisenbergModel(lattice=lattice, J=J, h=h)
    results = []
    energies = []
    for Sz in np.arange(-0.5*N, (0.5*N)+1, 1):
        partial_results = run_dmrg(model=model,lattice=lattice, s=Sz, n=n)
        for partial_result in partial_results:
            results.append(partial_result)
            energies.append(partial_result[0])

    results.sort(key = lambda results: results[0])
    energies.sort()

    # print("RESULTS: ", results)
    print("DMRG ENERGIES: ", energies)
    print("RESULTS.SHAPE: ", np.shape(results))

    filename = "DMRG_N=" + str(N) + ".csv"
    with open(filename, 'wb') as f:
        np.savetxt(f, results, delimiter=",")
