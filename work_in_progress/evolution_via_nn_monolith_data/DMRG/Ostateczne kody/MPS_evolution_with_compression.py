# Copyright Bartosz Rzepkowski, Wrocław 2020

from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Chain
from tenpy.models.spins import SpinModel, SpinChain
from tenpy.models.spins_nnn import SpinChainNNN
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms import tebd
import numpy as np
import sys, time

def HeisenbergModel(lattice, J, hz=0, bcy="ladder", bcx="open"):
    Jx = Jy = Jz = J
    hx = hy = 0
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

def print_mps(psi):
    for i in range(len(psi._B)):
        print(psi._B[i])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- N,")
        print("- chi_max,")
        print("- t_max.")
        sys.exit()

    N = int(sys.argv[1])
    chi_max = int(sys.argv[2])
    t_max = float(sys.argv[3])
    J = 1
    hz = 2
    dt = 0.1

    spinSite = SpinSite(S=0.5, conserve=None)
    lattice = Chain(L=N, site=spinSite)

    model = HeisenbergModel(lattice=lattice, J=J, hz=hz)
    model_MPO = model.calc_H_MPO()

    exact_diag = ExactDiag(model)
    exact_diag.build_full_H_from_mpo()
    exact_diag.full_diagonalization()
    print("eigenvalues: ", exact_diag.E)
    print("exact_diag.V: ", exact_diag.V)

    # model_quenched = HeisenbergModel(lattice=lattice, J=J, hz=0)
    # model_quenched_params = dict(lattice=lattice, Jx=1, Jy=1, Jz=1, Jxp=1, Jyp=1, Jzp=1, hx=0, hy=0, hz=0, bc_MPS='finite', conserve=None)
    # model_quenched = SpinChainNNN(model_quenched_params)
    # model_quenched_MPO = model_quenched.calc_H_MPO()

    # exact_diag_quenched = ExactDiag(model_quenched)
    # exact_diag_quenched.build_full_H_from_mpo()
    # exact_diag_quenched.full_diagonalization()
    # print("eigenvalues quenched: ", exact_diag_quenched.E)
    # print("exact_diag.V quenched: ", exact_diag_quenched.V)
    # sys.exit()

    eigenvectors = exact_diag.V
    eigenvectors.itranspose()

    for i in range(np.shape(eigenvectors)[0]):
        eigenvector = eigenvectors[i]
        eigenvector = eigenvector.split_legs()

        leg_labels = []
        for i in range(N):
            leg_labels.append('p'+str(i))
        eigenvector.iset_leg_labels(leg_labels)
        spinSites = [spinSite] * N
        mps = MPS.from_full(sites=spinSites, psi=eigenvector, form='B', cutoff=1e-16, normalize=True, bc='finite', outer_S=None)

        # print("mps: ", mps._B)
        # for j in range(len(mps._B)):
        #     print(mps._B[j])

        total_amount_of_vars = 0
        for j in range(len(mps._B)):
            arr_size = 1
            for subsize in np.shape(mps._B[j]):
                arr_size *= subsize
            total_amount_of_vars += arr_size
        # print("[0] total_amount_of_vars: ", total_amount_of_vars)

        energy = model_MPO.expectation_value(mps)
        # energy = model_quenched_MPO.expectation_value(mps)
        print("[0] energy: ", energy)

        entropies = mps.entanglement_entropy()
        mean_entropy = np.mean(entropies)
        # print("[0] mean_entropy: ", mean_entropy)

        ########################################################################
        # EXACT TIME EVOLUTION USING TEBD
        ########################################################################

        mps_exact_evolution = mps.copy()

        tebd_params = {
            'N_steps': 1,
            'dt': dt,
            'order': 2,
            'verbose': 0,
            'trunc_params': {'chi_max': 2**N, 'svd_min': 1.e-12}
        }

        # eng = tebd.Engine(mps_exact_evolution, model_quenched, tebd_params)
        eng = tebd.RandomUnitaryEvolution(mps_exact_evolution, tebd_params)

        # print("[INITIAL MPS]: ")
        # print_mps(mps)

        for n in range(int(t_max / dt + 0.5)):
            eng.run()
            energy_exact_evolution = model_MPO.expectation_value(mps_exact_evolution)
            # energy_exact_evolution = model_quenched_MPO.expectation_value(mps_exact_evolution)
            print("[1] energy: ", energy_exact_evolution, end="\r")
            # entanglement_entropy = mps_exact_evolution.entanglement_entropy()
            # print("  || entanglement: ", entanglement_entropy, end="\r")
            # S.append(psi.entanglement_entropy())

            # print("[EVOLVED MPS]: ")
            # print_mps(mps_exact_evolution)
            overlap_with_init = mps_exact_evolution.overlap(mps)
            # print("[1] overlap_with_init: ", overlap_with_init, end="\r")
            prob_of_being_in_init_state = overlap_with_init * np.conj(overlap_with_init)
            # print("[1] prob_of_being_in_init_state: ", prob_of_being_in_init_state, end="\r")
        # eng.run()
        # time.sleep(2)
        print()

        # print("mps_evolved: ", mps_exact_evolution._B)

        total_amount_of_vars = 0
        for j in range(len(mps_exact_evolution._B)):
            arr_size = 1
            for subsize in np.shape(mps_exact_evolution._B[j]):
                arr_size *= subsize
            total_amount_of_vars += arr_size

        # print("[1] total_amount_of_vars: ", total_amount_of_vars)

        # energy_exact_evolution = model_quenched_MPO.expectation_value(mps_exact_evolution)
        energy_exact_evolution = model_MPO.expectation_value(mps_exact_evolution)
        print("[1] energy: ", energy_exact_evolution)

        entropies_exact_evolution = mps_exact_evolution.entanglement_entropy()
        mean_entropy_exact_evolution = np.mean(entropies_exact_evolution)
        # print("[1] mean_entropy: ", mean_entropy_exact_evolution)

        overlap_with_init = mps.overlap(mps_exact_evolution)
        # print("[1] overlap_with_init: ", overlap_with_init)
        prob_of_being_in_init_state = overlap_with_init * np.conj(overlap_with_init)
        print("[1] prob_of_being_in_init_state: ", prob_of_being_in_init_state)
        ########################################################################
        # TIME EVOLUTION USING TEBD WITH COMPRESSION
        ########################################################################
        """
        mps_compressed_evolution = mps.copy()

        tebd_params_compressed = {
            'N_steps': 1,
            'dt': dt,
            'order': 2,
            'verbose': 0,
            'trunc_params': {
                    'chi_max': chi_max
            }
        }
        eng = tebd.Engine(mps_compressed_evolution, model_quenched, tebd_params_compressed)

        for n in range(int(t_max / dt + 0.5)):
            eng.run()
            energy_compressed_evolution = model_quenched_MPO.expectation_value(mps_compressed_evolution)
            print("[2] energy: ", energy_compressed_evolution, end="\r")

        # print("mps_evolved: ", mps_compressed_evolution._B)

        total_amount_of_vars = 0
        for j in range(len(mps_compressed_evolution._B)):
            arr_size = 1
            for subsize in np.shape(mps_compressed_evolution._B[j]):
                arr_size *= subsize
            total_amount_of_vars += arr_size

        # print("[2] total_amount_of_vars: ", total_amount_of_vars)

        energy_compressed_evolution = model_quenched_MPO.expectation_value(mps_compressed_evolution)
        print("[2] energy: ", energy_compressed_evolution)

        entropies_compressed_evolution = mps_compressed_evolution.entanglement_entropy()
        mean_entropy_compressed_evolution = np.mean(entropies_compressed_evolution)
        # print("[2] mean_entropy: ", mean_entropy_compressed_evolution)

        overlap = mps_exact_evolution.overlap(mps_compressed_evolution)
        # print("[2] overlap: ", overlap)
        """
        print("="*20)



    # results = []
    # energies = []
    # for Sz in np.arange(-0.5*N, (0.5*N)+1, 1):
    #     partial_results = run_dmrg(model=model,lattice=lattice, s=Sz, n=n)
    #     for partial_result in partial_results:
    #         results.append(partial_result)
    #         energies.append(partial_result[0])
    #
    # results.sort(key = lambda results: results[0])
    # energies.sort()
    #
    # # print("RESULTS: ", results)
    # print("DMRG ENERGIES: ", energies)
    # print("RESULTS.SHAPE: ", np.shape(results))
    #
    # filename = "DMRG_N=" + str(N) + ".csv"
    # with open(filename, 'wb') as f:
    #     np.savetxt(f, results, delimiter=",")
