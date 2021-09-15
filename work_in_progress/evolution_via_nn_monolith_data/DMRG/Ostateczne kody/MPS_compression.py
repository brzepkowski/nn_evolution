# Copyright Bartosz Rzepkowski, Wrocław 2020

from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Chain
from tenpy.models.spins import SpinModel, SpinChain
from tenpy.networks.mps import MPS
from tenpy.algorithms.exact_diag import ExactDiag
import numpy as np
import sys

def HeisenbergModel(lattice, J, h=0, bcy="ladder", bcx="open"):
    Jx = Jy = Jz = J
    hx = hy = 0
    hz = h
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

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect number of arguments. Provide them in the following form:")
        print("- N,")
        print("- chi_max.")
        sys.exit()

    N = int(sys.argv[1])
    chi_max = int(sys.argv[2])
    J = 1
    h = 2

    spinSite = SpinSite(S=0.5, conserve=None)
    lattice = Chain(L=N, site=spinSite)

    model = HeisenbergModel(lattice=lattice, J=J, h=h)

    exact_diag = ExactDiag(model)
    exact_diag.build_full_H_from_mpo()
    exact_diag.full_diagonalization()
    print("eigenvalues: ", exact_diag.E)
    # print("exact_diag.V: ", exact_diag.V)

    model_MPO = model.calc_H_MPO()

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
        print("[0] total_amount_of_vars: ", total_amount_of_vars)

        energy = model_MPO.expectation_value(mps)
        print("[0] energy: ", energy)

        entropies = mps.entanglement_entropy()
        mean_entropy = np.mean(entropies)
        print("[0] mean_entropy: ", mean_entropy)

        ########################################################################
        # VARIATIONAL COMPRESSION
        ########################################################################

        mps_compressed_var = mps.copy()

        compression_options = {
                        'combine':True,
                        'compression_method':'variational',
                        'trunc_params':{
                            'chi_max':chi_max,
                            'chi_min':1,
                            'verbose':0
                        },
                        'verbose':0
        }

        mps_compressed_var.compress(compression_options)
        # print("mps_compressed: ", mps_compressed_var._B)

        total_amount_of_vars = 0
        for j in range(len(mps_compressed_var._B)):
            arr_size = 1
            for subsize in np.shape(mps_compressed_var._B[j]):
                arr_size *= subsize
            total_amount_of_vars += arr_size
        print("[1] total_amount_of_vars: ", total_amount_of_vars)

        energy_compression_var = model_MPO.expectation_value(mps_compressed_var)
        print("[1] energy: ", energy_compression_var)

        entropies_compression_var = mps_compressed_var.entanglement_entropy()
        mean_entropy_compression_var = np.mean(entropies_compression_var)
        print("[1] mean_entropy: ", mean_entropy_compression_var)

        overlap_compression_var = mps.overlap(mps_compressed_var)
        print("[1] overlap: ", overlap_compression_var)

        ########################################################################
        # SVD COMPRESSION
        ########################################################################

        mps_compressed_svd = mps.copy()

        trunc_params = {
            'chi_max':chi_max,
            'chi_min':1,
            'verbose':0
        }

        mps_compressed_svd.compress_svd(trunc_params)
        # print("mps_compressed: ", mps_compressed_svd._B)

        total_amount_of_vars = 0
        for j in range(len(mps_compressed_svd._B)):
            arr_size = 1
            for subsize in np.shape(mps_compressed_svd._B[j]):
                arr_size *= subsize
            total_amount_of_vars += arr_size
        print("[2] total_amount_of_vars: ", total_amount_of_vars)

        energy_compression_svd = model_MPO.expectation_value(mps_compressed_svd)
        print("[2] energy: ", energy_compression_svd)

        entropies_compression_svd = mps_compressed_svd.entanglement_entropy()
        mean_entropy_compression_svd = np.mean(entropies_compression_svd)
        print("[2] mean_entropy: ", mean_entropy_compression_svd)

        overlap_compression_svd = mps.overlap(mps_compressed_svd)
        print("[2] overlap: ", overlap_compression_svd)

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
