# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

import numpy as np
from numpy.linalg import eig
from scipy.linalg import expm
import tenpy.linalg.np_conserved as npc
from tenpy.models.model import NearestNeighborModel, CouplingMPOModel
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinSite
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.algorithms import tebd
from tenpy.tools.params import asConfig
from tenpy.tools.hdf5_io import save, load
from ..tools.misc import generate_correlation_operator, entanglement_entropy
import sys, os
import copy

from .model import SpinChainInhomogeneousField

# dt and t_max arguments are passed only to make it possible to save all data purposed
# for the same simulation in the same catalog
def gen_and_diagonalize_hamil_before_quench(N, J, hz, dt, t_max):

    np.set_printoptions(linewidth=np.inf)

    # Create target directory if it doesn't exist
    dir_name = "data_SD_N=" + str(N) + "_dt=" + str(dt) + "_t_max=" + str(t_max)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name,  " created")

    ############################################################################
    # Instantiation od 1D lattice
    ############################################################################

    spinSite = SpinSite(S=0.5, conserve=None)
    save(spinSite, dir_name + '/spinSite.hdf5')

    lattice = Chain(L=N, site=spinSite)
    save(lattice, dir_name + '/lattice.hdf5')

    ############################################################################
    # Hamiltonian H
    ############################################################################

    print("#"*100)
    print("###### H ######")
    print("#"*100)

    model_params = {
        "L": N,
        "lattice": lattice,
        "bc_MPS": "finite",
        # "bc_y": bcy,
        # "bc_x": bcx,
        # "Lx": Lx,  # defines cylinder circumference
        # "Ly": Ly,  # defines cylinder circumference
        "Jx": 1,
        "Jy": 1,
        "Jz": 1,
        "hx": 0,
        "hy": 0,
        "hz": hz,
        "muJ": 0,
        "D": 0,
        "E": 0,
        "conserve": "best"   # Heisenberg coupling
    }

    # Define separate models for rows and columns
    model = SpinChainInhomogeneousField(model_params)
    model_MPO = model.calc_H_MPO()

    save(model_MPO, dir_name + '/model_MPO.hdf5')

    exact_diag = ExactDiag(model)
    exact_diag.build_full_H_from_mpo()
    exact_diag.full_diagonalization()

    H = exact_diag.full_H.to_ndarray()

    with open(dir_name + '/H.npy', 'wb') as f:
        np.save(f, H)

    # print(H)
    # print("eigenvalues: ", exact_diag.E)
    # print("exact_diag.V: ", exact_diag.V)
    eigenvalues_H = exact_diag.E
    # print("eigenvalues: ", eigenvalues_H)

    eigenvectors_H = exact_diag.V.to_ndarray()
    eigenvectors_H = eigenvectors_H.T

    # Calculate initial values of entanglement entropy
    entanglements = []
    for eigenvector in eigenvectors_H:
        eigenvector = np.array([eigenvector]).T
        mean_entanglement_entropy = entanglement_entropy(eigenvector, N)
        entanglements.append(mean_entanglement_entropy)

    eigenvectors_H_npc = exact_diag.V
    eigenvectors_H_npc.itranspose()

    with open(dir_name + '/eigenvalues_H.npy', 'wb') as f:
        np.save(f, eigenvalues_H)
    with open(dir_name + '/eigenvectors_H.npy', 'wb') as f:
        np.save(f, eigenvectors_H)
    save(eigenvectors_H_npc, dir_name + '/eigenvectors_H_npc.hdf5')

    eigenvalues_H = np.array([eigenvalues_H]).T
    np.savetxt(dir_name + "/exact_init_energies.csv", eigenvalues_H, delimiter=",", fmt='%f')
    entanglements = np.array([entanglements]).T
    np.savetxt(dir_name + "/exact_init_entanglements.csv", entanglements, delimiter=",", fmt='%f')


def gen_hamil_after_quench_and_evol_operator(N, J, dt, t_max):

    dir_name = "data_SD_N=" + str(N) + "_dt=" + str(dt) + "_t_max=" + str(t_max)

    lattice = load(dir_name + '/lattice.hdf5')

    ############################################################################
    # Hamiltonian HQ
    ############################################################################

    print("#"*100)
    print("###### HQ ######")
    print("#"*100)

    model_quenched_params = {
        "L": N,
        "lattice": lattice,
        "bc_MPS": "finite",
        # "bc_y": bcy,
        # "bc_x": bcx,
        # "Lx": Lx,  # defines cylinder circumference
        # "Ly": Ly,  # defines cylinder circumference
        "Jx": 1,
        "Jy": 1,
        "Jz": 1,
        "hx": 0,
        "hy": 0,
        "hz": 0,
        "muJ": 0,
        "D": 0,
        "E": 0,
        "conserve": "best"   # Heisenberg coupling
    }

    # Define separate models for rows and columns
    model_quenched = SpinChainInhomogeneousField(model_quenched_params)

    exact_diag_quenched = ExactDiag(model_quenched)
    exact_diag_quenched.build_full_H_from_mpo()
    # exact_diag_quenched.full_diagonalization() # We don't need to diagonalize the quenched Hamiltonian

    HQ = exact_diag_quenched.full_H.to_ndarray()

    with open(dir_name + '/HQ.npy', 'wb') as f:
        np.save(f, HQ)
    # Save model_quenched (it is later needed to instantiate the TEBD Engine object)
    save(model_quenched, dir_name + '/model_quenched.hdf5')

    # print(HQ)
    # print("eigenvalues: ", exact_diag_quenched.E)
    # print("exact_diag.V: ", exact_diag_quenched.V)

    # commutator = np.dot(H, HQ) - np.dot(HQ, H)
    # print("commutator:")
    # print(commutator)

    ############################################################################
    # Evolution
    ############################################################################

    print("#"*100)
    print("##### EVOLUTION #####")
    print("#"*100)

    U = expm(-(1j)*dt*HQ)

    # print("U")
    # print(U)

    with open(dir_name + '/U.npy', 'wb') as f:
        np.save(f, U)

# dt and t_max arguments are passed only to make it possible to save correlation
# operators in the same catalog as all other data generated previously
def gen_correlation_operators(N, dt, t_max):

    dir_name = "data_SD_N=" + str(N) + "_dt=" + str(dt) + "_t_max=" + str(t_max)

    print("Generating correlation operators... ", flush=True)
    # WARNING: To generate all possible correlation operators just uncomment below line
    # and delete the "N0 = 0" one. Also put the "for N1 in range(N0+1, N):" inside the uncommented one
    # for N0 in range(N-1):
    N0 = 0
    for N1 in range(N0+1, N):
        print(N1+1, " / ", N, end="\r")
        filename = 'S_corr_' + str(N0) + '_' + str(N1)
        S_corr = generate_correlation_operator(N0, N1, N)

        with open(dir_name + "/" + filename + '.npy', 'wb') as f:
            np.save(f, S_corr)
        del S_corr
    print(N1+1, " / ", N)
    print("Done!")
