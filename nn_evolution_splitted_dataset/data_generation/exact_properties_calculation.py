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
from tenpy.tools.hdf5_io import load
from ..tools.misc import entanglement_entropy
import sys
import copy

from .model import SpinChainInhomogeneousField

def calculate_properties(N, dt, t_max):

    np.set_printoptions(linewidth=np.inf)

    ############################################################################
    # Load data from files
    ############################################################################

    dir_name = "data_SD_N=" + str(N) + "_dt=" + str(dt) + "_t_max=" + str(t_max)

    # Eigenvalues and eigenvectors
    # with open(dir_name + '/eigenvalues_H.npy', 'rb') as f:
    #     eigenvalues_H = np.load(f)
    with open(dir_name + '/eigenvectors_H.npy', 'rb') as f:
        eigenvectors_H = np.load(f)
    # eigenvectors_H_npc = load(dir_name + '/eigenvectors_H_npc.hdf5')

    # Hamiltonians, evolution operator and model representing H (which is used
    # to calculate expected energy using MPSs)
    with open(dir_name + '/H.npy', 'rb') as f:
        H = np.load(f)
    # with open(dir_name + '/HQ.npy', 'rb') as f:
    #     HQ = np.load(f)
    with open(dir_name + '/U.npy', 'rb') as f:
        U = np.load(f)
    model_MPO = load(dir_name + '/model_MPO.hdf5')

    # Load spinSite used for creating MPSs
    spinSite = load(dir_name + '/spinSite.hdf5')

    # Load model_quenched (it is later needed to instantiate the TEBD Engine object)
    model_quenched = load(dir_name + '/model_quenched.hdf5')

    with open(dir_name + '/exact_evol_input.npy', 'rb') as f:
        exact_evol_input = np.load(f)
    with open(dir_name + '/exact_evol_output.npy', 'rb') as f:
        exact_evol_output = np.load(f)

    ############################################################################
    # Calculation of properties of each state-vector
    ############################################################################

    energies = []
    entanglements = []
    eigenvectors_number = np.shape(eigenvectors_H)[0]

    for _t in range(int(t_max/dt)):
        one_timestep_energies = []
        one_timestep_entanglement = []

        for i in range(eigenvectors_number):
            # Prepare vector for further use
            eigenvector = exact_evol_output[i + _t*eigenvectors_number]
            eigenvector = np.array([eigenvector]).T

            # Compute the expected energy
            energy = float(np.dot(np.dot(np.conj(eigenvector).T, H), eigenvector)[0][0])
            # print("ENERGY [EXACT]: ", energy)
            mean_entanglement_entropy = entanglement_entropy(eigenvector, N)
            # print("MEAN ENTANGLEMENT ENTROPY: ", mean_entanglement_entropy)
            one_timestep_energies.append(energy)
            one_timestep_entanglement.append(mean_entanglement_entropy)
        energies.append(one_timestep_energies)
        entanglements.append(one_timestep_entanglement)

    energies = np.stack(energies)
    energies = energies.T
    # print("energies: ", energies)

    entanglements = np.stack(entanglements)
    entanglements = entanglements.T
    # print("entanglements: ", entanglements)

    ############################################################################
    # Save precise values of energies and entanglement entropy
    ############################################################################

    print("Saving precise values of energies...", end="", flush=True)
    np.savetxt(dir_name + "/exact_evol_energies.csv", energies, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving precise values of entanglement entropies...", end="", flush=True)
    np.savetxt(dir_name + "/exact_evol_entanglements.csv", entanglements, delimiter=",", fmt='%f')
    print(" done!")
