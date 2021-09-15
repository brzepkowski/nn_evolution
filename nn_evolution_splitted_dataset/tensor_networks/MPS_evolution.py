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

import sys
import copy
import os.path
from os import path

from ..data_generation.model import SpinChainInhomogeneousField

def conduct_tebd_time_evolution(N, chi_max, dt, t_max):

    np.set_printoptions(linewidth=np.inf)

    ############################################################################
    # Load data from files
    ############################################################################

    dir_name = "data_SD_N=" + str(N) + "_dt=" + str(dt) + "_t_max=" + str(t_max)

    # Eigenvalues and eigenvectors
    # with open(dir_name + '/eigenvalues_H.npy', 'rb') as f:
    #     eigenvalues_H = np.load(f)
    # with open(dir_name + '/eigenvectors_H.npy', 'rb') as f:
    #     eigenvectors_H = np.load(f)
    eigenvectors_H_npc = load(dir_name + '/eigenvectors_H_npc.hdf5')

    # Hamiltonians, evolution operator and model representing H (which is used
    # to calculate expected energy using MPSs)
    with open(dir_name + '/H.npy', 'rb') as f:
        H = np.load(f)
    # with open(dir_name + '/HQ.npy', 'rb') as f:
    #     HQ = np.load(f)
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
    # Time evolution for each eigenvector
    ############################################################################

    energies = []
    overlaps = []
    entanglements = []
    sizes = []

    eigenvectors_number = np.shape(eigenvectors_H_npc)[0]

    for i in range(eigenvectors_number):

        npc_vector = eigenvectors_H_npc[i]
        npc_vector = npc_vector.split_legs()

        leg_labels = []
        for j in range(N):
            leg_labels.append('p'+str(j))
        npc_vector.iset_leg_labels(leg_labels)
        spinSites = [spinSite] * N
        psi = MPS.from_full(sites=spinSites, psi=npc_vector, form='B', cutoff=1e-16, normalize=True, bc='finite', outer_S=None)

        total_amount_of_vars = 0
        for k in range(len(psi._B)):
            arr_size = 1
            for subsize in np.shape(psi._B[k]):
                arr_size *= subsize
            total_amount_of_vars += arr_size
        # print("[0] total_amount_of_vars: ", total_amount_of_vars)

        entropies_npc = psi.entanglement_entropy()
        mean_entropy_npc = np.mean(entropies_npc)
        # print("[0] mean_entropy: ", mean_entropy_npc)

        energy_npc = float(model_MPO.expectation_value(psi))
        # print("INITIAL ENERGY   [MPS]: ", energy_npc)

        # WARNING: In this script we are doing evolution differently, than in e.g. tools.misc.
        # We are evolving each eigenvector for desired number of timesteps, and then go on
        # to the next eigenvector. Despite that, final resuls are stored in the same
        # fashion, as the ones obtained from tools.misc script.
        #
        # We are switching to this way of conducting evolution, because it is necessary
        # to create TEBD object. If we would like to switch between eigenvectors and
        # evolve each for one timestep, we would have to create the TEBD-Engine object each time,
        # or store the same number of TEBD-ENgine objects as there are eigenvectors of
        # the system, that we are studying.

        ########################################################################
        # Prepare for TEBD evolution with compression
        ########################################################################

        tebd_params_compressed = {
            'N_steps': 1,
            'dt': dt,
            'order': 2,
            'verbose': 0,
            'trunc_params': {
                    'chi_max': chi_max
            }
        }
        eng = tebd.Engine(psi, model_quenched, tebd_params_compressed)

        ########################################################################
        # MAIN LOOP OF EVOLUTION
        ########################################################################

        one_vector_energies = []
        one_vector_overlaps = []
        one_vector_entanglements = []
        one_vector_sizes = []

        for t in range(int(t_max/dt)):

            eng.run()
            energy_compressed_evolution = float(model_MPO.expectation_value(psi))
            one_vector_energies.append(energy_compressed_evolution)

            entanglements_compressed_evolution = psi.entanglement_entropy()
            mean_entanglement_compressed_evolution = np.mean(entanglements_compressed_evolution)
            one_vector_entanglements.append(mean_entanglement_compressed_evolution)

            psi_copy = copy.deepcopy(psi)

            total_amount_of_vars = 0
            for k in range(len(psi_copy._B)):
                arr_size = 1
                for subsize in np.shape(psi_copy._B[k]):
                    arr_size *= subsize
                total_amount_of_vars += arr_size
            one_vector_sizes.append(total_amount_of_vars)

            # Contract all tensors in psi to compute correlations and overlap
            fully_contracted_psi = psi_copy._B[0]
            fully_contracted_psi.ireplace_label('p', 'p0')

            # Contract all tensors included in the MPS
            for j in range(1, len(psi._B), 1):
                tensor = psi_copy._B[j]
                p_leg_label = 'p'+str(j)
                tensor.ireplace_label('p', p_leg_label)
                fully_contracted_psi = npc.tensordot(fully_contracted_psi, tensor, axes=[['vR'], ['vL']])

            # Combine all legs, so that we get a 1D vector
            fully_contracted_psi = fully_contracted_psi.combine_legs(fully_contracted_psi.get_leg_labels())
            fully_contracted_psi = fully_contracted_psi.to_ndarray()

            # Add dummy additional dimension, which will allow for doing a transposition
            fully_contracted_psi = np.array([fully_contracted_psi]).T


            ####################################################################
            # CORRELATIONS
            ####################################################################
            # WARNING: Correlations obtained in this block of code are not saved
            # anywhere. This code is added only to show, how they might be calculated.
            # But calculating these correlations for each vector would take a lot
            # of time, because of which this code just shows how it might be done, if needed.
            """
            N0 = 0
            for N1 in range(N0+1, N):

                # Load spin correlation operator from file
                S_corr_filename = dir_name + '/S_corr_' + str(N0) + '_' + str(N1) + '.npy'
                with open(S_corr_filename, 'rb') as f:
                    S_corr = np.load(f)

                correlation = np.dot(np.dot(np.conjugate(fully_contracted_psi.T), S_corr), fully_contracted_psi)[0][0]
                print("N0: ", N0, ", N1: ", N1, " -> ", correlation)
            """
            ####################################################################

            exact_vector = exact_evol_output[i + t*eigenvectors_number]
            exact_vector = np.array([exact_vector]).T

            overlap = np.dot(np.conj(exact_vector).T, fully_contracted_psi)[0][0]
            overlap_squared = float(np.conj(overlap)*overlap)
            one_vector_overlaps.append(overlap_squared)

        energies.append(one_vector_energies)
        sizes.append(one_vector_sizes)
        overlaps.append(one_vector_overlaps)
        entanglements.append(one_vector_entanglements)

    ################################################################################
    #  Create folder to store all results
    ################################################################################

    dir_name += "/tn_evol_chi_max=" + str(chi_max)

    # Create target directory if doesn't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name, " created!")

    ################################################################################
    # Save overlaps, energies, entanglements and sizes
    ################################################################################

    print("Saving overlaps between tensor networks and exact state-vectors...", end="")
    np.savetxt(dir_name + "/tn_overlaps.csv", overlaps, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving energies obtained from tensor networks...", end="")
    np.savetxt(dir_name + "/tn_energies.csv", energies, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving entanglement entropies obtained from tensor networks...", end="")
    np.savetxt(dir_name + "/tn_entanglements.csv", entanglements, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving sizes of tensor networks...", end="")
    np.savetxt(dir_name + "/tn_sizes.csv", sizes, delimiter=",", fmt='%f')
    print(" done!")
