# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

from tenpy.networks.site import SpinSite
from tenpy.models.lattice import Chain
from tenpy.models.spins import SpinModel, SpinChain
from tenpy.networks.mps import MPS
from tenpy.algorithms.exact_diag import ExactDiag
from tenpy.tools.hdf5_io import load
import numpy as np
import sys
import os.path
from os import path

# NOTE: dt and t_max are added as dummy arguments, only to make it possible to
# access proper catalog with saved data
def conduct_mps_input_recreation(N, chi_max, dt, t_max):

    np.set_printoptions(linewidth=np.inf)

    ############################################################################
    # Load data from files
    ############################################################################

    dir_name = "data_N=" + str(N) + "_dt=" + str(dt) + "_t_max=" + str(t_max)

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
    # Compression of each eigenvector
    ############################################################################

    var_compression_energies = []
    var_compression_overlaps = []
    var_compression_entanglements = []
    var_compression_sizes = []

    svd_compression_energies = []
    svd_compression_overlaps = []
    svd_compression_entanglements = []
    svd_compression_sizes = []

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

        ########################################################################
        # VARIATIONAL COMPRESSION
        ########################################################################

        psi_compressed_var = psi.copy()

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

        psi_compressed_var.compress(compression_options)
        # print("mps_compressed: ", mps_compressed_var._B)

        total_amount_of_vars = 0
        for j in range(len(psi_compressed_var._B)):
            arr_size = 1
            for subsize in np.shape(psi_compressed_var._B[j]):
                arr_size *= subsize
            total_amount_of_vars += arr_size
        var_compression_sizes.append(total_amount_of_vars)
        # print("[1] total_amount_of_vars: ", total_amount_of_vars)

        energy_compression_var = model_MPO.expectation_value(psi_compressed_var)
        var_compression_energies.append(energy_compression_var)
        # print("[1] energy: ", energy_compression_var)

        entanglements_compression_var = psi_compressed_var.entanglement_entropy()
        mean_entanglement_compression_var = np.mean(entanglements_compression_var)
        var_compression_entanglements.append(mean_entanglement_compression_var)
        # print("[1] mean_entanglement: ", mean_entanglement_compression_var)

        overlap_compression_var = psi.overlap(psi_compressed_var)
        overlap_compression_var_squared = float(np.conj(overlap_compression_var)*overlap_compression_var)
        var_compression_overlaps.append(overlap_compression_var_squared)
        # print("[1] overlap: ", overlap_compression_var)

        ########################################################################
        # SVD COMPRESSION
        ########################################################################

        psi_compressed_svd = psi.copy()

        trunc_params = {
            'chi_max':chi_max,
            'chi_min':1,
            'verbose':0
        }

        psi_compressed_svd.compress_svd(trunc_params)
        # print("mps_compressed: ", mps_compressed_svd._B)

        total_amount_of_vars = 0
        for j in range(len(psi_compressed_svd._B)):
            arr_size = 1
            for subsize in np.shape(psi_compressed_svd._B[j]):
                arr_size *= subsize
            total_amount_of_vars += arr_size
        svd_compression_sizes.append(total_amount_of_vars)
        # print("[2] total_amount_of_vars: ", total_amount_of_vars)

        energy_compression_svd = model_MPO.expectation_value(psi_compressed_svd)
        svd_compression_energies.append(energy_compression_svd)
        # print("[2] energy: ", energy_compression_svd)

        entanglements_compression_svd = psi_compressed_svd.entanglement_entropy()
        mean_entanglement_compression_svd = np.mean(entanglements_compression_svd)
        svd_compression_entanglements.append(mean_entanglement_compression_svd)
        # print("[2] mean_entanglement: ", mean_entanglement_compression_svd)

        overlap_compression_svd = psi.overlap(psi_compressed_svd)
        overlap_compression_svd_squared = float(np.conj(overlap_compression_svd)*overlap_compression_svd)
        svd_compression_overlaps.append(overlap_compression_svd_squared)
        # print("[2] overlap: ", overlap_compression_svd)

        # print("="*20)

    ################################################################################
    #  Create folder to store all results
    ################################################################################

    dir_name += "/tn_input_recreation_chi_max=" + str(chi_max)

    # Create target directory if doesn't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory ", dir_name, " created!")

    ################################################################################
    # Save overlaps, energies, entanglements and sizes for VARIATIONAL COMPRESSION
    ################################################################################

    print("Saving overlaps between tensor networks and exact state-vectors...", end="")
    var_compression_overlaps = np.array([var_compression_overlaps]).T
    np.savetxt(dir_name + "/tn_var_compr_overlaps.csv", var_compression_overlaps, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving energies obtained from tensor networks...", end="")
    var_compression_energies = np.array([var_compression_energies]).T
    np.savetxt(dir_name + "/tn_var_compr_energies.csv", var_compression_energies, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving entanglement entropies obtained from tensor networks...", end="")
    var_compression_entanglements = np.array([var_compression_entanglements]).T
    np.savetxt(dir_name + "/tn_var_compr_entanglements.csv", var_compression_entanglements, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving sizes of tensor networks...", end="")
    var_compression_sizes = np.array([var_compression_sizes]).T
    np.savetxt(dir_name + "/tn_var_compr_sizes.csv", var_compression_sizes, delimiter=",", fmt='%f')
    print(" done!")

    ################################################################################
    # Save overlaps, energies, entanglements and sizes for SVD COMPRESSION
    ################################################################################

    print("Saving overlaps between tensor networks and exact state-vectors...", end="")
    svd_compression_overlaps = np.array([svd_compression_overlaps]).T
    np.savetxt(dir_name + "/tn_svd_compr_overlaps.csv", svd_compression_overlaps, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving energies obtained from tensor networks...", end="")
    svd_compression_energies = np.array([svd_compression_energies]).T
    np.savetxt(dir_name + "/tn_svd_compr_energies.csv", svd_compression_energies, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving entanglement entropies obtained from tensor networks...", end="")
    svd_compression_entanglements = np.array([svd_compression_entanglements]).T
    np.savetxt(dir_name + "/tn_svd_compr_entanglements.csv", svd_compression_entanglements, delimiter=",", fmt='%f')
    print(" done!")

    print("Saving sizes of tensor networks...", end="")
    svd_compression_sizes = np.array([svd_compression_sizes]).T
    np.savetxt(dir_name + "/tn_svd_compr_sizes.csv", svd_compression_sizes, delimiter=",", fmt='%f')
    print(" done!")
