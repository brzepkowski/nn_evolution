# Copyright Bartosz Rzepkowski, Wrocław 2020-2021

import numpy as np
from ..tools.misc import evolve_whole_set_many_evolution_steps, generate_linearly_combined_vector
from ..tools.misc import reshape_set, flatten_set
from math import floor
import sys
import copy

def conduct_exact_evol_and_gen_sets(N, dt, t_max, set_number, gen_test_and_nn_input_sets=False):
    COMBINED_SETS_MULTIPLIER = 20

    # np.set_printoptions(linewidth=np.inf)

    ############################################################################
    # Load data from files
    ############################################################################

    dir_name = "data_SD_N=" + str(N) + "_dt=" + str(dt) + "_t_max=" + str(t_max)

    # Eigenvalues and eigenvectors
    with open(dir_name + '/eigenvalues_H.npy', 'rb') as f:
        eigenvalues_H = np.load(f)
    with open(dir_name + '/eigenvectors_H.npy', 'rb') as f:
        eigenvectors_H = np.load(f)
    with open(dir_name + '/U.npy', 'rb') as f:
        U = np.load(f)

    ############################################################################
    # Exact time evolution of each eigenvector + TEST SET generation
    ############################################################################

    if gen_test_and_nn_input_sets:
        # Below method takes each evolves each eigenvector for on timestep dt at a time.
        # So as a result we will have evolved whole set of eigenvectors, but to check
        # the state of particular eigenvecter in consecutive timesteps, we would have to
        # take every i + t*len(eigenvectors_H) vector
        print("Evolving testing set...", end="", flush=True)
        test_input_vectors, test_output_vectors = evolve_whole_set_many_evolution_steps(eigenvectors_H, U, dt, t_max)
        print(" done!")

        eigenvectors_number = np.shape(eigenvectors_H)[0]

        with open(dir_name + '/exact_evol_input.npy', 'wb') as f:
            np.save(f, test_input_vectors)
        with open(dir_name + '/exact_evol_output.npy', 'wb') as f:
            np.save(f, test_output_vectors)

        print("Reshaping testing_input set... ", end="", flush=True)
        reshaped_test_input_vectors = reshape_set(test_input_vectors)
        print(" done!")
        del test_input_vectors
        print("Flattening testing_input set... ", end="", flush=True)
        reshaped_test_input_vectors = flatten_set(reshaped_test_input_vectors)
        print(" done!")
        print("Saving testing_input set... ", end="", flush=True)
        with open(dir_name + '/test_input.npy', 'wb') as f:
            np.save(f, reshaped_test_input_vectors)
        print(" done!")
        del reshaped_test_input_vectors

        print("Reshaping testing_output set... ", end="", flush=True)
        reshaped_test_output_vectors = reshape_set(test_output_vectors)
        print(" done!")
        del test_output_vectors
        print("Flattening testing_output set... ", end="", flush=True)
        reshaped_test_output_vectors = flatten_set(reshaped_test_output_vectors)
        print(" done!")
        print("Saving testing_output set... ", end="", flush=True)
        with open(dir_name + '/test_output.npy', 'wb') as f:
            np.save(f, reshaped_test_output_vectors)
        print(" done!")
        del reshaped_test_output_vectors

        print("Saving testing eigenvalues... ", end="", flush=True)
        with open(dir_name + '/test_eigenvalues.npy', 'wb') as f: # Save remaining eigenvalues, because these correspond to vectors in the testing set
            np.save(f, eigenvalues_H)
        print(" done!")
        del eigenvalues_H

        print("#"*40)

    ############################################################################
    # Prepare basis for generation of TRAINING and VALIDATION SETS
    ############################################################################

    # We set the fixed size of training + validation sets to be
    # COMBINED_SETS_MULTIPLIER times larger than the number of eigenvectors in H
    combined_sets_size = COMBINED_SETS_MULTIPLIER*np.shape(eigenvectors_H)[0]
    validation_set_size = floor(combined_sets_size/4)
    training_set_size = combined_sets_size - validation_set_size

    training_set = []
    validation_set = []

    ############################################################################
    # Evolve, save and delete TRAINING set
    ############################################################################

    print("Generating training set...", end="", flush=True)
    for i in range(training_set_size):
        training_set.append(generate_linearly_combined_vector(eigenvectors_H))
    print(" done!")

    print("Evolving training set...", end="", flush=True)
    (training_input_vectors, training_output_vectors) = evolve_whole_set_many_evolution_steps(training_set, U, dt, t_max)
    print(" done!")

    del training_set

    print("Reshaping training_input set... ", end="", flush=True)
    reshaped_training_input_vectors = reshape_set(training_input_vectors)
    print(" done!")
    del training_input_vectors
    print("Flattening training_input set... ", end="", flush=True)
    reshaped_training_input_vectors = flatten_set(reshaped_training_input_vectors)
    print(" done!")
    print("Saving training_input set... ", end="", flush=True)
    training_input_filename = (dir_name + '/training_input_{set_number:d}.npy')
    training_input_filename = training_input_filename.format(
        set_number=set_number,
    )
    with open(training_input_filename, 'wb') as f:
        np.save(f, reshaped_training_input_vectors)
    print(" done!")
    del reshaped_training_input_vectors


    print("Reshaping training_output set... ", end="", flush=True)
    reshaped_training_output_vectors = reshape_set(training_output_vectors)
    print(" done!")
    del training_output_vectors
    print("Flattening training_output set... ", end="", flush=True)
    reshaped_training_output_vectors = flatten_set(reshaped_training_output_vectors)
    print(" done!")
    print("Saving training_output set... ", end="", flush=True)
    training_output_filename = (dir_name + '/training_output_{set_number:d}.npy')
    training_output_filename = training_output_filename.format(
        set_number=set_number,
    )
    with open(training_output_filename, 'wb') as f:
        np.save(f, reshaped_training_output_vectors)
    print(" done!")

    del reshaped_training_output_vectors

    print("#"*40)

    ############################################################################
    # Evolve, save and delete VALIDATION set
    ############################################################################

    print("Generating validation set...", end="", flush=True)
    for i in range(validation_set_size):
        validation_set.append(generate_linearly_combined_vector(eigenvectors_H))
    print(" done!")

    print("Evolving validation set...", end="", flush=True)
    (validation_input_vectors, validation_output_vectors) = evolve_whole_set_many_evolution_steps(validation_set, U, dt, t_max)
    print(" done!")

    del validation_set

    print("Reshaping validation_input set... ", end="", flush=True)
    reshaped_validation_input_vectors = reshape_set(validation_input_vectors)
    print(" done!")
    del validation_input_vectors
    print("Flattening validation_input set... ", end="", flush=True)
    reshaped_validation_input_vectors = flatten_set(reshaped_validation_input_vectors)
    print(" done!")
    print("Saving validation_input set... ", end="", flush=True)
    validation_input_filename = (dir_name + '/validation_input_{set_number:d}.npy')
    validation_input_filename = validation_input_filename.format(
        set_number=set_number,
    )
    with open(validation_input_filename, 'wb') as f:
        np.save(f, reshaped_validation_input_vectors)
    print(" done!")
    del reshaped_validation_input_vectors


    print("Reshaping validation_output set... ", end="", flush=True)
    reshaped_validation_output_vectors = reshape_set(validation_output_vectors)
    print(" done!")
    del validation_output_vectors
    print("Flattening validation_output set... ", end="", flush=True)
    reshaped_validation_output_vectors = flatten_set(reshaped_validation_output_vectors)
    print(" done!")
    print("Saving validation_output set... ", end="", flush=True)
    validation_output_filename = (dir_name + '/validation_output_{set_number:d}.npy')
    validation_output_filename = validation_output_filename.format(
        set_number=set_number,
    )
    with open(validation_output_filename, 'wb') as f:
        np.save(f, reshaped_validation_output_vectors)
    print(" done!")
    del reshaped_validation_output_vectors

    print("#"*40)

    ############################################################################
    # Reshape and save INPUT FOR THE NN, so that it (NN) can make predictions,
    # which we will be comparing with exact evolution
    ############################################################################

    if gen_test_and_nn_input_sets:
        print("Reshaping and flattening input for NN... ", end="", flush=True)
        # Input for NN consists of the eigenvectors of H
        nn_input_for_predictions = reshape_set(eigenvectors_H, remove_input_matrix=False)
        nn_input_for_predictions = flatten_set(nn_input_for_predictions)
        print(" done!")

        print("Saving input for NN... ", end="", flush=True)
        with open(dir_name + '/nn_input_for_predictions.npy', 'wb') as f:
            np.save(f, nn_input_for_predictions)
        print(" done!")

        del nn_input_for_predictions

        print("#"*40)
