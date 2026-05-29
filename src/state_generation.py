"""
state_generation.py

Generation of random quantum states used in the paper

"Machine-Learning Estimation of Quantum Fisher Information from Collective Observables".

This script generates a dataset composed of

1. Random pure states
2. Random mixed states (convex combinations of pure states)
3. Hybrid states (pure-diagonal mixtures)
4. General random density matrices

Author:
Luis D. Zambrano Palma and Yusef Maleki

Repository:
https://github.com/Ldzambra/QFI-SVR
"""

import numpy as np
import random


# Generate a random normalized pure quantum state.
def sample_pure_state(dim):

    state = np.random.randn(dim) + 1j*np.random.randn(dim)

    return state / np.linalg.norm(state)


# Generate a diagonal density operator from a random probability distribution.
def sample_population_state(dim):

    populations = np.random.rand(dim)

    populations /= populations.sum()

    return np.diag(populations)


# Build a mixed quantum state from an ensemble of pure states.
def build_density_operator(num_components, dim):

    probabilities = np.random.dirichlet(np.ones(num_components))

    density_operator = np.zeros((dim, dim), dtype=complex)

    for p in probabilities:

        state = sample_pure_state(dim)

        density_operator += p*np.outer(state, state.conj())

    return density_operator


# Create an ensemble of mixed density operators.
def create_mixed_ensemble(num_samples, num_components, dim):

    return [build_density_operator(num_components, dim) for _ in range(num_samples)]


# Create states interpolating between coherent and population states.
def create_hybrid_ensemble(num_samples, dim):

    hybrid_ensemble = []

    for _ in range(num_samples):

        mixing_parameter = random.random()

        state = sample_pure_state(dim)

        coherent_state = np.outer(state, state.conj())

        population_state = sample_population_state(dim)

        density_operator = (mixing_parameter*population_state + (1-mixing_parameter)*coherent_state)

        hybrid_ensemble.append(density_operator)

    return hybrid_ensemble


# Generate a density operator from the Ginibre construction.
def sample_ginibre_state(dim):

    G = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)

    density_operator = G @ G.conj().T

    density_operator /= np.trace(density_operator)

    return density_operator


# Create an ensemble of Ginibre density operators.
def create_ginibre_ensemble(num_samples, dim):

    return [sample_ginibre_state(dim) for _ in range(num_samples)]


# Assemble and save the complete quantum-state dataset.
def main():

    num_pure_states = 4000
    num_mixed_states = 6000
    num_hybrid_states = 4000
    num_ginibre_states = 6000

    nq = 4

    dim = 2**nq
    
    # Generate all state families.
    mixed_state_ensemble = create_mixed_ensemble(num_mixed_states, dim, dim)
    pure_state_ensemble = [np.outer(state := sample_pure_state(dim), state.conj()) for _ in range(num_pure_states)]
    hybrid_state_ensemble = create_hybrid_ensemble(num_hybrid_states, dim)
    ginibre_state_ensemble = create_ginibre_ensemble(num_ginibre_states, dim)
    
    # Assemble all state families into a single training dataset.
    quantum_state_dataset = (pure_state_ensemble + mixed_state_ensemble + hybrid_state_ensemble + ginibre_state_ensemble)

    # Randomize the ordering of the generated quantum states.
    random.shuffle(quantum_state_dataset)

    #Save them
    np.save(f"qfi_dataset_{nq}qubits.npy", quantum_state_dataset)

    print(f"Dataset generated successfully: " f"{len(quantum_state_dataset)} states")


if __name__ == "__main__":
    main()
