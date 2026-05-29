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


# Generate a random pure state sampled from the complex Hilbert space C^dim.
def generate_random_pure_state(dim):

    psi = np.random.randn(dim) + 1j * np.random.randn(dim)

    return psi / np.linalg.norm(psi)


# Generate a random diagonal density matrix.
def generate_random_diag_state(dim):

    diag = np.random.rand(dim)

    diag /= diag.sum()

    return np.diag(diag)


# Generate a mixed state as a convex combination of random pure states.
def generate_random_mixed_state(n_pure, dim):

    weights = np.random.dirichlet(np.ones(n_pure))

    rho = np.zeros((dim, dim), dtype=complex)

    for w in weights:

        psi = generate_random_pure_state(dim)

        rho += w * np.outer(psi, psi.conj())

    return rho


# Generate a collection of mixed states.
def generate_mixed_states(num_states, n_pure, dim):

    return [generate_random_mixed_state(n_pure, dim)
            for _ in range(num_states)]


# Generate hybrid states obtained from a convex combination of a random pure state and a random diagonal state.
def generate_convex_combination_list(num_states, dim):

    states = []

    for _ in range(num_states):

        t = random.random()

        psi = generate_random_pure_state(dim)

        rho_pure = np.outer(psi, psi.conj())

        rho_diag = generate_random_diag_state(dim)

        rho = t * rho_diag + (1 - t) * rho_pure

        states.append(rho)

    return states


# Generate a general random density matrix using the Ginibre ensemble.
def generate_random_density_matrix(dim):

    A = (np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim))

    rho = A @ A.conj().T

    rho /= np.trace(rho)

    return rho


# Generate a collection of random density matrices.
def generate_random_density_matrices(num_states, dim):

    return [generate_random_density_matrix(dim) for _ in range(num_states)]


# Generate and save the complete dataset.
def main():

    num_pure_matrices = 4000
    num_mixed_matrices = 6000
    num_convex_combinations = 4000
    num_general_states = 6000

    num_states_for_mixed = 16
    dim = 16

    mixed_rho_list  = generate_mixed_states(num_mixed_matrices, num_states_for_mixed, dim)
    pure_rho_list   = [np.outer(psi := generate_random_pure_state(dim), psi.conj()) for _ in range(num_pure_matrices)]
    hybrid_rho_list = generate_convex_combination_list(num_convex_combinations, dim)
    general_rho_list = generate_random_density_matrices(num_general_states=num_general_states, dim=dim)

  
    density_matrix_list = (mixed_rho_list+ pure_rho_list+ hybrid_rho_list+ general_rho_list)

    random.shuffle(density_matrix_list)

    np.save("random_density_matrices_4qubit.npy", density_matrix_list)

    print(f"Dataset generated successfully: " f"{len(density_matrix_list)} states")


if __name__ == "__main__":
    main()
