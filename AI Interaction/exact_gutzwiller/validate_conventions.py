"""Independent Fock-space validation of the BdG -> BCS pairing-matrix convention."""

from __future__ import annotations

import itertools

import numpy as np

from model import ModelParameters, TwoOrbitalModel, bcs_amplitude, gaussian_pairing_matrix


def create(state: int, mode: int):
    if state & (1 << mode):
        return None, 0
    sign = -1 if ((state & ((1 << mode) - 1)).bit_count() % 2) else 1
    return state | (1 << mode), sign


def annihilate(state: int, mode: int):
    if not (state & (1 << mode)):
        return None, 0
    sign = -1 if ((state & ((1 << mode) - 1)).bit_count() % 2) else 1
    return state & ~(1 << mode), sign


def fock_hamiltonian(h: np.ndarray, delta: np.ndarray) -> np.ndarray:
    n = h.shape[0]
    dimension = 1 << n
    result = np.zeros((dimension, dimension), dtype=np.complex128)

    def add(state: int, column: int, coefficient: complex) -> None:
        if state is not None:
            result[state, column] += coefficient

    for column in range(dimension):
        for i in range(n):
            for j in range(n):
                s1, sign1 = annihilate(column, j)
                if s1 is not None:
                    s2, sign2 = create(s1, i)
                    add(s2, column, h[i, j] * sign1 * sign2)

                s1, sign1 = annihilate(column, j)
                if s1 is not None:
                    s2, sign2 = annihilate(s1, i)
                    add(s2, column, -0.5 * np.conjugate(delta[i, j]) * sign1 * sign2)

                s1, sign1 = create(column, j)
                if s1 is not None:
                    s2, sign2 = create(s1, i)
                    add(s2, column, 0.5 * delta[i, j] * sign1 * sign2)
    return result


def main() -> None:
    model = TwoOrbitalModel(2, 2, ModelParameters(m=1.5))
    h_bdg = model.bdg_hamiltonian(phi_x=0.23, phi_y=-0.17)
    z, gap = gaussian_pairing_matrix(h_bdg)
    h, delta = model.normal_and_pairing_blocks(phi_x=0.23, phi_y=-0.17)
    h_fock = fock_hamiltonian(h, delta)

    even_states = [state for state in range(1 << model.n_modes) if state.bit_count() % 2 == 0]
    h_even = h_fock[np.ix_(even_states, even_states)]
    energies, vectors = np.linalg.eigh(h_even)
    direct = vectors[:, 0]

    gaussian = np.zeros(len(even_states), dtype=np.complex128)
    for index, state in enumerate(even_states):
        occupied = [mode for mode in range(model.n_modes) if state & (1 << mode)]
        gaussian[index] = bcs_amplitude(z, occupied)
    gaussian /= np.linalg.norm(gaussian)

    pivot = int(np.argmax(np.abs(direct)))
    direct *= np.exp(-1j * np.angle(direct[pivot]))
    gaussian *= np.exp(-1j * np.angle(gaussian[pivot]))
    error = np.linalg.norm(direct - gaussian)

    print(f"Fock-space even-sector ground energy: {energies[0]:.12f}")
    print(f"Fock-space even-sector gap: {energies[1] - energies[0]:.12f}")
    print(f"BdG single-particle gap: {gap:.12f}")
    print(f"Gaussian/Fock ground-state vector error: {error:.3e}")
    print(f"Gaussian norm before normalization check: {np.linalg.norm(gaussian):.12f}")
    if error > 1e-9:
        raise SystemExit("FAILED: BCS amplitudes do not match the Fock-space ground state")
    print("PASSED: BdG, pairing-matrix, and Pfaffian conventions agree with exact Fock space")


if __name__ == "__main__":
    main()
