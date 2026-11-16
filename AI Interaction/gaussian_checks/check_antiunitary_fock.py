"""Check the reflection-conjugation action in an explicit 2x2 Fock space."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import PARAMETERS
from gaussian_overlap import normalized_gaussian_overlap
from model import TwoOrbitalModel, gaussian_bcs_state
from validate_gaussian_overlap import fock_vector


def create(mask, mode):
    bit = 1 << mode
    if mask & bit:
        return None, 0.0
    sign = -1.0 if ((mask & (bit - 1)).bit_count() % 2) else 1.0
    return mask | bit, sign


def annihilate(mask, mode):
    bit = 1 << mode
    if not mask & bit:
        return None, 0.0
    sign = -1.0 if ((mask & (bit - 1)).bit_count() % 2) else 1.0
    return mask ^ bit, sign


def physical_hamiltonian(h_bdg):
    n = h_bdg.shape[0] // 2
    h = h_bdg[:n, :n]
    delta = h_bdg[:n, n:]
    result = np.zeros((1 << n, 1 << n), dtype=complex)
    for mask in range(1 << n):
        for i in range(n):
            for j in range(n):
                if abs(h[i, j]) > 1.0e-14:
                    mask_j, sign_j = annihilate(mask, j)
                    if mask_j is not None:
                        mask_i, sign_i = create(mask_j, i)
                        if mask_i is not None:
                            result[mask_i, mask] += h[i, j] * sign_i * sign_j
        for i in range(n):
            for j in range(n):
                if abs(delta[i, j]) > 1.0e-14:
                    mask_j, sign_j = create(mask, j)
                    if mask_j is not None:
                        mask_i, sign_i = create(mask_j, i)
                        if mask_i is not None:
                            result[mask_i, mask] += 0.5 * delta[i, j] * sign_i * sign_j
                    mask_j, sign_j = annihilate(mask, j)
                    if mask_j is not None:
                        mask_i, sign_i = annihilate(mask_j, i)
                        if mask_i is not None:
                            result[mask_i, mask] += -0.5 * delta[i, j].conj() * sign_i * sign_j
    return result


def reflected_vector(vector, model):
    n = model.n_modes
    result = np.zeros_like(vector)
    for mask in range(1 << n):
        occupied = [mode for mode in range(n) if mask & (1 << mode)]
        mapped = []
        for mode in occupied:
            cell, orbital = divmod(mode, 2)
            x, y = cell % model.lx, cell // model.lx
            mapped.append(model.mode((1 - x) % model.lx, y, orbital))
        mapped.sort()
        new_mask = sum(1 << mode for mode in mapped)
        sign = 1.0
        for source_position, mode in enumerate(mapped):
            target_position = occupied.index(mode)
            if (source_position - target_position) % 2:
                sign = -sign
        # The parity of a permutation is more reliably obtained by inversions.
        inversions = sum(
            mapped[i] > mapped[j]
            for i in range(len(mapped)) for j in range(i + 1, len(mapped))
        )
        # mapped was sorted, so calculate the sign before sorting instead.
        del inversions
        result[new_mask] = vector[mask].conj()
    return result


def reflected_vector_correct(vector, model):
    n = model.n_modes
    result = np.zeros_like(vector)
    for mask in range(1 << n):
        occupied = [mode for mode in range(n) if mask & (1 << mode)]
        mapped = []
        for mode in occupied:
            cell, orbital = divmod(mode, 2)
            x, y = cell % model.lx, cell // model.lx
            mapped.append(model.mode((1 - x) % model.lx, y, orbital))
        inversions = sum(
            mapped[i] > mapped[j]
            for i in range(len(mapped)) for j in range(i + 1, len(mapped))
        )
        new_mask = sum(1 << mode for mode in mapped)
        result[new_mask] = (-1.0 if inversions % 2 else 1.0) * vector[mask].conj()
    return result


def main():
    model = TwoOrbitalModel(2, 2, PARAMETERS)
    point = (0.37, 0.0)
    h_bdg = model.bdg_hamiltonian(*point, pairing_twist="linear")
    state = gaussian_bcs_state(h_bdg)
    vector = fock_vector(state.z, state.reference_mask)
    vector /= np.linalg.norm(vector)
    hamiltonian = physical_hamiltonian(h_bdg)
    energies, vectors = np.linalg.eigh(hamiltonian)
    ground = vectors[:, 0]
    overlap = np.vdot(ground, vector)
    print(f"Gaussian/Fock ground overlap magnitude: {abs(overlap):.12f}")
    reflected = reflected_vector_correct(vector, model)
    alpha = np.vdot(vector, reflected)
    print(f"<psi|Theta psi>: {alpha:+.12f}, magnitude={abs(alpha):.12f}")
    print(f"||Theta psi-psi||/||psi||: {np.linalg.norm(reflected-vector):.3e}")
    print(f"||Theta psi+psi||/||psi||: {np.linalg.norm(reflected+vector):.3e}")
    state2 = gaussian_bcs_state(model.bdg_hamiltonian(0.38, 0.0, pairing_twist="linear"), reference_mask=state.reference_mask)
    vector2 = fock_vector(state2.z, state.reference_mask)
    vector2 /= np.linalg.norm(vector2)
    direct = np.vdot(vector, vector2)
    determinant = normalized_gaussian_overlap(state.z, state2.z)
    print(f"Nearby direct overlap: {direct:+.12f}")
    print(f"Nearby determinant overlap: {determinant:+.12f}")


if __name__ == "__main__":
    main()
