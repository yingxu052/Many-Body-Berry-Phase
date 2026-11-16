"""Validate determinant Gaussian overlaps against explicit small Fock spaces."""

from __future__ import annotations

import numpy as np

from gaussian_overlap import normalized_gaussian_overlap
from model import ModelParameters, TwoOrbitalModel, bcs_amplitude, gaussian_bcs_state


def basis_change_sign(original: int, reference: int, n_modes: int) -> int:
    """Sign for writing a d-basis occupation in the canonical c Fock basis."""
    state = 0
    sign = 1

    def create(mode: int) -> None:
        nonlocal state, sign
        bit = 1 << mode
        if state & bit:
            sign = 0
            return
        if (state & (bit - 1)).bit_count() % 2:
            sign = -sign
        state |= bit

    def annihilate(mode: int) -> None:
        nonlocal state, sign
        bit = 1 << mode
        if not state & bit:
            sign = 0
            return
        if (state & (bit - 1)).bit_count() % 2:
            sign = -sign
        state &= ~bit

    for mode in range(n_modes):
        if reference & (1 << mode):
            create(mode)
    for mode in range(n_modes - 1, -1, -1):
        if original & (1 << mode):
            if reference & (1 << mode):
                annihilate(mode)
            else:
                create(mode)
    return sign if state == (original ^ reference) else 0


def fock_vector(z: np.ndarray, reference_mask: int) -> np.ndarray:
    n_modes = z.shape[0]
    vector = np.zeros(1 << n_modes, dtype=np.complex128)
    for d_state in range(1 << n_modes):
        original = d_state ^ reference_mask
        occupied = [mode for mode in range(n_modes) if d_state & (1 << mode)]
        vector[original] = basis_change_sign(original, reference_mask, n_modes) * bcs_amplitude(z, occupied)
    return vector


def direct_overlap(z1: np.ndarray, z2: np.ndarray, reference_mask: int) -> complex:
    vector1 = fock_vector(z1, reference_mask)
    vector2 = fock_vector(z2, reference_mask)
    return np.vdot(vector1, vector2) / np.sqrt(np.vdot(vector1, vector1).real * np.vdot(vector2, vector2).real)


def main() -> None:
    rng = np.random.default_rng(20261028)
    largest_error = 0.0
    for n_modes in (2, 4, 6):
        references = (0, sum(1 << mode for mode in range(0, n_modes, 2)))
        for reference in references:
            for _ in range(5):
                z1 = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
                z2 = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
                z1 = 0.2 * (z1 - z1.T)
                z2 = 0.2 * (z2 - z2.T)
                expected = direct_overlap(z1, z2, reference)
                obtained = normalized_gaussian_overlap(z1, z2)
                largest_error = max(largest_error, abs(expected - obtained))

    model = TwoOrbitalModel(2, 2, ModelParameters(m=1.5))
    state1 = gaussian_bcs_state(model.bdg_hamiltonian(0.23, -0.17))
    state2 = gaussian_bcs_state(model.bdg_hamiltonian(0.24, -0.17))
    assert state1.reference_mask == state2.reference_mask
    expected = direct_overlap(state1.z, state2.z, state1.reference_mask)
    obtained = normalized_gaussian_overlap(state1.z, state2.z)
    largest_error = max(largest_error, abs(expected - obtained))

    print(f"Gaussian determinant/Fock-space maximum overlap error: {largest_error:.3e}")
    if largest_error > 1e-10:
        raise SystemExit("FAILED: determinant Gaussian overlap disagrees with Fock space")
    print("PASSED: normalized unprojected Gaussian overlaps agree with exact Fock space")


if __name__ == "__main__":
    main()
