"""Validate single-occupancy overlaps against explicit small Fock spaces."""

from __future__ import annotations

import numpy as np

from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state
from single_occupancy_overlap import single_occupancy_overlap
from validate_gaussian_overlap import fock_vector


def project_single_occupancy(vector: np.ndarray, n_cells: int) -> np.ndarray:
    projected = vector.copy()
    for state in range(len(vector)):
        for cell in range(n_cells):
            occupancy = bool(state & (1 << (2 * cell))) + bool(state & (1 << (2 * cell + 1)))
            if occupancy != 1:
                projected[state] = 0.0
                break
    return projected


def direct_result(z1, z2, n_cells, reference1, reference2):
    vector1 = project_single_occupancy(fock_vector(z1, reference1), n_cells)
    vector2 = project_single_occupancy(fock_vector(z2, reference2), n_cells)
    full1 = fock_vector(z1, reference1)
    full2 = fock_vector(z2, reference2)
    norm1 = float(np.vdot(vector1, vector1).real)
    norm2 = float(np.vdot(vector2, vector2).real)
    return {
        "normalized_overlap": np.vdot(vector1, vector2) / np.sqrt(norm1 * norm2),
        "norm1_fraction": norm1 / float(np.vdot(full1, full1).real),
        "norm2_fraction": norm2 / float(np.vdot(full2, full2).real),
    }


def compare(z1, z2, n_cells, reference1, reference2) -> float:
    expected = direct_result(z1, z2, n_cells, reference1, reference2)
    obtained = single_occupancy_overlap(
        z1, z2, n_cells, reference1, reference2, workers=2,
    )
    error = max(
        abs(expected["normalized_overlap"] - obtained["normalized_overlap"]),
        abs(expected["norm1_fraction"] - obtained["norm1_fraction"]),
        abs(expected["norm2_fraction"] - obtained["norm2_fraction"]),
    )
    if obtained["counted_configurations"] != 2**n_cells:
        raise AssertionError("The enumerator did not visit the full spin Hilbert space")
    return float(error)


def main() -> None:
    rng = np.random.default_rng(20261030)
    largest_error = 0.0
    # The c-basis and d-basis particle numbers must have compatible parity.
    for n_cells in (2, 4):
        n_modes = 2 * n_cells
        for _ in range(4):
            z1 = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
            z2 = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(size=(n_modes, n_modes))
            z1 = 0.15 * (z1 - z1.T)
            z2 = 0.15 * (z2 - z2.T)
            reference1 = int(rng.integers(0, 1 << n_modes))
            reference2 = int(rng.integers(0, 1 << n_modes))
            if reference1.bit_count() % 2:
                reference1 ^= 1
            if reference2.bit_count() % 2:
                reference2 ^= 1
            largest_error = max(
                largest_error, compare(z1, z2, n_cells, reference1, reference2),
            )

    model = TwoOrbitalModel(2, 2, ModelParameters(m=1.5))
    state1 = gaussian_bcs_state(model.bdg_hamiltonian(0.23, -0.17))
    state2 = gaussian_bcs_state(model.bdg_hamiltonian(0.24, -0.17))
    largest_error = max(
        largest_error,
        compare(
            state1.z, state2.z, model.n_cells,
            state1.reference_mask, state2.reference_mask,
        ),
    )
    print(f"Maximum single-occupancy overlap/norm error: {largest_error:.3e}")
    if largest_error > 1e-10:
        raise SystemExit("FAILED: single-occupancy enumeration disagrees with Fock space")
    print("PASSED: single-occupancy projected norms and complex overlaps agree with Fock space")


if __name__ == "__main__":
    main()
