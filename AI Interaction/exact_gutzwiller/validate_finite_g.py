"""Validate exact finite-g overlaps against direct Fock-space projection."""

from __future__ import annotations

import numpy as np

from finite_g_overlap import finite_g_overlap
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state
from validate_gaussian_overlap import fock_vector


def direct_finite_g(vector1, vector2, n_cells, g):
    projected1 = vector1.copy()
    projected2 = vector2.copy()
    for state in range(len(vector1)):
        double_occupancy = sum(
            bool(state & (1 << (2 * cell))) and bool(state & (1 << (2 * cell + 1)))
            for cell in range(n_cells)
        )
        weight = g ** double_occupancy
        projected1[state] *= weight
        projected2[state] *= weight
    numerator = np.vdot(projected1, projected2)
    denominator = np.sqrt(np.vdot(projected1, projected1).real * np.vdot(projected2, projected2).real)
    return numerator / denominator


def main() -> None:
    model = TwoOrbitalModel(2, 2, ModelParameters(m=1.5))
    state1 = gaussian_bcs_state(model.bdg_hamiltonian(0.23, -0.17))
    state2 = gaussian_bcs_state(model.bdg_hamiltonian(0.24, -0.17))
    assert state1.reference_mask == state2.reference_mask
    vector1 = fock_vector(state1.z, state1.reference_mask)
    vector2 = fock_vector(state2.z, state2.reference_mask)

    largest_error = 0.0
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        expected = direct_finite_g(vector1, vector2, model.n_cells, g)
        result = finite_g_overlap(
            state1.z, state2.z, model.n_cells, g,
            reference_mask=state1.reference_mask, workers=2,
        )
        error = abs(expected - result["normalized_overlap"])
        largest_error = max(largest_error, error)
        print(
            f"g={g:.2f}: direct={expected.real:+.12f}{expected.imag:+.12f}i, "
            f"Wick={result['normalized_overlap'].real:+.12f}{result['normalized_overlap'].imag:+.12f}i, "
            f"error={error:.3e}"
        )

    print(f"Maximum finite-g overlap error: {largest_error:.3e}")
    if largest_error > 1e-10:
        raise SystemExit("FAILED: finite-g Wick overlap disagrees with direct Fock space")
    print("PASSED: finite-g overlaps, including both endpoints, agree with Fock space")


if __name__ == "__main__":
    main()
