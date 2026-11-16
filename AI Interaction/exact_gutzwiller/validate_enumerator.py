"""Validate the compiled enumerator against independent small-system calculations."""

from __future__ import annotations

import numpy as np

from model import (
    ModelParameters,
    TwoOrbitalModel,
    bcs_amplitude,
    gaussian_bcs_state,
    reference_pfaffian,
)
from overlap import compiled_pfaffian, projected_overlap
from validate_conventions import fock_hamiltonian


def fock_projected_state(model, phi_x, phi_y):
    h, delta = model.normal_and_pairing_blocks(phi_x, phi_y)
    h_fock = fock_hamiltonian(h, delta)
    even_states = [s for s in range(1 << model.n_modes) if s.bit_count() % 2 == 0]
    energies, vectors = np.linalg.eigh(h_fock[np.ix_(even_states, even_states)])
    full = np.zeros(1 << model.n_modes, dtype=np.complex128)
    full[even_states] = vectors[:, 0]
    for state in even_states:
        for cell in range(model.n_cells):
            if (state & (1 << (2 * cell))) and (state & (1 << (2 * cell + 1))):
                full[state] = 0.0
                break
    norm = float(np.vdot(full, full).real)
    return full, norm


def main():
    rng = np.random.default_rng(20261028)
    largest_error = 0.0
    for n in (0, 2, 4, 6, 8, 10):
        for _ in range(5):
            matrix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            matrix = 0.5 * (matrix - matrix.T)
            expected = reference_pfaffian(matrix)
            obtained = compiled_pfaffian(matrix)
            largest_error = max(largest_error, abs(expected - obtained))
    print(f"Random complex Pfaffian maximum error: {largest_error:.3e}")
    if largest_error > 1e-9:
        raise SystemExit("FAILED: compiled Pfaffian disagrees with recursive reference")

    cases = [
        ("regular vacuum reference", 1.5, (0.23, -0.17), (0.24, -0.17)),
        ("singular vacuum reference", 0.0, (0.0, 0.0), (0.05, 0.0)),
    ]
    for label, mass, phi1, phi2 in cases:
        model = TwoOrbitalModel(2, 2, ModelParameters(m=mass))
        state1 = gaussian_bcs_state(model.bdg_hamiltonian(*phi1))
        state2 = gaussian_bcs_state(model.bdg_hamiltonian(*phi2))
        result = projected_overlap(
            state1.z, state2.z, model.n_cells,
            reference_mask1=state1.reference_mask,
            reference_mask2=state2.reference_mask,
            workers=2,
        )

        fock1, projected_norm1 = fock_projected_state(model, *phi1)
        fock2, projected_norm2 = fock_projected_state(model, *phi2)
        fock_overlap = np.vdot(fock1, fock2) / np.sqrt(projected_norm1 * projected_norm2)

        unprojected_norms = []
        for z in (state1.z, state2.z):
            total = 0.0
            for state in range(1 << model.n_modes):
                occupied = [mode for mode in range(model.n_modes) if state & (1 << mode)]
                total += abs(bcs_amplitude(z, occupied)) ** 2
            unprojected_norms.append(total)
        enumeration_projection_norms = (
            result["norm1"] / unprojected_norms[0],
            result["norm2"] / unprojected_norms[1],
        )
        norm_error = max(
            abs(enumeration_projection_norms[0] - projected_norm1),
            abs(enumeration_projection_norms[1] - projected_norm2),
        )
        overlap_error = abs(abs(result["normalized_overlap"]) - abs(fock_overlap))
        print(f"Case: {label}")
        print(f"  Reference masks: {state1.reference_mask}, {state2.reference_mask}")
        print(f"  Projected-norm error against Fock space: {norm_error:.3e}")
        print(f"  Normalized-overlap error against Fock space: {overlap_error:.3e}")
        if norm_error > 1e-10 or overlap_error > 1e-10:
            raise SystemExit("FAILED: projected enumeration disagrees with Fock space")
    print("PASSED: compiled projected norms and overlaps agree with exact Fock space")


if __name__ == "__main__":
    main()
