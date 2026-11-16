"""Independently check a Gaussian Berry loop using explicit Fock vectors."""

from __future__ import annotations

import math

import numpy as np

from berry_phase import twist_loop
from gaussian_overlap import normalized_gaussian_overlap
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state
from validate_gaussian_overlap import fock_vector


def main() -> None:
    model = TwoOrbitalModel(2, 2, ModelParameters(t=1.0, t1=1.0, m=1.5, delta_p=1.0, delta_s=0.5))
    _, points = twist_loop(4)
    states = [gaussian_bcs_state(model.bdg_hamiltonian(*point)) for point in points]
    references = {state.reference_mask for state in states}
    assert len(references) == 1
    reference = next(iter(references))
    vectors = [fock_vector(state.z, reference) for state in states]
    vectors = [vector / np.linalg.norm(vector) for vector in vectors]

    direct_phases = []
    determinant_phases = []
    for index, state in enumerate(states):
        next_index = (index + 1) % len(states)
        direct = np.vdot(vectors[index], vectors[next_index])
        determinant = normalized_gaussian_overlap(state.z, states[next_index].z)
        direct_phases.append(float(np.angle(direct)))
        determinant_phases.append(float(np.angle(determinant)))

    direct_total = math.fsum(direct_phases)
    determinant_total = math.fsum(determinant_phases)
    error = abs(direct_total - determinant_total)
    print(f"Explicit 2 x 2 Fock-space Berry phase: {direct_total:.3e} rad")
    print(f"Determinant-formula Berry phase: {determinant_total:.3e} rad")
    print(f"Loop phase difference: {error:.3e} rad")
    if error > 1e-10:
        raise SystemExit("FAILED: independent Gaussian loop calculations disagree")
    print("PASSED: the zero Gaussian loop phase is reproduced by explicit Fock vectors")


if __name__ == "__main__":
    main()
