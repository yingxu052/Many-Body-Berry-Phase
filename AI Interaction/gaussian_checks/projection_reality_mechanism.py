"""Targeted checks of the reality class responsible for projected-overlap phases."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import PARAMETERS, twist_loop
from gaussian_overlap import normalized_gaussian_overlap
from model import TwoOrbitalModel, gaussian_bcs_state
from single_occupancy_overlap import single_occupancy_overlap


REFERENCE_MASK = 0xAAC8DA33


def mixed_hamiltonian(model, point, mixing, imaginary=0.0):
    linear = model.bdg_hamiltonian(*point, pairing_twist="linear", pairing_imag=imaginary)
    cosine = model.bdg_hamiltonian(*point, pairing_twist="cosine")
    return (1.0 - mixing) * linear + mixing * cosine


def main():
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    point = (0.37, 0.0)
    print("Pairing-matrix reality diagnostics")
    for label, hamiltonian in (
        ("linear", mixed_hamiltonian(model, point, 0.0)),
        ("mixed", mixed_hamiltonian(model, point, 0.5)),
        ("imaginary", mixed_hamiltonian(model, point, 0.5, imaginary=1.0e-4)),
    ):
        state = gaussian_bcs_state(hamiltonian, reference_mask=REFERENCE_MASK)
        z = state.z
        print(
            f"  {label:10s}: ||Re Z||={np.linalg.norm(z.real):.8f}, "
            f"||Im Z||={np.linalg.norm(z.imag):.8f}, "
            f"max |Z-Z*|={np.max(np.abs(z-z.conj())):.8f}"
        )

    # Compare two real-profile states, and then states differing by a uniform
    # real mass perturbation that preserves the lattice translations.
    h0 = mixed_hamiltonian(model, point, 0.5)
    h1 = mixed_hamiltonian(model, point, 0.5001)
    mass = np.zeros_like(h0)
    n = model.n_modes
    for cell in range(model.n_cells):
        mass[2 * cell, 2 * cell] = 0.013
        mass[2 * cell + 1, 2 * cell + 1] = -0.013
    mass[n:, n:] = -mass[:n, :n].T
    h2 = h0 + mass
    # A comparably sized imaginary pairing perturbation breaks the reality class.
    h3 = mixed_hamiltonian(model, point, 0.5, imaginary=1.0e-4)
    states = [gaussian_bcs_state(h, reference_mask=REFERENCE_MASK) for h in (h0, h1, h2, h3)]
    print("\nOverlap-ratio tests")
    for label, state2 in zip(("real profile", "real mass", "imaginary pairing"), states[1:]):
        gaussian = normalized_gaussian_overlap(states[0].z, state2.z)
        projected = single_occupancy_overlap(
            states[0].z, state2.z, model.n_cells,
            reference_mask1=REFERENCE_MASK, reference_mask2=REFERENCE_MASK,
            workers=8,
        )["normalized_overlap"]
        ratio = projected / gaussian
        print(
            f"  {label:18s}: ratio={ratio.real:+.15f}{ratio.imag:+.3e}j, "
            f"phase difference={np.angle(ratio):+.3e}"
        )


if __name__ == "__main__":
    main()
