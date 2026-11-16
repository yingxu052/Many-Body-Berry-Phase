"""Search for reflection times complex-conjugation symmetries of the twisted BdG Hamiltonian."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import PARAMETERS
from model import TwoOrbitalModel


def mixed_hamiltonian(model, point, imaginary=0.0):
    linear = model.bdg_hamiltonian(*point, pairing_twist="linear", pairing_imag=imaginary)
    cosine = model.bdg_hamiltonian(*point, pairing_twist="cosine")
    return 0.5 * (linear + cosine)


def orbital_candidates():
    phases = (1.0, -1.0, 1.0j, -1.0j)
    candidates = []
    for qc in phases:
        for qd in phases:
            candidates.append((f"diag({qc},{qd})", np.diag([qc, qd])))
            swap = np.array([[0.0, qc], [qd, 0.0]], dtype=complex)
            candidates.append((f"swap({qc},{qd})", swap))
    return candidates


def spatial_reflections(model, direction):
    """Reflect the twisted coordinate and allow translation/reflection in the other."""
    for sign_other in (-1, 1):
        for shift_twisted in range(4):
            for shift_other in range(4):
                mapping = []
                for y in range(4):
                    for x in range(4):
                        if direction == "x":
                            xp = (-x + shift_twisted) % 4
                            yp = (sign_other * y + shift_other) % 4
                        else:
                            xp = (sign_other * x + shift_other) % 4
                            yp = (-y + shift_twisted) % 4
                        mapping.append((xp, yp))
                label = (
                    f"R{direction}: twisted_shift={shift_twisted}, "
                    f"other_sign={sign_other}, other_shift={shift_other}"
                )
                yield label, mapping


def nambu_unitary(model, mapping, orbital):
    n = model.n_modes
    unitary = np.zeros((n, n), dtype=complex)
    for y in range(4):
        for x in range(4):
            xp, yp = mapping[4 * y + x]
            for source_orbital in range(2):
                source = model.mode(x, y, source_orbital)
                for target_orbital in range(2):
                    target = model.mode(xp, yp, target_orbital)
                    unitary[target, source] = orbital[target_orbital, source_orbital]
    return np.block([
        [unitary, np.zeros_like(unitary)],
        [np.zeros_like(unitary), unitary.conj()],
    ])


def error(unitary, hamiltonian, target=None):
    transformed = unitary @ hamiltonian.conj() @ unitary.conj().T
    if target is None:
        target = hamiltonian
    return np.linalg.norm(transformed - target) / np.linalg.norm(target)


def main():
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    tests = (
        ("x", (0.37, 0.0)),
        ("y", (np.pi, 0.83)),
        ("x", (2.0 * np.pi - 0.37, np.pi)),
        ("y", (2.0 * np.pi, 2.0 * np.pi - 0.83)),
    )
    for direction, point in tests:
        real = mixed_hamiltonian(model, point)
        imaginary = mixed_hamiltonian(model, point, imaginary=1.0e-4)
        opposite_imaginary = mixed_hamiltonian(model, point, imaginary=-1.0e-4)
        candidates = []
        for spatial_label, mapping in spatial_reflections(model, direction):
            for orbital_label, orbital in orbital_candidates():
                unitary = nambu_unitary(model, mapping, orbital)
                real_error = error(unitary, real)
                candidates.append((real_error, spatial_label, orbital_label, unitary))
        candidates.sort(key=lambda item: item[0])
        print(f"\nPoint {point}, reflection direction {direction}")
        for real_error, spatial_label, orbital_label, unitary in candidates[:1]:
            print(
                f"  real error={real_error:.3e}, imaginary error={error(unitary, imaginary):.3e}, "
                f"+eta to -eta error={error(unitary, imaginary, opposite_imaginary):.3e}\n"
                f"    {spatial_label}; {orbital_label}"
            )


if __name__ == "__main__":
    main()
