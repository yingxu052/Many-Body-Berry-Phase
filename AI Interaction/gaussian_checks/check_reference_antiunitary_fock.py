"""Test Gaussian ground-state and antiunitary phases across Slater references."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from check_antiunitary_fock import physical_hamiltonian, reflected_vector_correct
from complex_twist_berry_convergence import PARAMETERS
from model import TwoOrbitalModel, gaussian_bcs_state
from validate_gaussian_overlap import fock_vector


def main():
    model = TwoOrbitalModel(2, 2, PARAMETERS)
    h_bdg = model.bdg_hamiltonian(0.37, 0.0, pairing_twist="linear")
    hamiltonian = physical_hamiltonian(h_bdg)
    energies, vectors = np.linalg.eigh(hamiltonian)
    ground = vectors[:, 0]
    rows = []
    for mask in range(1 << model.n_modes):
        try:
            state = gaussian_bcs_state(h_bdg, reference_mask=mask)
        except ValueError:
            continue
        vector = fock_vector(state.z, mask)
        vector /= np.linalg.norm(vector)
        ground_overlap = np.vdot(ground, vector)
        reflected = reflected_vector_correct(vector, model)
        alpha = np.vdot(vector, reflected)
        eigen_error = np.linalg.norm(reflected - alpha * vector)
        rows.append((mask, abs(ground_overlap), alpha, eigen_error))
    worst_ground = min(rows, key=lambda row: row[1])
    worst_eigen = max(rows, key=lambda row: row[3])
    print(f"Valid references: {len(rows)}")
    print(f"Worst ground overlap: mask={worst_ground[0]:02x}, |overlap|={worst_ground[1]:.12f}")
    print(
        f"Worst antiunitary eigenstate: mask={worst_eigen[0]:02x}, "
        f"alpha={worst_eigen[2]:+.6f}, error={worst_eigen[3]:.3e}"
    )
    non_unit_alpha = [row for row in rows if abs(abs(row[2]) - 1.0) > 1.0e-8]
    print(f"References with non-unit antiunitary overlap: {len(non_unit_alpha)}")
    nontrivial = [row for row in rows if abs(np.angle(row[2])) > 1.0e-8]
    print(f"References with nontrivial alpha phase: {len(nontrivial)}")
    for row in nontrivial[:10]:
        print(f"  mask={row[0]:02x}: alpha={row[2]:+.8f}, |ground overlap|={row[1]:.8f}")


if __name__ == "__main__":
    main()
