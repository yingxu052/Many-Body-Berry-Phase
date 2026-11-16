"""Compare the extracted reference amplitude with Gaussian norm factors."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import PARAMETERS
from model import TwoOrbitalModel
from vertex_real_gauge_berry import reflection_symmetries, state_and_antiunitary_phase


model = TwoOrbitalModel(4, 4, PARAMETERS)
h = model.bdg_hamiltonian(0.0, 0.0, pairing_twist="linear")
action = reflection_symmetries(model)[0]
z, alpha = state_and_antiunitary_phase(h, action)
determinant = np.linalg.det(np.eye(model.n_modes) + z.conj().T @ z).real
print(f"|alpha|={abs(alpha):.12f}")
print(f"det(I+Z^dag Z)={determinant:.12f}")
print(f"sqrt determinant={np.sqrt(determinant):.12f}")
print(f"|alpha|*det={abs(alpha)*determinant:.12f}")
print(f"|alpha|*sqrt(det)={abs(alpha)*np.sqrt(determinant):.12f}")
