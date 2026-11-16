"""Compare fixed-reference Gaussian states from two construction paths."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from check_antiunitary_fock import physical_hamiltonian, reflected_vector_correct
from complex_twist_berry_convergence import PARAMETERS
from model import TwoOrbitalModel, gaussian_bcs_state
from validate_gaussian_overlap import fock_vector
import vertex_real_gauge_berry as vertex


model = TwoOrbitalModel(2, 2, PARAMETERS)
h = model.bdg_hamiltonian(0.37, 0.0, pairing_twist="linear")
mask = 0x03
vertex.REFERENCE_MASK = mask
action = vertex.reference_action(model, "x")
z1, extracted = vertex.state_and_antiunitary_phase(h, action)
z2 = gaussian_bcs_state(h, reference_mask=mask).z
print(f"max |z1-z2|={np.max(np.abs(z1-z2)):.3e}")
for label, z in (("vertex", z1), ("model", z2)):
    vector = fock_vector(z, mask)
    vector /= np.linalg.norm(vector)
    alpha = np.vdot(vector, reflected_vector_correct(vector, model))
    print(f"{label}: alpha={alpha:+.12f}, |alpha|={abs(alpha):.12f}")
