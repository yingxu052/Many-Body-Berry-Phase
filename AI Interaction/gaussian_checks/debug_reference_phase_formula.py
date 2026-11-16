"""Validate the Pfaffian antiunitary-phase formula against small Fock space."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from check_antiunitary_fock import reflected_vector_correct
from complex_twist_berry_convergence import PARAMETERS
from model import TwoOrbitalModel
from validate_gaussian_overlap import fock_vector
import vertex_real_gauge_berry as vertex


model = TwoOrbitalModel(2, 2, PARAMETERS)
h = model.bdg_hamiltonian(0.37, 0.0, pairing_twist="linear")
for mask in (0x00, 0x03, 0x12, 0x53, 0x69):
    try:
        vertex.REFERENCE_MASK = mask
        action = vertex.reference_action(model, "x")
        z, extracted = vertex.state_and_antiunitary_phase(h, action)
    except ValueError as error:
        print(f"mask={mask:02x}: {error}")
        continue
    vector = fock_vector(z, mask)
    vector /= np.linalg.norm(vector)
    exact = np.vdot(vector, reflected_vector_correct(vector, model))
    print(
        f"mask={mask:02x}: extracted={extracted:+.8f}, |extracted|={abs(extracted):.8f}; "
        f"Fock={exact:+.8f}, |Fock|={abs(exact):.8f}"
    )
