"""Validate the standalone exact-enumeration manuscript update."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
sys.path.insert(0, str(ROOT / "gaussian_checks"))

from model import ModelParameters, TwoOrbitalModel
from manuscript_twist_hamiltonian import bdg_manuscript
from unprojected_nu2_check import compute_loop


model = TwoOrbitalModel(4, 4, ModelParameters(t=1.0, t1=1.0, m=-2.0, delta_p=1.0, delta_s=0.5))
points = [(0.0, 0.0), (0.3, 0.4), (1.2, 2.1), (np.pi, np.pi / 2)]
errors = [
    float(np.max(np.abs(model.bdg_hamiltonian(*point) -
                        bdg_manuscript(model, *point, seam="first"))))
    for point in points
]
print(f"Default exact-model maximum matrix error: {max(errors):.3e}")
assert max(errors) < 1e-12

result = compute_loop(32, 0x5057F515)
phase = result["berry_phase_radians"]
print(f"Standalone exact Gaussian phase (32 steps/leg): {phase:+.15f}")
print(f"Minimum link magnitude: {result['minimum_link_magnitude']:.15f}")
assert abs(abs(phase) - np.pi / 2) < 1e-10
print("Standalone exact-model checks passed.")
