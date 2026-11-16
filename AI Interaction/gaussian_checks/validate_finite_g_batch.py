"""Check batched finite-g enumeration against the original scalar routine."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))

from finite_g_overlap import finite_g_overlap, finite_g_overlaps
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state


model = TwoOrbitalModel(4, 4, ModelParameters(m=-2.0))
reference = 0x5057F515
state1 = gaussian_bcs_state(model.bdg_hamiltonian(0.0, 0.0), reference_mask=reference)
state2 = gaussian_bcs_state(model.bdg_hamiltonian(0.3, 0.4), reference_mask=reference)
g_values = [0.0, 0.5, 1.0]
batch = finite_g_overlaps(state1.z, state2.z, 16, g_values, reference, workers=12)
scalar = finite_g_overlap(state1.z, state2.z, 16, 0.5, reference, workers=12)
error = abs(batch[1]["normalized_overlap"] - scalar["normalized_overlap"])
norm_error = max(
    abs(batch[1]["norm1_fraction"] - scalar["norm1_fraction"]),
    abs(batch[1]["norm2_fraction"] - scalar["norm2_fraction"]),
)
print(f"Batched versus scalar overlap error: {error:.3e}")
print(f"Batched versus scalar norm error: {norm_error:.3e}")
print(f"Batched overlaps: {[item['normalized_overlap'] for item in batch]}")
assert error < 1e-10
assert norm_error < 1e-10
print("Batched finite-g validation passed.")
