"""Regression checks for the corrected server-ready manuscript Hamiltonian."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server_ready"))
sys.path.insert(0, str(ROOT / "gaussian_checks"))

import berry_phase as exact
import gutzwiller_mc_sweep as mc
import gutzwiller_mc_sweep_parallel as mcp
from manuscript_twist_hamiltonian import bdg_manuscript


points = [(0.0, 0.0), (0.3, 0.4), (1.2, 2.1), (np.pi, np.pi / 2)]
params = exact.ModelParameters(t=1.0, t1=1.0, m=-2.0, delta_p=1.0, delta_s=0.5)
models = [
    ("exact", exact.TwoOrbitalModel(4, 4, params)),
    ("mc", mc.TwoOrbitalModel(4, 4, mc.ModelParameters())),
    ("parallel", mcp.TwoOrbitalModel(4, 4, mcp.ModelParameters())),
]
reference_model = mc.TwoOrbitalModel(4, 4, mc.ModelParameters())
for name, model in models:
    errors = [
        float(np.max(np.abs(model.bdg_hamiltonian(*point) -
                            bdg_manuscript(reference_model, *point, seam="first"))))
        for point in points
    ]
    print(f"{name} maximum matrix error: {max(errors):.3e}")
    assert max(errors) < 1e-12

steps = 32
unprojected = exact.compute_berry_phases([0.0], steps_per_leg=steps)[0]
print(f"Exact server Gaussian phase ({steps} steps/leg): {unprojected:+.15f}")
assert abs(abs(unprojected) - np.pi / 2) < 1e-10

model = mc.TwoOrbitalModel(4, 4, mc.ModelParameters())
loop = mc.twist_loop(steps)
blocks = [mc.positive_blocks(model.bdg_hamiltonian(*point)) for point in loop]
reference = mc.choose_reference(blocks, model.n_cells, None)
states = [mc.make_state(u, v, gap, reference) for u, v, gap in blocks]
for name, module in [("serial", mc), ("parallel", mcp)]:
    if name == "parallel":
        parallel_model = module.TwoOrbitalModel(4, 4, module.ModelParameters())
        parallel_blocks = [module.positive_blocks(parallel_model.bdg_hamiltonian(*point)) for point in loop]
        parallel_states = [module.make_state(u, v, gap, reference) for u, v, gap in parallel_blocks]
        result = module.compute_berry_phase(
            0.0, parallel_states, loop, reference,
            chains=1, burn_in=1, samples=1, seed=17, link_jobs=1,
        )
    else:
        result = module.compute_berry_phase(
            0.0, states, loop, reference,
            chains=1, burn_in=1, samples=1, seed=17,
        )
    print(f"{name} MC-driver Gaussian phase: {result['berry_phase']:+.15f}")
    assert abs(result["berry_phase"] - unprojected) < 1e-12

# Exercise finite-g exact enumeration on the full 4x4 physical Hilbert space.
projected = exact.compute_berry_phases([0.5, 1.0], steps_per_leg=2)
print("Exact 4x4 finite-g phases (2 steps/leg):", projected)
assert np.all(np.isfinite(projected))

# Minimal sampling smoke tests. The process pool is tested through the CLI.
for name, module in [("serial", mc), ("parallel", mcp)]:
    small_model = module.TwoOrbitalModel(4, 4, module.ModelParameters())
    small_loop = module.twist_loop(1)
    small_blocks = [module.positive_blocks(small_model.bdg_hamiltonian(*point)) for point in small_loop]
    small_reference = module.choose_reference(small_blocks, small_model.n_cells, None)
    small_states = [module.make_state(u, v, gap, small_reference) for u, v, gap in small_blocks]
    kwargs = {"link_jobs": 1} if name == "parallel" else {}
    result = module.compute_berry_phase(
        0.5, small_states, small_loop, small_reference,
        chains=1, burn_in=1, samples=2, seed=23, **kwargs,
    )
    print(f"{name} finite-g sampling smoke phase: {result['berry_phase']:+.9f}")
    assert math.isfinite(result["berry_phase"])

print("All server-ready regression checks passed.")
