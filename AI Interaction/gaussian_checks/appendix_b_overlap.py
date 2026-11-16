"""Gaussian-state overlaps using Appendix B of SET_Transition_2DJWT.

For BdG eigenvector blocks (U_i, V_i), define
    U_overlap = U_1^dagger U_2 + V_1^dagger V_2.
The normalized overlap is exp(tr(log(U_overlap^dagger))/2), evaluated here as
sqrt(det(U_overlap^dagger)) with NumPy's principal square-root branch.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server_ready"))
import gutzwiller_mc_sweep as gm


def appendix_b_overlap(u1, v1, u2, v2) -> complex:
    """Return <Psi_1|Psi_2> using Eqs. (B7)-(B8)."""
    u_overlap = u1.conj().T @ u2 + v1.conj().T @ v2
    return complex(np.sqrt(np.linalg.det(u_overlap.conj().T)))


def berry_phases(model: gm.TwoOrbitalModel, steps_per_leg: int) -> dict:
    points = gm.twist_loop(steps_per_leg)
    blocks = [gm.positive_blocks(model.bdg_hamiltonian(*point)) for point in points]
    appendix_overlaps = [
        appendix_b_overlap(
            blocks[i][0], blocks[i][1],
            blocks[(i + 1) % len(blocks)][0], blocks[(i + 1) % len(blocks)][1],
        )
        for i in range(len(blocks))
    ]

    # Independent comparison with the previously used common-reference Z formula.
    reference = gm.choose_reference(blocks, model.n_cells, None)
    states = [gm.make_state(u, v, gap, reference) for u, v, gap in blocks]
    determinant_overlaps = [
        gm.normalized_gaussian_overlap(
            state.z, states[(i + 1) % len(states)].z
        )
        for i, state in enumerate(states)
    ]

    def total_phase(overlaps):
        total = math.fsum(float(np.angle(overlap)) for overlap in overlaps)
        return float(np.angle(np.exp(1j * total)))

    appendix_phase = total_phase(appendix_overlaps)
    determinant_phase = total_phase(determinant_overlaps)
    phase_differences = np.angle(
        np.array(appendix_overlaps) / np.array(determinant_overlaps)
    )
    return {
        "steps_per_leg": steps_per_leg,
        "appendix_b_phase_radians": appendix_phase,
        "appendix_b_phase_over_pi": appendix_phase / np.pi,
        "determinant_phase_radians": determinant_phase,
        "phase_difference_radians": np.angle(np.exp(1j * (appendix_phase - determinant_phase))),
        "maximum_link_absolute_error": float(np.max(np.abs(
            np.array(appendix_overlaps) - np.array(determinant_overlaps)
        ))),
        "maximum_link_phase_difference": float(np.max(np.abs(phase_differences))),
        "maximum_link_magnitude_difference": float(np.max(np.abs(
            np.abs(appendix_overlaps) - np.abs(determinant_overlaps)
        ))),
        "total_product_ratio": complex(
            np.prod(appendix_overlaps) / np.prod(determinant_overlaps)
        ).real,
        "minimum_appendix_link_magnitude": float(np.min(np.abs(appendix_overlaps))),
        "reference_mask_for_determinant_check": reference,
        "appendix_link_phases": [float(np.angle(x)) for x in appendix_overlaps],
        "determinant_link_phases": [float(np.angle(x)) for x in determinant_overlaps],
    }
