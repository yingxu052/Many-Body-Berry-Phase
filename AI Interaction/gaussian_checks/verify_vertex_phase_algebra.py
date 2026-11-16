"""Verify the no-inverse vertex products Theta_out Theta_in and their phases."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gaussian_checks"))
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from analytic_vertex_operators import operators, relative_error
from complex_twist_berry_convergence import PARAMETERS
from model import TwoOrbitalModel, gaussian_bcs_state, gaussian_parity
from vertex_real_gauge_berry import (
    REFERENCE_MASK,
    basis_change_sign,
    compute,
    pfaffian,
)


def reflection_action(model, kind):
    """Return the data for <reference|Theta|psi> for Rx, Ry, or Tx Ry."""
    reference_modes = [
        mode for mode in range(model.n_modes) if REFERENCE_MASK & (1 << mode)
    ]

    def forward(x, y):
        if kind == "x":
            return (1 - x) % model.lx, y
        if kind == "y":
            return x, (1 - y) % model.ly
        if kind == "yt":
            return (x + 1) % model.lx, (1 - y) % model.ly
        raise ValueError(kind)

    def inverse(x, y):
        if kind == "x":
            return (1 - x) % model.lx, y
        if kind == "y":
            return x, (1 - y) % model.ly
        return (x - 1) % model.lx, (1 - y) % model.ly

    source_modes = []
    for mode in reference_modes:
        cell, orbital = divmod(mode, 2)
        x, y = cell % model.lx, cell // model.lx
        xp, yp = inverse(x, y)
        source_modes.append(model.mode(xp, yp, orbital))
    source_modes.sort()
    source_mask = sum(1 << mode for mode in source_modes)

    mapped_modes = []
    unitary_phase = 1.0 + 0.0j
    for mode in source_modes:
        cell, orbital = divmod(mode, 2)
        x, y = cell % model.lx, cell // model.lx
        xp, yp = forward(x, y)
        mapped_modes.append(model.mode(xp, yp, orbital))
        if kind in ("y", "yt"):
            unitary_phase *= 1.0j if orbital == 0 else -1.0j
    inversions = sum(
        mapped_modes[i] > mapped_modes[j]
        for i in range(len(mapped_modes))
        for j in range(i + 1, len(mapped_modes))
    )
    if inversions % 2:
        unitary_phase = -unitary_phase
    if set(mapped_modes) != set(reference_modes):
        raise RuntimeError(f"{kind} did not restore the reference configuration")
    return source_mask, unitary_phase


def antiunitary_phase(hamiltonian, action):
    n = hamiltonian.shape[0] // 2
    z = gaussian_bcs_state(hamiltonian, reference_mask=REFERENCE_MASK).z
    source_mask, unitary_phase = action
    transformed_mask = source_mask ^ REFERENCE_MASK
    occupied = [mode for mode in range(n) if transformed_mask & (1 << mode)]
    amplitude = complex(basis_change_sign(source_mask, REFERENCE_MASK, n))
    if occupied:
        amplitude *= pfaffian(z[np.ix_(occupied, occupied)])
    alpha = unitary_phase * amplitude.conjugate()
    return complex(alpha / abs(alpha)), float(abs(alpha))


def main():
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    rx, ry, ryt = operators(model)
    actions = {kind: reflection_action(model, kind) for kind in ("x", "y", "yt")}
    vertices = [
        ("v1_pi_0", (np.pi, 0.0), "x", "y", ry @ rx.conj()),
        ("v2_pi_pi", (np.pi, np.pi), "y", "x", rx @ ry.conj()),
        ("v3_2pi_pi", (2 * np.pi, np.pi), "x", "yt", ryt @ rx.conj()),
        ("v4_closure", (2 * np.pi, 2 * np.pi), "yt", "x", rx @ ryt.conj()),
    ]
    identity = np.eye(2 * model.n_modes)
    vertex_phases = compute(4, mixing=0.0)["vertex_phases"]
    rows = []
    for (label, point, incoming, outgoing, u), delta in zip(vertices, vertex_phases):
        h = model.bdg_hamiltonian(*point, pairing_twist="linear")
        alpha_in, magnitude_in = antiunitary_phase(h, actions[incoming])
        alpha_out, magnitude_out = antiunitary_phase(h, actions[outgoing])
        eigenvalue = alpha_out / alpha_in
        square = u @ u
        square_particle = square[: model.n_modes, : model.n_modes]
        rows.append({
            "vertex": label,
            "point": point,
            "incoming": incoming,
            "outgoing": outgoing,
            "unitary_symmetry_error": relative_error(u @ h @ u.conj().T, h),
            "u_square_plus_identity": float(np.max(np.abs(square + identity))),
            "u_square_minus_identity": float(np.max(np.abs(square - identity))),
            "u_square_is_negative_particle_identity": bool(
                np.allclose(square_particle, -np.eye(model.n_modes))
            ),
            "gaussian_parity": gaussian_parity(h),
            "alpha_in": [alpha_in.real, alpha_in.imag],
            "alpha_out": [alpha_out.real, alpha_out.imag],
            "alpha_magnitudes": [magnitude_in, magnitude_out],
            "u_ground_eigenvalue": [eigenvalue.real, eigenvalue.imag],
            "u_ground_eigenvalue_over_pi": float(np.angle(eigenvalue) / np.pi),
            "measured_vertex_delta_over_pi": float(delta / np.pi),
            "eigenvalue_relation_error": float(abs(eigenvalue - np.exp(2j * delta))),
        })
    payload = {
        "definition": "U=Theta_out Theta_in (no inverse); U|psi>=u|psi>, u=exp(2i delta)",
        "parameters": vars(PARAMETERS),
        "vertices": rows,
    }
    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(
            f"{row['vertex']}: sym_err={row['unitary_symmetry_error']:.2e}, "
            f"U^2+I={row['u_square_plus_identity']:.2e}, "
            f"parity={row['gaussian_parity']}, "
            f"arg(u)/pi={row['u_ground_eigenvalue_over_pi']:+.6f}, "
            f"delta/pi={row['measured_vertex_delta_over_pi']:+.6f}, "
            f"relation_err={row['eigenvalue_relation_error']:.2e}"
        )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
