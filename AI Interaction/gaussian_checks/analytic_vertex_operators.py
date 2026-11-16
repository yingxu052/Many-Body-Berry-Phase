"""Analytic reflection-conjugation gauges and their unitary vertex products."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gaussian_checks"))
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import PARAMETERS
from model import TwoOrbitalModel


def nambu_unitary(model, mapping, orbital):
    n = model.n_modes
    particle = np.zeros((n, n), dtype=complex)
    for y in range(model.ly):
        for x in range(model.lx):
            xp, yp = mapping(x, y)
            for source_orbital in range(2):
                source = model.mode(x, y, source_orbital)
                for target_orbital in range(2):
                    target = model.mode(xp, yp, target_orbital)
                    particle[target, source] = orbital[target_orbital, source_orbital]
    return np.block([
        [particle, np.zeros_like(particle)],
        [np.zeros_like(particle), particle.conj()],
    ])


def operators(model):
    identity = np.eye(2)
    orbital_y = np.diag([1j, -1j])
    rx = nambu_unitary(
        model, lambda x, y: ((1 - x) % model.lx, y), identity
    )
    ry = nambu_unitary(
        model, lambda x, y: (x, (1 - y) % model.ly), orbital_y
    )
    ry_translated = nambu_unitary(
        model, lambda x, y: ((x + 1) % model.lx, (1 - y) % model.ly), orbital_y
    )
    return rx, ry, ry_translated


def relative_error(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-15))


def antiunitary_error(u, h):
    return relative_error(u @ h.conj() @ u.conj().T, h)


def mixed_hamiltonian(model, phi_x, phi_y):
    linear = model.bdg_hamiltonian(phi_x, phi_y, pairing_twist="linear")
    cosine = model.bdg_hamiltonian(phi_x, phi_y, pairing_twist="cosine")
    return 0.5 * (linear + cosine)


def main():
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    rx, ry, ryt = operators(model)
    n = model.n_modes
    identity = np.eye(2 * n)
    samples = [
        ("leg_1_x", (0.37, 0.0), rx),
        ("leg_2_y", (np.pi, 0.83), ry),
        ("leg_3_x", (2 * np.pi - 0.37, np.pi), rx),
        ("leg_4_y_translated", (2 * np.pi, 2 * np.pi - 0.83), ryt),
    ]
    checks = []
    for label, point, u in samples:
        h = mixed_hamiltonian(model, *point)
        checks.append({
            "leg": label,
            "point": point,
            "antiunitary_symmetry_error": antiunitary_error(u, h),
            "theta_squared_minus_identity": float(np.max(np.abs(u @ u.conj() - identity))),
        })

    # Theta_x=R_x K, Theta_y=R_y K. Since Theta_x^{-1}=K R_x^dagger,
    # Theta_y Theta_x^{-1}=R_y R_x^dagger is an ordinary unitary.
    products = {
        "vertex_1_pi_0_theta_y_theta_x_inverse": ry @ rx.conj().T,
        "vertex_2_pi_pi_theta_x_theta_y_inverse": rx @ ry.conj().T,
        "vertex_3_2pi_pi_theta_y_translated_theta_x_inverse": ryt @ rx.conj().T,
        "vertex_4_closure_theta_x_theta_y_translated_inverse": rx @ ryt.conj().T,
    }
    product_rows = []
    for label, u in products.items():
        particle = u[:n, :n]
        off_diagonal = float(np.max(np.abs(particle - np.diag(np.diag(particle)))))
        product_rows.append({
            "product": label,
            "unitary_error": float(np.max(np.abs(u @ u.conj().T - identity))),
            "particle_block_off_diagonal_max": off_diagonal,
            "is_onsite_large_gauge": bool(off_diagonal < 1e-12),
            "is_spatial_permutation_with_orbital_phases": bool(
                np.allclose(np.abs(particle).sum(axis=0), 1)
                and np.allclose(np.abs(particle).sum(axis=1), 1)
                and np.all((np.abs(particle) < 1e-12) | (np.abs(np.abs(particle) - 1) < 1e-12))
            ),
        })
    ordered = list(products.values())
    cycle_product = ordered[3] @ ordered[2] @ ordered[1] @ ordered[0]

    payload = {
        "definitions": {
            "theta_x": "R_x K, R_x:(x,y)->(1-x,y), orbital diag(1,1)",
            "theta_y": "R_y K, R_y:(x,y)->(x,1-y), orbital diag(i,-i)",
            "theta_y_last_leg": "T_x R_y K, T_x:(x,y)->(x+1,y)",
        },
        "antiunitary_checks": checks,
        "vertex_products": product_rows,
        "cycle_product_minus_identity": float(np.max(np.abs(cycle_product - identity))),
        "conclusion": (
            "The real-gauge vertex products are spatial rotation/reflection/translation "
            "unitaries, not onsite large-gauge transformations. Their ordered cycle "
            "product is the identity. Their numerical phases alone therefore do not "
            "identify the adiabatic twist holonomy with a flux Wilson commutator."
        ),
    }
    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for row in checks:
        print(row)
    for row in product_rows:
        print(row)
    print(payload["conclusion"])
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
