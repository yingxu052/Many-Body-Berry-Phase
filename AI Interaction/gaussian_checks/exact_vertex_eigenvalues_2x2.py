"""Exact-Fock-space cross-check of vertex U eigenvalues, including odd parity."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gaussian_checks"))
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from analytic_vertex_operators import operators
from majorana_chern import occupied_chern
from model import ModelParameters, TwoOrbitalModel, gaussian_parity


def fermion_operators(n):
    dimension = 1 << n
    creation = []
    annihilation = []
    for mode in range(n):
        c = np.zeros((dimension, dimension), dtype=complex)
        for state in range(dimension):
            if state & (1 << mode):
                continue
            target = state | (1 << mode)
            sign = (-1) ** ((state & ((1 << mode) - 1)).bit_count())
            c[target, state] = sign
        creation.append(c)
        annihilation.append(c.conj().T)
    return creation, annihilation


def many_body_hamiltonian(h_bdg):
    n = h_bdg.shape[0] // 2
    h = h_bdg[:n, :n]
    delta = h_bdg[:n, n:]
    c, a = fermion_operators(n)
    result = np.zeros((1 << n, 1 << n), dtype=complex)
    for i in range(n):
        for j in range(n):
            result += h[i, j] * (c[i] @ a[j])
            result += 0.5 * delta[i, j] * (c[i] @ c[j])
            result -= 0.5 * delta[i, j].conjugate() * (a[i] @ a[j])
    return result, c


def many_body_unitary(particle, creation):
    """Lift c_i^dagger -> sum_j conj(P_ij) c_j^dagger, with U|0>=|0>."""
    n = particle.shape[0]
    dimension = 1 << n
    transformed = [
        sum((particle[i, j].conjugate() * creation[j] for j in range(n)),
            start=np.zeros_like(creation[0]))
        for i in range(n)
    ]
    result = np.zeros((dimension, dimension), dtype=complex)
    for state in range(dimension):
        vector = np.zeros(dimension, dtype=complex)
        vector[0] = 1.0
        # |S>=c_0^dagger c_1^dagger ... |0>, so build from right to left.
        for mode in range(n - 1, -1, -1):
            if state & (1 << mode):
                vector = transformed[mode] @ vector
        result[:, state] = vector
    return result


def main():
    masses = [-4.875, -4.375, -4.125, -3.625, -3.375, -3.0, -2.0,
              -1.0, -0.625, -0.375, 0.375, 0.625, 1.0, 2.0, 3.0,
              3.375, 3.625, 4.125, 4.375, 4.625, 4.875]
    base = dict(t=1.0, t1=1.0, delta_p=1.0, delta_s=0.5)
    model = TwoOrbitalModel(2, 2, ModelParameters(m=-2.0, **base))
    rx, ry, ryt = operators(model)
    vertex_data = [
        ("v1", (np.pi, 0.0), ry @ rx.conj()),
        ("v2", (np.pi, np.pi), rx @ ry.conj()),
        ("v3", (2 * np.pi, np.pi), ryt @ rx.conj()),
        ("v4", (2 * np.pi, 2 * np.pi), rx @ ryt.conj()),
    ]
    unitaries = {}
    for label, _, u in vertex_data:
        unitaries[label] = many_body_unitary(
            u[:model.n_modes, :model.n_modes], fermion_operators(model.n_modes)[0]
        )
    parity = np.diag([(-1) ** state.bit_count() for state in range(1 << model.n_modes)])
    rows = []
    for mass in masses:
        params = ModelParameters(m=mass, **base)
        model = TwoOrbitalModel(2, 2, params)
        chern, _ = occupied_chern(params, grid=40)
        row = {"m": mass, "nu_M": int(-round(chern)), "vertices": []}
        for label, point, _ in vertex_data:
            h_bdg = model.bdg_hamiltonian(*point, pairing_twist="linear")
            h_many, _ = many_body_hamiltonian(h_bdg)
            energies, states = np.linalg.eigh(h_many)
            ground = states[:, 0]
            u = complex(ground.conj() @ unitaries[label] @ ground)
            p = complex(ground.conj() @ parity @ ground)
            entry = {
                "vertex": label,
                "gaussian_parity": gaussian_parity(h_bdg),
                "exact_parity": p.real,
                "many_body_gap": float(energies[1] - energies[0]),
                "u": [u.real, u.imag],
                "arg_u_over_pi": float(np.angle(u) / np.pi),
                "u_unit_error": float(abs(abs(u) - 1.0)),
                "square_relation_error": float(np.max(np.abs(
                    unitaries[label] @ unitaries[label] - parity
                ))),
            }
            row["vertices"].append(entry)
        rows.append(row)
        values = [
            f"{v['vertex']}:p={v['exact_parity']:+.0f},"
            f"u/pi={v['arg_u_over_pi']:+.3f},gap={v['many_body_gap']:.2e}"
            for v in row["vertices"]
        ]
        print(f"m={mass:+.3f},nu={row['nu_M']:+d}: " + "; ".join(values), flush=True)
    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps({"lattice": [2, 2], "rows": rows}, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
