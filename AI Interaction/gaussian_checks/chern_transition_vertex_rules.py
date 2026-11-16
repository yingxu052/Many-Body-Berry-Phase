"""Evaluate vertex-gap and eigenvalue-ratio rules at every Chern transition."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from model import ModelParameters, TwoOrbitalModel


def main():
    transitions = [
        (-4.5, -4.875, -4.375, 0, 1),
        (-3.5, -3.625, -3.375, 1, 2),
        (-0.5, -0.625, -0.375, 2, 0),
        (0.5, 0.375, 0.625, 0, -2),
        (3.5, 3.375, 3.625, -2, -1),
        (4.5, 4.375, 4.625, -1, 0),
    ]
    exact_path = Path(__file__).resolve().with_name("exact_vertex_eigenvalues_2x2.json")
    exact_rows = {row["m"]: row for row in json.loads(exact_path.read_text())["rows"]}
    base = dict(t=1.0, t1=1.0, delta_p=1.0, delta_s=0.5)
    points = [(np.pi, 0), (np.pi, np.pi), (2 * np.pi, np.pi), (0, 0)]
    output_rows = []
    for critical, left_mass, right_mass, nu_left, nu_right in transitions:
        model = TwoOrbitalModel(4, 4, ModelParameters(m=critical, **base))
        gaps = []
        for point in points:
            energies = np.linalg.eigvalsh(model.bdg_hamiltonian(*point, pairing_twist="linear"))
            n = model.n_modes
            gaps.append(float(min(energies[n], -energies[n - 1])))
        left = exact_rows[left_mass]["vertices"][3]
        right = exact_rows[right_mass]["vertices"][3]
        u_left = complex(*left["u"])
        u_right = complex(*right["u"])
        ratio = u_right / u_left
        delta_gamma = 0.5 * np.angle(ratio)
        row = {
            "critical_m": critical,
            "left_m": left_mass,
            "right_m": right_mass,
            "nu_left": nu_left,
            "nu_right": nu_right,
            "delta_nu": nu_right - nu_left,
            "critical_vertex_gaps_4x4": gaps,
            "u4_left_2x2": [u_left.real, u_left.imag],
            "u4_right_2x2": [u_right.real, u_right.imag],
            "u4_ratio_2x2": [ratio.real, ratio.imag],
            "delta_gamma_from_ratio_mod_pi": float(delta_gamma % np.pi),
            "delta_gamma_over_pi_from_ratio": float(delta_gamma / np.pi),
            "delta_nu_over_4_mod_1": float(((nu_right - nu_left) / 4) % 1),
        }
        output_rows.append(row)
        print(
            f"m_c={critical:+.1f}, Delta_nu={nu_right-nu_left:+d}: "
            f"gaps={np.array2string(np.asarray(gaps), precision=2)}, "
            f"u4R/u4L={ratio:+.3f}, "
            f"Delta_gamma/pi={delta_gamma/np.pi:+.3f} (mod 1), "
            f"Delta_nu/4={(nu_right-nu_left)/4:+.3f} (mod 1)"
        )
    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps({"transitions": output_rows}, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
