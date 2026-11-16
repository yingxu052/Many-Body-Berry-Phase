"""Scan vertex gaps, parity, and U eigenvalues through the mass phase diagram."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gaussian_checks"))
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from analytic_vertex_operators import operators
from complex_twist_berry_convergence import PARAMETERS
from majorana_chern import occupied_chern
from model import ModelParameters, TwoOrbitalModel, gaussian_parity
from verify_vertex_phase_algebra import antiunitary_phase, reflection_action


def main():
    masses = [-4.875, -4.625, -4.375, -4.125, -3.875, -3.625,
              -3.375, -3.0, -2.0, -1.0, -0.625, -0.375, -0.125,
              0.125, 0.375, 0.625, 1.0, 2.0, 3.0, 3.375,
              3.625, 3.875, 4.125, 4.375, 4.625, 4.875]
    base = dict(t=1.0, t1=1.0, delta_p=1.0, delta_s=0.5)
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    rx, ry, ryt = operators(model)
    actions = {kind: reflection_action(model, kind) for kind in ("x", "y", "yt")}
    vertices = [
        ("v1", (np.pi, 0.0), "x", "y"),
        ("v2", (np.pi, np.pi), "y", "x"),
        ("v3", (2 * np.pi, np.pi), "x", "yt"),
        ("v4", (2 * np.pi, 2 * np.pi), "yt", "x"),
    ]
    rows = []
    for mass in masses:
        params = ModelParameters(m=mass, **base)
        model = TwoOrbitalModel(4, 4, params)
        chern, zero_gap = occupied_chern(params, grid=40)
        row = {"m": mass, "nu_M": int(-round(chern)), "zero_twist_gap": zero_gap,
               "vertices": []}
        for label, point, incoming, outgoing in vertices:
            h = model.bdg_hamiltonian(*point, pairing_twist="linear")
            energies = np.linalg.eigvalsh(h)
            n = model.n_modes
            gap = float(min(energies[n], -energies[n - 1]))
            parity = gaussian_parity(h)
            entry = {"vertex": label, "gap": gap, "parity": parity}
            try:
                alpha_in, mag_in = antiunitary_phase(h, actions[incoming])
                alpha_out, mag_out = antiunitary_phase(h, actions[outgoing])
                u = alpha_out / alpha_in
                entry.update({
                    "u": [u.real, u.imag],
                    "arg_u_over_pi": float(np.angle(u) / np.pi),
                    "alpha_magnitudes": [mag_in, mag_out],
                })
            except Exception as exc:
                entry["error"] = str(exc)
            row["vertices"].append(entry)
        rows.append(row)
        values = []
        for entry in row["vertices"]:
            values.append(
                f"{entry['vertex']}:g={entry['gap']:.3g},p={entry['parity']},"
                f"u/pi={entry.get('arg_u_over_pi', float('nan')):+.3f}"
            )
        print(f"m={mass:+.3f}, nu={row['nu_M']:+d}: " + "; ".join(values), flush=True)
    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
