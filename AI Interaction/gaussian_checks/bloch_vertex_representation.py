"""Bloch-spinor derivation of the four vertex U eigenvalues.

At v4=(2pi,2pi), the allowed momenta are periodic and U4 maps k to -k.
At each TRIM, h(k)=xi(k)sigma_z and the p-wave pairing vanishes. The
remaining Delta_s pairing makes the even-sector ground energy -|Delta_s|,
whereas the lowest odd-sector energy is -|xi|. An odd sector therefore has
lower energy exactly when |xi|>|Delta_s|. Its U4 eigenvalue follows from
V(k)=exp(i*k_y)*diag(i,-i), acting on creation operators.
"""

import json
from pathlib import Path

import numpy as np


M = -2.0
T = 1.0
DELTA_S = 0.5


def xi(kx, ky):
    return -2.0 * T * np.cos(kx) - 2.0 * T * np.cos(ky) - M


def self_inverse(kx, ky):
    return (
        np.allclose(np.exp(2j * kx), 1.0)
        and np.allclose(np.exp(2j * ky), 1.0)
    )


def allowed_momenta(size, boundary):
    # boundary=0: periodic; boundary=1: antiperiodic.
    return 2.0 * np.pi * (np.arange(size) + 0.5 * boundary) / size


def main():
    size = 4
    vertices = [
        ("v1", 1, 0),
        ("v2", 1, 1),
        ("v3", 0, 1),
        ("v4", 0, 0),
    ]
    rows = []
    for label, boundary_x, boundary_y in vertices:
        momenta = [
            (kx, ky)
            for kx in allowed_momenta(size, boundary_x)
            for ky in allowed_momenta(size, boundary_y)
        ]
        trim = [(kx, ky) for kx, ky in momenta if self_inverse(kx, ky)]
        row = {
            "vertex": label,
            "boundary_x": "antiperiodic" if boundary_x else "periodic",
            "boundary_y": "antiperiodic" if boundary_y else "periodic",
            "self_inverse_momentum_count": len(trim),
            "sector_eigenvalues": [],
        }
        if label != "v4":
            # Distinct +/-k sectors: even BCS ground state and eigenvalue +1.
            row["u"] = 1
        else:
            eigenvalues = []
            for kx, ky in trim:
                normal_xi = float(xi(kx, ky))
                odd_ground = abs(normal_xi) > abs(DELTA_S)
                if not odd_ground:
                    sector_u = 1.0 + 0.0j
                    occupied = "even BCS sector"
                else:
                    # h=xi*sigma_z: c has energy xi, d has energy -xi.
                    occupied = "c" if normal_xi < 0.0 else "d"
                    q = np.exp(1j * ky)
                    sector_u = q * 1j if occupied == "c" else -q * 1j
                eigenvalues.append(sector_u)
                row["sector_eigenvalues"].append({
                    "k_over_pi": [float(kx / np.pi), float(ky / np.pi)],
                    "xi": normal_xi,
                    "ground_parity": "odd" if odd_ground else "even",
                    "lowest_odd_spinor": occupied,
                    "translation_phase_exp_i_ky": [float(np.cos(ky)), float(np.sin(ky))],
                    "u_sector": [float(np.real(sector_u)), float(np.imag(sector_u))],
                })
            row["u"] = float(np.real(np.prod(eigenvalues)))
            row["odd_sector_count"] = sum(
                sector["ground_parity"] == "odd" for sector in row["sector_eigenvalues"]
            )
            row["product_imaginary_residual"] = float(np.imag(np.prod(eigenvalues)))
        rows.append(row)

    payload = {
        "parameters": {"t": T, "m": M, "delta_s": DELTA_S},
        "lattice": [size, size],
        "conventions": {
            "normal_h_at_TRIM": "h(k)=xi(k)*sigma_z",
            "xi": "-2*t*cos(kx)-2*t*cos(ky)-m",
            "U4_creation_action": "V(k)=exp(i*k_y)*diag(i,-i)",
            "even_sector_ground_energy_at_TRIM": "-abs(delta_s)",
            "odd_sector_ground_energy_at_TRIM": "-abs(xi)",
        },
        "vertices": rows,
        "u_product": float(np.prod([row["u"] for row in rows])),
    }
    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(f"{row['vertex']}: TRIM={row['self_inverse_momentum_count']}, u={row['u']:+.0f}")
        for sector in row["sector_eigenvalues"]:
            print("  ", sector)
    print(f"product(u)={payload['u_product']:+.0f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
