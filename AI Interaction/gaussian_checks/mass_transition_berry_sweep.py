"""Quick unprojected-Gaussian test of Berry-phase changes across Chern transitions."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gaussian_checks"))
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import (
    appendix_b_overlap,
    positive_blocks,
    twist_loop,
)
from majorana_chern import occupied_chern
from model import ModelParameters, TwoOrbitalModel


BASE = dict(t=1.0, t1=1.0, delta_p=1.0, delta_s=0.5)


def compute_loop(lx: int, ly: int, steps: int, mass: float, pairing: str):
    params = ModelParameters(m=float(mass), **BASE)
    model = TwoOrbitalModel(lx, ly, params)
    blocks = [
        positive_blocks(model.bdg_hamiltonian(*point, pairing_twist=pairing))
        for point in twist_loop(steps)
    ]
    overlaps = [
        appendix_b_overlap(*blocks[index][:2], *blocks[(index + 1) % len(blocks)][:2])
        for index in range(len(blocks))
    ]
    total = math.fsum(float(np.angle(overlap)) for overlap in overlaps)
    return {
        "phase_mod_pi": float(total % np.pi),
        "minimum_link_magnitude": float(np.min(np.abs(overlaps))),
        "minimum_twist_gap": float(min(block[2] for block in blocks)),
    }


def main():
    lx = ly = 4
    steps = 32
    # A quarter-spacing grid shifted by 0.125 avoids all critical half-integer masses.
    masses = sorted(set(np.linspace(-4.875, 4.875, 40).tolist() + [-2.0]))
    rows = []
    for mass in masses:
        params = ModelParameters(m=float(mass), **BASE)
        occupied_c, zero_twist_gap = occupied_chern(params, grid=40)
        row = {
            "m": float(mass),
            # Sign convention: m=-2 is the manuscript's nu_M=2 phase.
            "nu_M": int(-round(occupied_c)),
            "occupied_bdg_chern": float(occupied_c),
            "zero_twist_gap": float(zero_twist_gap),
        }
        for pairing in ("complex", "cosine"):
            row[pairing] = compute_loop(lx, ly, steps, mass, pairing)
        rows.append(row)
        print(
            f"m={mass:+.3f} nu_M={row['nu_M']:+d} "
            f"complex={row['complex']['phase_mod_pi']:.9f} "
            f"cosine={row['cosine']['phase_mod_pi']:.9f} "
            f"gap={min(row[p]['minimum_twist_gap'] for p in ('complex', 'cosine')):.6f}",
            flush=True,
        )

    # Track the continuous branch modulo pi, then measure changes between the
    # nearest sampled gapped phases on opposite sides of every Chern jump.
    for pairing in ("complex", "cosine"):
        unwrapped = np.unwrap(
            [row[pairing]["phase_mod_pi"] for row in rows], period=np.pi
        )
        for row, value in zip(rows, unwrapped):
            row[pairing]["unwrapped_mod_pi_branch"] = float(value)

    jumps = []
    for left, right in zip(rows[:-1], rows[1:]):
        if left["nu_M"] == right["nu_M"]:
            continue
        jump = {
            "m_left": left["m"],
            "m_right": right["m"],
            "critical_m_midpoint": 0.5 * (left["m"] + right["m"]),
            "nu_left": left["nu_M"],
            "nu_right": right["nu_M"],
            "delta_nu": right["nu_M"] - left["nu_M"],
        }
        for pairing in ("complex", "cosine"):
            gamma_left = left[pairing]["unwrapped_mod_pi_branch"]
            gamma_right = right[pairing]["unwrapped_mod_pi_branch"]
            jump[f"{pairing}_left"] = gamma_left
            jump[f"{pairing}_right"] = gamma_right
            jump[f"{pairing}_delta"] = gamma_right - gamma_left
            jump[f"{pairing}_delta_over_pi"] = (gamma_right - gamma_left) / np.pi
        jumps.append(jump)

    output = Path(__file__).resolve().with_suffix(".json")
    payload = {
        "lattice": [lx, ly],
        "steps_per_leg": steps,
        "parameters": BASE,
        "method": "Appendix-B Gaussian overlaps; phases and changes are defined modulo pi",
        "nu_sign_convention": "nu_M=-occupied_BdG_Chern so that m=-2 has nu_M=2",
        "rows": rows,
        "chern_transition_jumps": jumps,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")

    fig, (ax, ax_nu) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for pairing, marker in (("complex", "o"), ("cosine", "s")):
        reliable = [row for row in rows if row[pairing]["minimum_link_magnitude"] > 0.9]
        ax.plot(
            [row["m"] for row in reliable],
            [row[pairing]["phase_mod_pi"] for row in reliable],
            marker=marker,
            ms=4,
            lw=1.4,
            label=f"{pairing} pairing path",
        )
    ax.set_ylabel(r"Berry phase $\gamma$ mod $\pi$")
    ax.set_yticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    ax.set_yticklabels(["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    ax.set_ylim(-0.08, np.pi + 0.08)
    ax.grid(alpha=0.25)
    ax.legend()

    ax_nu.step(masses, [row["nu_M"] for row in rows], where="mid", color="black")
    ax_nu.set_ylabel(r"$\nu_M$")
    ax_nu.set_xlabel(r"mass parameter $m$")
    ax_nu.set_yticks(sorted(set(row["nu_M"] for row in rows)))
    ax_nu.grid(alpha=0.25)
    for jump in jumps:
        for axis in (ax, ax_nu):
            axis.axvline(jump["critical_m_midpoint"], color="gray", ls=":", lw=1)
    fig.suptitle(
        r"Unprojected Gaussian state, $4\times4$, 32 steps per leg"
        "\nPoints with minimum link magnitude below 0.9 are omitted"
    )
    fig.tight_layout()
    figure = Path(__file__).resolve().with_suffix(".png")
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    print("\nChanges between nearest samples across Chern transitions:")
    for jump in jumps:
        print(
            f"m={jump['m_left']:+.3f}->{jump['m_right']:+.3f}, "
            f"Delta_nu={jump['delta_nu']:+d}, "
            f"Delta_complex/pi={jump['complex_delta_over_pi']:+.9f}, "
            f"Delta_cosine/pi={jump['cosine_delta_over_pi']:+.9f}"
        )
    print(f"Saved {output}")
    print(f"Saved {figure}")


if __name__ == "__main__":
    main()
