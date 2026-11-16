"""Berry phase for s_linear(phi) + i*eta*sin(phi) seam pairing."""

import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import PARAMETERS, twist_loop
from gaussian_overlap import normalized_gaussian_overlap
from model import TwoOrbitalModel, gaussian_bcs_state


REFERENCE_MASK = 0xAAC8DA33


def compute(eta, steps_per_leg):
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    points = twist_loop(steps_per_leg)
    states = [
        gaussian_bcs_state(
            model.bdg_hamiltonian(
                *point, pairing_twist="linear", pairing_imag=eta,
            ),
            reference_mask=REFERENCE_MASK,
        )
        for point in points
    ]
    overlaps = [
        normalized_gaussian_overlap(state.z, states[(index + 1) % len(states)].z)
        for index, state in enumerate(states)
    ]
    link_phases = np.angle(overlaps)
    leg_phases = [
        math.fsum(link_phases[leg * steps_per_leg:(leg + 1) * steps_per_leg])
        for leg in range(4)
    ]
    total = math.fsum(leg_phases)
    closure = np.max(np.abs(
        model.bdg_hamiltonian(
            2.0 * np.pi, 2.0 * np.pi, pairing_twist="linear", pairing_imag=eta,
        )
        - model.bdg_hamiltonian(0.0, 0.0, pairing_twist="linear", pairing_imag=eta)
    ))
    return {
        "eta": eta,
        "steps_per_leg": steps_per_leg,
        "leg_phases_radians": leg_phases,
        "berry_phase_radians": total,
        "berry_phase_over_pi": total / np.pi,
        "deviation_from_minus_pi_over_2": total + np.pi / 2.0,
        "minimum_link_magnitude": float(np.min(np.abs(overlaps))),
        "minimum_bdg_gap": float(min(state.gap for state in states)),
        "maximum_reference_condition": float(max(s.reference_condition for s in states)),
        "hamiltonian_closure_error": float(closure),
    }


def main():
    etas = [-0.2, -0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
    sweep = []
    for eta in etas:
        result = compute(eta, 128)
        sweep.append(result)
        print(
            f"eta={eta:+.2f}: theta={result['berry_phase_radians']:+.12f}, "
            f"theta/pi={result['berry_phase_over_pi']:+.12f}, "
            f"delta={result['deviation_from_minus_pi_over_2']:+.9e}, "
            f"gap={result['minimum_bdg_gap']:.9f}",
            flush=True,
        )

    convergence = []
    for eta in (-0.1, 0.1):
        for steps in (16, 32, 64, 128, 256):
            result = compute(eta, steps)
            convergence.append(result)
            print(
                f"convergence eta={eta:+.1f} M={steps:3d}: "
                f"theta={result['berry_phase_radians']:+.12f}, "
                f"delta={result['deviation_from_minus_pi_over_2']:+.9e}",
                flush=True,
            )

    small = [r for r in sweep if abs(r["eta"]) <= 0.05]
    cubic_fit = np.polyfit(
        [r["eta"] for r in small],
        [r["deviation_from_minus_pi_over_2"] for r in small],
        3,
    )
    print("Cubic fit for delta(eta), |eta|<=0.05:", cubic_fit)

    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps({
        "parameters": vars(PARAMETERS),
        "pairing_scale": "s_linear(phi) + i*eta*sin(phi)",
        "reference_mask": REFERENCE_MASK,
        "cubic_fit_delta_for_small_eta_descending_powers": cubic_fit.tolist(),
        "sweep_m128": sweep,
        "mesh_convergence": convergence,
    }, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].plot(etas, [r["berry_phase_over_pi"] for r in sweep], "o-", lw=1.7, ms=5)
    axes[0].axhline(-0.5, color="0.3", ls=":", lw=1.0)
    axes[0].set(xlabel=r"Imaginary strength $\eta$", ylabel=r"Berry phase $\theta_B/\pi$")
    axes[0].grid(alpha=0.2)
    axes[0].set_title("Berry phase versus imaginary pairing component")

    axes[1].plot(etas, [r["deviation_from_minus_pi_over_2"] for r in sweep], "o-", color="#c55a11", lw=1.7, ms=5)
    axes[1].axhline(0.0, color="0.3", lw=0.8)
    axes[1].set(xlabel=r"Imaginary strength $\eta$", ylabel=r"$\theta_B+\pi/2$, rad")
    axes[1].grid(alpha=0.2)
    axes[1].set_title("Deviation from the quantized value")
    fig.tight_layout()
    figure_path = Path(__file__).resolve().with_suffix(".png")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
