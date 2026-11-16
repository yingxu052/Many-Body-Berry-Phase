"""Leg-resolved response of the Berry phase to imaginary seam pairing."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INPUT = Path(__file__).resolve().with_name("linear_imaginary_pairing_berry.json")
OUTPUT = Path(__file__).resolve().with_suffix(".json")
FIGURE = Path(__file__).resolve().with_suffix(".png")


def main():
    payload = json.loads(INPUT.read_text())
    sweep = payload["sweep_m128"]
    baseline = next(r for r in sweep if r["eta"] == 0.0)
    base_legs = np.asarray(baseline["leg_phases_radians"])

    small = [r for r in sweep if abs(r["eta"]) <= 0.05]
    etas = np.asarray([r["eta"] for r in small])
    deviations = np.asarray([
        np.asarray(r["leg_phases_radians"]) - base_legs for r in small
    ])
    # Cubic fit for each leg; coefficient order is eta^3, eta^2, eta, constant.
    fits = np.asarray([np.polyfit(etas, deviations[:, leg], 3) for leg in range(4)])
    linear_coefficients = fits[:, 2]

    all_etas = np.asarray([r["eta"] for r in sweep])
    all_deviations = np.asarray([
        np.asarray(r["leg_phases_radians"]) - base_legs for r in sweep
    ])

    convergence_input = payload["mesh_convergence"]
    convergence = []
    for steps in sorted({r["steps_per_leg"] for r in convergence_input}):
        minus = next(
            r for r in convergence_input
            if r["eta"] == -0.1 and r["steps_per_leg"] == steps
        )
        plus = next(
            r for r in convergence_input
            if r["eta"] == 0.1 and r["steps_per_leg"] == steps
        )
        derivative = (
            np.asarray(plus["leg_phases_radians"])
            - np.asarray(minus["leg_phases_radians"])
        ) / 0.2
        convergence.append({
            "steps_per_leg": steps,
            "symmetric_leg_derivatives_at_eta_0": derivative.tolist(),
            "total_derivative": float(np.sum(derivative)),
        })
        print(
            f"M={steps:3d}: leg derivatives={np.array2string(derivative, precision=10)}, "
            f"total={np.sum(derivative):+.12f}"
        )

    print("Small-eta linear coefficients:", linear_coefficients)
    print("Sum:", np.sum(linear_coefficients))
    print("Cubic coefficients:", fits[:, 0])

    result = {
        "definition": "A_leg = partial theta_leg / partial eta at eta=0",
        "pairing_scale": payload["pairing_scale"],
        "linear_coefficients": linear_coefficients.tolist(),
        "quadratic_coefficients": fits[:, 1].tolist(),
        "cubic_coefficients": fits[:, 0].tolist(),
        "constant_coefficients": fits[:, 3].tolist(),
        "sum_of_linear_coefficients": float(np.sum(linear_coefficients)),
        "mesh_convergence": convergence,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")

    colors = ("#2563a5", "#c55a11", "#548235", "#7030a0")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    for leg, color in enumerate(colors):
        ax.plot(
            all_etas, all_deviations[:, leg], "o-", ms=4, lw=1.5,
            color=color, label=f"Leg {leg + 1}",
        )
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.axvline(0.0, color="0.75", lw=0.8)
    ax.set(
        xlabel=r"Imaginary strength $\eta$",
        ylabel=r"$\delta\theta_\ell(\eta)=\theta_\ell(\eta)-\theta_\ell(0)$, rad",
        title="Leg-resolved additional phase",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)

    ax = axes[1]
    positions = np.arange(4)
    bars = ax.bar(positions, linear_coefficients, color=colors, width=0.62)
    for bar, value in zip(bars, linear_coefficients):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value - 0.0007 if value < 0 else value + 0.0007,
            f"{value:+.6f}", ha="center", va="top" if value < 0 else "bottom",
            fontsize=9,
        )
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set(
        xlabel="Twist leg", ylabel=r"$A_\ell=\partial\theta_\ell/\partial\eta|_0$",
        xticks=positions, xticklabels=("1", "2", "3", "4"),
        title=f"Linear response; sum = {np.sum(linear_coefficients):+.7f} rad",
    )
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=220)
    plt.close(fig)
    print(f"Saved {OUTPUT}")
    print(f"Saved {FIGURE}")


if __name__ == "__main__":
    main()
