"""Aggregate and plot exact Gaussian versus parton-projected overlap responses."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
STEPS = (4, 8, 16, 32, 64, 128)


def load_results():
    results = {}
    for steps in STEPS:
        path = HERE / f"projected_pairing_response_m{steps}_h0.0001.json"
        payload = json.loads(path.read_text())
        for result in payload["results"]:
            results[(result["deformation"], steps)] = result
    return results


def main():
    results = load_results()
    imaginary = [results[("imaginary", steps)] for steps in STEPS]
    real = [results[("real_profile", steps)] for steps in STEPS]
    finest_imaginary = imaginary[-1]
    finest_real = real[-1]

    convergence = [
        {
            "steps_per_leg": steps,
            "gaussian_total": result["gaussian_total_coefficient"],
            "projected_total": result["projected_total_coefficient"],
            "minimum_projected_link_magnitude": result["minimum_projected_link_magnitude"],
        }
        for steps, result in zip(STEPS, imaginary)
    ]
    richardson = (
        finest_imaginary["projected_total_coefficient"]
        + (
            finest_imaginary["projected_total_coefficient"]
            - imaginary[-2]["projected_total_coefficient"]
        ) / 3.0
    )
    real_leg_errors = [abs(value) for value in finest_real["projected_leg_coefficients"]]

    summary = {
        "projection": "one parton per unit cell",
        "derivative_step": 1.0e-4,
        "imaginary_convergence": convergence,
        "imaginary_richardson_extrapolation_O_h2": richardson,
        "finest_imaginary_leg_coefficients": {
            "gaussian": finest_imaginary["gaussian_leg_coefficients"],
            "projected": finest_imaginary["projected_leg_coefficients"],
        },
        "finest_real_leg_coefficients": {
            "gaussian": finest_real["gaussian_leg_coefficients"],
            "projected": finest_real["projected_leg_coefficients"],
        },
        "finest_real_total": {
            "gaussian": finest_real["gaussian_total_coefficient"],
            "projected": finest_real["projected_total_coefficient"],
        },
        "finest_real_maximum_projected_gaussian_link_difference":
            finest_real["maximum_projected_gaussian_link_difference"],
        "finest_imaginary_maximum_projected_gaussian_link_difference":
            finest_imaginary["maximum_projected_gaussian_link_difference"],
        "minimum_projected_norm_fraction":
            finest_imaginary["minimum_projected_norm_fraction"],
    }
    summary_path = HERE / "projected_pairing_response_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("Imaginary-response convergence")
    print(" steps/leg       Gaussian       projected   min |projected link|")
    for row in convergence:
        print(
            f"{row['steps_per_leg']:10d}  {row['gaussian_total']:13.10f}  "
            f"{row['projected_total']:13.10f}  {row['minimum_projected_link_magnitude']:18.12f}"
        )
    print(f"O(h^2) Richardson extrapolation: {richardson:.12f}")
    print("\nImaginary leg coefficients at 128 steps/leg")
    print(" leg       Gaussian       projected")
    for leg, (gaussian, projected) in enumerate(zip(
        finest_imaginary["gaussian_leg_coefficients"],
        finest_imaginary["projected_leg_coefficients"],
    ), start=1):
        print(f" {leg:3d}  {gaussian:13.10f}  {projected:13.10f}")
    print("\nReal-profile checks at 128 steps/leg")
    print(f"projected leg coefficients: {finest_real['projected_leg_coefficients']}")
    print(f"projected total: {finest_real['projected_total_coefficient']:.16e}")
    print(f"maximum absolute projected leg coefficient: {max(real_leg_errors):.3e}")
    print(
        "maximum |projected-Gaussian| link difference: "
        f"{finest_real['maximum_projected_gaussian_link_difference']:.3e}"
    )
    print(
        "minimum projected link magnitude (real, imaginary): "
        f"{finest_real['minimum_projected_link_magnitude']:.12f}, "
        f"{finest_imaginary['minimum_projected_link_magnitude']:.12f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    ax = axes[0]
    ax.plot(
        STEPS, [row["gaussian_total"] for row in convergence],
        "o-", color="#2563a5", label="Gaussian",
    )
    ax.plot(
        STEPS, [row["projected_total"] for row in convergence],
        "s-", color="#c55a11", label="Parton projected",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(STEPS, [str(steps) for steps in STEPS])
    ax.set(
        xlabel="Steps per twist leg",
        ylabel="Total imaginary-response coefficient",
        title="(a) Mesh convergence",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    coordinate = np.linspace(0.0, 4.0, 4 * 128 + 1)
    for ax, deformation, title in (
        (axes[1], "imaginary", "(b) Imaginary deformation"),
        (axes[2], "real_profile", "(c) Real profile deformation"),
    ):
        finest = results[(deformation, 128)]
        for key, label, color, linestyle in (
            ("gaussian_link_coefficients", "Gaussian", "#2563a5", "-"),
            ("projected_link_coefficients", "Parton projected", "#c55a11", "--"),
        ):
            cumulative = np.concatenate([[0.0], np.cumsum(finest[key])])
            ax.plot(
                coordinate, cumulative, linestyle=linestyle, lw=1.8,
                color=color, label=label,
            )
        for boundary in (1, 2, 3):
            ax.axvline(boundary, color="0.78", lw=0.8)
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.set(
            xlabel="Twist leg",
            ylabel="Cumulative first-order response",
            xlim=(0, 4), xticks=(0, 1, 2, 3, 4), title=title,
        )
        ax.grid(alpha=0.18)
        ax.legend(frameon=False)

    fig.tight_layout()
    figure_path = HERE / "projected_pairing_response.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print(f"\nSaved {summary_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
