"""First-order Gaussian-overlap response to real and imaginary pairing deformations."""

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
STEP = 1.0e-5


def response(steps_per_leg, deformation):
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    points = twist_loop(steps_per_leg)

    def hamiltonian(point, parameter):
        if deformation == "imaginary":
            return model.bdg_hamiltonian(
                *point, pairing_twist="linear", pairing_imag=parameter,
            )
        if deformation == "real_profile":
            linear = model.bdg_hamiltonian(*point, pairing_twist="linear")
            cosine = model.bdg_hamiltonian(*point, pairing_twist="cosine")
            center = 0.5
            return (1.0 - center - parameter) * linear + (center + parameter) * cosine
        raise ValueError(deformation)

    def make_states(parameter):
        return [
            gaussian_bcs_state(hamiltonian(point, parameter), reference_mask=REFERENCE_MASK)
            for point in points
        ]

    center_states = make_states(0.0)
    plus_states = make_states(STEP)
    minus_states = make_states(-STEP)
    derivatives = [
        (plus.z - minus.z) / (2.0 * STEP)
        for plus, minus in zip(plus_states, minus_states)
    ]

    analytic_links = []
    direct_links = []
    identity = np.eye(model.n_modes, dtype=np.complex128)
    for index in range(len(points)):
        next_index = (index + 1) % len(points)
        z1 = center_states[index].z
        z2 = center_states[next_index].z
        dz1 = derivatives[index]
        dz2 = derivatives[next_index]
        cross = identity + z1.conj().T @ z2
        inverse_cross = np.linalg.inv(cross)
        log_derivative = np.trace(
            inverse_cross @ (dz1.conj().T @ z2 + z1.conj().T @ dz2)
        )
        analytic_links.append(0.5 * log_derivative.imag)

        plus_overlap = normalized_gaussian_overlap(plus_states[index].z, plus_states[next_index].z)
        minus_overlap = normalized_gaussian_overlap(minus_states[index].z, minus_states[next_index].z)
        direct_links.append(
            (np.angle(plus_overlap) - np.angle(minus_overlap)) / (2.0 * STEP)
        )

    analytic_links = np.asarray(analytic_links)
    direct_links = np.asarray(direct_links)
    leg_coefficients = [
        math.fsum(analytic_links[leg * steps_per_leg:(leg + 1) * steps_per_leg])
        for leg in range(4)
    ]
    return {
        "deformation": deformation,
        "steps_per_leg": steps_per_leg,
        "finite_difference_step": STEP,
        "analytic_leg_coefficients": leg_coefficients,
        "analytic_total_coefficient": math.fsum(leg_coefficients),
        "analytic_link_coefficients": analytic_links.tolist(),
        "direct_leg_coefficients": [
            math.fsum(direct_links[leg * steps_per_leg:(leg + 1) * steps_per_leg])
            for leg in range(4)
        ],
        "maximum_absolute_link_coefficient": float(np.max(np.abs(analytic_links))),
        "maximum_formula_direct_link_error": float(np.max(np.abs(analytic_links - direct_links))),
        "rms_link_coefficient": float(np.sqrt(np.mean(analytic_links ** 2))),
    }


def main():
    results = []
    for deformation in ("imaginary", "real_profile"):
        for steps in (32, 64, 128):
            result = response(steps, deformation)
            results.append(result)
            summary = {k: v for k, v in result.items() if k != "analytic_link_coefficients"}
            print(json.dumps(summary, indent=2), flush=True)

    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps({
        "formula": "q = (1/2) Im tr[(I+Z1^dag Z2)^(-1) (dZ1^dag Z2 + Z1^dag dZ2)]",
        "real_profile": "H(lambda)=(1-lambda) H_linear + lambda H_cosine, differentiated at lambda=1/2",
        "imaginary_profile": "s_linear(phi) + i eta sin(phi), differentiated at eta=0",
        "results": results,
    }, indent=2) + "\n")

    finest = {
        deformation: next(
            r for r in results
            if r["deformation"] == deformation and r["steps_per_leg"] == 128
        )
        for deformation in ("imaginary", "real_profile")
    }
    coordinate = np.linspace(0.0, 4.0, 4 * 128 + 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for deformation, label, color in (
        ("imaginary", "Imaginary deformation", "#2563a5"),
        ("real_profile", "Real linear-cosine interpolation", "#c55a11"),
    ):
        cumulative = np.concatenate([
            [0.0], np.cumsum(finest[deformation]["analytic_link_coefficients"])
        ])
        ax.plot(coordinate, cumulative, lw=1.8, label=label, color=color)
    for boundary in (1, 2, 3):
        ax.axvline(boundary, color="0.75", lw=0.8)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set(
        xlabel="Twist leg", ylabel="Cumulative first-order response",
        xlim=(0, 4), xticks=(0, 1, 2, 3, 4),
        title="Integrated Gaussian-overlap response",
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_path = Path(__file__).resolve().with_suffix(".png")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
