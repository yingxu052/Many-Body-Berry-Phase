"""Decompose complex-path Berry phases relative to the cosine reference path."""

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


def compute_path(steps_per_leg, pairing_twist):
    """Use one Slater reference to obtain a continuous gauge along the loop."""
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    points = twist_loop(steps_per_leg)
    states = [
        gaussian_bcs_state(
            model.bdg_hamiltonian(*point, pairing_twist=pairing_twist),
            reference_mask=REFERENCE_MASK,
        )
        for point in points
    ]
    overlaps = [
        normalized_gaussian_overlap(state.z, states[(index + 1) % len(states)].z)
        for index, state in enumerate(states)
    ]
    link_phases = np.angle(overlaps)
    leg_phases = np.array([
        math.fsum(link_phases[leg * steps_per_leg:(leg + 1) * steps_per_leg])
        for leg in range(4)
    ])
    return {
        "link_phases": link_phases,
        "leg_phases": leg_phases,
        "total_phase": float(math.fsum(leg_phases)),
        "minimum_link_magnitude": float(np.min(np.abs(overlaps))),
        "maximum_reference_condition": float(max(s.reference_condition for s in states)),
    }


def select_negative_topological_branch(path):
    """Move a possible pi branch ambiguity to the final closure link."""
    adjustment = 0
    if path["total_phase"] > 0.0:
        adjustment = -1
        path["link_phases"][-1] -= np.pi
        path["leg_phases"][-1] -= np.pi
        path["total_phase"] -= np.pi
    return adjustment


def main():
    convergence = []
    detailed = None
    for steps in (16, 32, 64, 128, 256):
        complex_path = compute_path(steps, "complex")
        cosine_path = compute_path(steps, "cosine")
        adjustment = select_negative_topological_branch(cosine_path)
        extra_legs = complex_path["leg_phases"] - cosine_path["leg_phases"]
        result = {
            "steps_per_leg": steps,
            "complex_leg_phases": complex_path["leg_phases"].tolist(),
            "cosine_reference_leg_phases": cosine_path["leg_phases"].tolist(),
            "reference_pi_adjustment_on_closure": adjustment,
            "extra_leg_phases": extra_legs.tolist(),
            "complex_total_phase": complex_path["total_phase"],
            "cosine_reference_total_phase": cosine_path["total_phase"],
            "extra_total_phase": float(math.fsum(extra_legs)),
            "complex_minimum_link_magnitude": complex_path["minimum_link_magnitude"],
            "cosine_minimum_link_magnitude": cosine_path["minimum_link_magnitude"],
            "complex_maximum_reference_condition": complex_path["maximum_reference_condition"],
            "cosine_maximum_reference_condition": cosine_path["maximum_reference_condition"],
        }
        convergence.append(result)
        print(
            f"M={steps:3d}: complex={complex_path['total_phase']:+.12f}, "
            f"reference={cosine_path['total_phase']:+.12f}, "
            f"extra={result['extra_total_phase']:+.12f}; "
            f"extra legs={np.array2string(extra_legs, precision=10)}",
            flush=True,
        )
        if steps == 256:
            detailed = {
                "result": result,
                "complex_link_phases": complex_path["link_phases"].tolist(),
                "cosine_link_phases": cosine_path["link_phases"].tolist(),
            }

    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps({
        "parameters": vars(PARAMETERS),
        "reference_mask": REFERENCE_MASK,
        "method": "normalized Gaussian overlaps in one common Slater-reference gauge",
        "definition": "extra = complex path - cosine reference path, leg by leg",
        "convergence": convergence,
        "detailed_m256": detailed,
    }, indent=2) + "\n")

    complex_links = detailed["complex_link_phases"]
    cosine_links = detailed["cosine_link_phases"]
    complex_cumulative = np.concatenate([[0.0], np.cumsum(complex_links)])
    cosine_cumulative = np.concatenate([[0.0], np.cumsum(cosine_links)])
    extra_cumulative = complex_cumulative - cosine_cumulative
    coordinate = np.linspace(0.0, 4.0, 4 * 256 + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    ax.plot(coordinate, complex_cumulative, lw=2.0, label="Complex pairing")
    ax.plot(coordinate, cosine_cumulative, lw=1.8, ls="--", label="Cosine reference")
    ax.plot(coordinate, extra_cumulative, lw=1.8, label="Additional contribution")
    for boundary in (1, 2, 3):
        ax.axvline(boundary, color="0.75", lw=0.8)
    ax.axhline(-np.pi / 2, color="0.35", lw=0.8, ls=":", label=r"$-\pi/2$")
    ax.set(xlabel="Twist leg", ylabel="Accumulated phase, rad", xlim=(0, 4), xticks=(0, 1, 2, 3, 4))
    ax.grid(alpha=0.16)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Accumulated Berry phase")

    ax = axes[1]
    extra_legs = np.asarray(detailed["result"]["extra_leg_phases"])
    positions = np.arange(4)
    bars = ax.bar(positions, extra_legs, color="#c55a11", width=0.62)
    for bar, value in zip(bars, extra_legs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.004 if value >= 0 else -0.004),
            f"{value:+.5f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9,
        )
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set(
        xlabel="Twist leg", ylabel=r"$\delta\theta_\ell=\theta_\ell^{complex}-\theta_\ell^{cos}$, rad",
        xticks=positions, xticklabels=("1", "2", "3", "4"),
        title=f"Additional contribution; total = {math.fsum(extra_legs):+.6f} rad",
    )
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    figure_path = Path(__file__).resolve().with_suffix(".png")
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    print(f"Saved {output}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
