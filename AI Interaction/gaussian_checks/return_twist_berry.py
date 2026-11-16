"""Compare the original 2pi twist loop with a pi-to-0 return loop."""

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import PARAMETERS
from gaussian_overlap import normalized_gaussian_overlap
from model import TwoOrbitalModel, gaussian_bcs_state


REFERENCE_MASK = 0xAAC8DA33
VERTICES = {
    "original_2pi": np.array([
        [0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi],
        [2.0 * np.pi, np.pi], [2.0 * np.pi, 2.0 * np.pi],
    ]),
    "return_to_zero": np.array([
        [0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi],
        [0.0, np.pi], [0.0, 0.0],
    ]),
}


def path_points(vertices, steps_per_leg):
    return np.array([
        (1.0 - step / steps_per_leg) * vertices[leg]
        + (step / steps_per_leg) * vertices[leg + 1]
        for leg in range(4)
        for step in range(steps_per_leg)
    ])


def compute(path_name, pairing_twist, eta, steps_per_leg):
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    points = path_points(VERTICES[path_name], steps_per_leg)
    states = [
        gaussian_bcs_state(
            model.bdg_hamiltonian(
                *point, pairing_twist=pairing_twist, pairing_imag=eta,
            ),
            reference_mask=REFERENCE_MASK,
        )
        for point in points
    ]
    overlaps = [
        normalized_gaussian_overlap(state.z, states[(index + 1) % len(states)].z)
        for index, state in enumerate(states)
    ]
    phases = np.angle(overlaps)
    legs = [
        math.fsum(phases[leg * steps_per_leg:(leg + 1) * steps_per_leg])
        for leg in range(4)
    ]
    total = math.fsum(legs)
    return {
        "path": path_name,
        "pairing_twist": pairing_twist,
        "eta": eta,
        "steps_per_leg": steps_per_leg,
        "leg_phases_radians": legs,
        "berry_phase_radians": total,
        "berry_phase_over_pi": total / np.pi,
        "phase_mod_pi_over_pi": (total / np.pi) % 1.0,
        "minimum_link_magnitude": float(np.min(np.abs(overlaps))),
        "minimum_bdg_gap": float(min(s.gap for s in states)),
    }


def main():
    cases = [
        ("cosine", 0.0),
        ("linear", 0.0),
        ("linear", 0.1),
        ("complex", 0.0),
    ]
    results = []
    for steps in (32, 64, 128, 256):
        for pairing, eta in cases:
            for path_name in VERTICES:
                result = compute(path_name, pairing, eta, steps)
                results.append(result)
                print(
                    f"M={steps:3d} {path_name:14s} {pairing:7s} eta={eta:.1f}: "
                    f"theta={result['berry_phase_radians']:+.12f}, "
                    f"theta/pi={result['berry_phase_over_pi']:+.12f}, "
                    f"gap={result['minimum_bdg_gap']:.9f}",
                    flush=True,
                )

    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps({
        "parameters": vars(PARAMETERS),
        "vertices": {key: value.tolist() for key, value in VERTICES.items()},
        "results": results,
    }, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
