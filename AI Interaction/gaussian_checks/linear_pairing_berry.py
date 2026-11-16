"""Berry phase for piecewise-linear seam-pairing interpolation."""

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import (
    PARAMETERS, appendix_b_overlap, positive_blocks, twist_loop,
)
from gaussian_overlap import normalized_gaussian_overlap
from model import TwoOrbitalModel, gaussian_bcs_state


def compute(lx, ly, steps_per_leg, pairing_twist):
    model = TwoOrbitalModel(lx, ly, PARAMETERS)
    points = twist_loop(steps_per_leg)
    first = gaussian_bcs_state(
        model.bdg_hamiltonian(*points[0], pairing_twist=pairing_twist),
        reference_mask=0xAAC8DA33 if (lx, ly) == (4, 4) else None,
    )
    states = [first] + [
        gaussian_bcs_state(
            model.bdg_hamiltonian(*point, pairing_twist=pairing_twist),
            reference_mask=first.reference_mask,
        )
        for point in points[1:]
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
    wrapped = float(np.angle(np.exp(1j * total)))
    closure = np.max(np.abs(
        model.bdg_hamiltonian(2.0 * np.pi, 2.0 * np.pi, pairing_twist=pairing_twist)
        - model.bdg_hamiltonian(0.0, 0.0, pairing_twist=pairing_twist)
    ))
    return {
        "lx": lx,
        "ly": ly,
        "steps_per_leg": steps_per_leg,
        "pairing_twist": pairing_twist,
        "reference_mask": first.reference_mask,
        "leg_phases_radians": leg_phases,
        "unwrapped_phase_radians": total,
        "berry_phase_radians": wrapped,
        "berry_phase_over_pi": wrapped / np.pi,
        "phase_mod_pi_over_pi": (total / np.pi) % 1.0,
        "minimum_link_magnitude": float(np.min(np.abs(overlaps))),
        "minimum_bdg_gap": float(min(state.gap for state in states)),
        "maximum_reference_condition": float(max(s.reference_condition for s in states)),
        "hamiltonian_closure_error": float(closure),
    }


def compute_appendix(lx, ly, steps_per_leg, pairing_twist):
    """Reference-independent check, with phases compared modulo pi."""
    model = TwoOrbitalModel(lx, ly, PARAMETERS)
    points = twist_loop(steps_per_leg)
    blocks = [
        positive_blocks(model.bdg_hamiltonian(*point, pairing_twist=pairing_twist))
        for point in points
    ]
    overlaps = [
        appendix_b_overlap(*blocks[index][:2], *blocks[(index + 1) % len(blocks)][:2])
        for index in range(len(blocks))
    ]
    total = math.fsum(float(np.angle(overlap)) for overlap in overlaps)
    wrapped = float(np.angle(np.exp(1j * total)))
    return {
        "lx": lx,
        "ly": ly,
        "steps_per_leg": steps_per_leg,
        "pairing_twist": pairing_twist,
        "method": "Appendix-B overlap; phase modulo pi",
        "berry_phase_radians": wrapped,
        "berry_phase_over_pi": wrapped / np.pi,
        "phase_mod_pi_over_pi": (total / np.pi) % 1.0,
        "minimum_link_magnitude": float(np.min(np.abs(overlaps))),
        "minimum_bdg_gap": float(min(block[2] for block in blocks)),
    }


def main():
    mesh_results = []
    for steps in (4, 8, 16, 32, 64, 128, 256):
        result = compute(4, 4, steps, "linear")
        mesh_results.append(result)
        print(
            f"linear 4x4 M={steps:3d}: theta={result['berry_phase_radians']:+.12f}, "
            f"theta/pi={result['berry_phase_over_pi']:+.12f}, "
            f"gap={result['minimum_bdg_gap']:.9f}, "
            f"min|O|={result['minimum_link_magnitude']:.9f}",
            flush=True,
        )

    size_results = []
    for size in range(4, 10):
        result = compute_appendix(size, size, 32, "linear")
        size_results.append(result)
        print(
            f"linear {size}x{size} M=32: theta={result['berry_phase_radians']:+.12f}, "
            f"theta/pi={result['berry_phase_over_pi']:+.12f}, "
            f"gap={result['minimum_bdg_gap']:.9f}",
            flush=True,
        )

    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps({
        "parameters": vars(PARAMETERS),
        "pairing_scale": "+1 to -1 linearly for phi in [0,pi], then -1 to +1 for phi in [pi,2pi]",
        "normal_twist": "exp(i phi), unchanged",
        "mesh_convergence": mesh_results,
        "size_scan": size_results,
    }, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
