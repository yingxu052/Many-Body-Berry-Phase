"""Cross-check the unprojected nu_M=2 Berry phase using two common references."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from berry_phase import twist_loop
from gaussian_overlap import normalized_gaussian_overlap
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state


ROOT = Path(__file__).resolve().parent
PARAMETERS = ModelParameters(m=-2.0)


def compute_loop(steps: int, reference_mask: int):
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    vertices, points = twist_loop(steps)
    states = [
        gaussian_bcs_state(model.bdg_hamiltonian(*point), reference_mask=reference_mask)
        for point in points
    ]
    links = []
    for index, state in enumerate(states):
        next_index = (index + 1) % len(states)
        overlap = normalized_gaussian_overlap(state.z, states[next_index].z)
        links.append({
            "initial_point": index,
            "final_point": next_index,
            "normalized_overlap": {"real": overlap.real, "imag": overlap.imag},
            "magnitude": abs(overlap),
            "phase": float(np.angle(overlap)),
        })
    total_phase = math.fsum(link["phase"] for link in links)
    berry_phase = float(np.angle(np.exp(1j * total_phase)))
    return {
        "steps_per_leg": steps,
        "reference_mask": reference_mask,
        "maximum_reference_condition": max(state.reference_condition for state in states),
        "minimum_bdg_gap": min(state.gap for state in states),
        "minimum_link_magnitude": min(link["magnitude"] for link in links),
        "total_unwrapped_link_phase": total_phase,
        "berry_phase_radians": berry_phase,
        "berry_phase_over_pi": berry_phase / np.pi,
        "vertices": vertices.tolist(),
        "links": links,
    }


def main() -> None:
    references = {
        "common_a": 0x5057F515,
        "common_b": 0xAAC8DA33,
        "common_c": 0x08ABB8DB,
    }
    summaries = []
    detailed = {}
    for steps in (2, 4, 8, 16, 32, 64):
        for label, mask in references.items():
            result = compute_loop(steps, mask)
            summaries.append({
                key: value for key, value in result.items()
                if key not in ("vertices", "links")
            })
            if steps == 64:
                detailed[label] = result
            print(
                f"M={steps:2d}, {label}: theta={result['berry_phase_radians']:+.12f} rad "
                f"= {result['berry_phase_over_pi']:+.12f} pi; "
                f"min|O|={result['minimum_link_magnitude']:.12f}; "
                f"max cond={result['maximum_reference_condition']:.3e}",
                flush=True,
            )
    payload = {
        "lattice": [4, 4],
        "parameters": vars(PARAMETERS),
        "projection": "none (unprojected fermionic Gaussian states)",
        "method": "normalized Gaussian overlaps with one common Slater reference per entire loop",
        "summaries": summaries,
        "detailed_m64": detailed,
    }
    output = ROOT / "unprojected_nu2_berry_check.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Result file: {output.resolve()}")


if __name__ == "__main__":
    main()
