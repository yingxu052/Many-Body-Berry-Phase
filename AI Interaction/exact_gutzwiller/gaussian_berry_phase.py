"""Compute unprojected Gaussian Berry phases and compare them with full projection."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from berry_phase import twist_loop
from gaussian_overlap import normalized_gaussian_overlap
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state


ROOT = Path(__file__).resolve().parent
PARAMETERS = ModelParameters(t=1.0, t1=1.0, m=1.5, delta_p=1.0, delta_s=0.5)


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def compute_loop(steps: int) -> dict:
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    vertices, points = twist_loop(steps)
    states = [gaussian_bcs_state(model.bdg_hamiltonian(*point)) for point in points]
    references = {state.reference_mask for state in states}
    if len(references) != 1:
        raise ValueError(f"The Gaussian states do not share one Slater reference: {references}")

    links = []
    for index, state in enumerate(states):
        next_index = (index + 1) % len(states)
        overlap = normalized_gaussian_overlap(state.z, states[next_index].z)
        links.append({
            "initial_point": index,
            "final_point": next_index,
            "phi_initial": points[index].tolist(),
            "phi_final": points[next_index].tolist(),
            "normalized_overlap": {"real": overlap.real, "imag": overlap.imag},
            "magnitude": abs(overlap),
            "phase": float(np.angle(overlap)),
        })

    total_phase = math.fsum(link["phase"] for link in links)
    berry_phase = float(np.angle(np.exp(1j * total_phase)))
    endpoint_error = np.linalg.norm(
        model.bdg_hamiltonian(2 * np.pi, 2 * np.pi) - model.bdg_hamiltonian(0.0, 0.0)
    )
    return {
        "lattice": [4, 4],
        "parameters": vars(PARAMETERS),
        "projection": "none (unprojected fermionic Gaussian state)",
        "steps_per_leg": steps,
        "number_of_links": len(links),
        "vertices": vertices.tolist(),
        "minimum_bdg_gap": min(state.gap for state in states),
        "parities": [state.parity for state in states],
        "reference_mask": next(iter(references)),
        "hamiltonian_endpoint_closure_error": float(endpoint_error),
        "minimum_link_magnitude": min(link["magnitude"] for link in links),
        "total_unwrapped_link_phase": total_phase,
        "berry_phase_radians": berry_phase,
        "berry_phase_over_pi": berry_phase / np.pi,
        "links": links,
    }


def main() -> None:
    comparison = []
    for steps in (2, 4, 8, 16):
        gaussian = compute_loop(steps)
        gaussian_path = ROOT / f"gaussian_berry_result_m{steps}.json"
        write_json(gaussian_path, gaussian)
        projected = read_json(ROOT / f"berry_result_m{steps}.json")
        phase_difference = float(np.angle(np.exp(1j * (
            projected["berry_phase_radians"] - gaussian["berry_phase_radians"]
        ))))
        comparison.append({
            "steps_per_leg": steps,
            "unprojected_gaussian_phase_radians": gaussian["berry_phase_radians"],
            "unprojected_gaussian_phase_over_pi": gaussian["berry_phase_over_pi"],
            "fully_projected_phase_radians": projected["berry_phase_radians"],
            "fully_projected_phase_over_pi": projected["berry_phase_over_pi"],
            "projected_minus_unprojected_radians": phase_difference,
            "unprojected_minimum_link_magnitude": gaussian["minimum_link_magnitude"],
        })
        print(
            f"M={steps:2d}: Gaussian={gaussian['berry_phase_radians']:+.12f} rad "
            f"({gaussian['berry_phase_over_pi']:+.12f} pi), "
            f"projected={projected['berry_phase_radians']:+.12f} rad, "
            f"difference={phase_difference:+.12f} rad"
        )

    comparison_path = ROOT / "projection_comparison.json"
    write_json(comparison_path, {
        "lattice": [4, 4],
        "parameters": vars(PARAMETERS),
        "projection_convention": "full: product_i (1-n_ic n_id); no finite Gutzwiller g",
        "meshes": comparison,
    })
    print(f"Comparison file: {comparison_path}")


if __name__ == "__main__":
    main()
