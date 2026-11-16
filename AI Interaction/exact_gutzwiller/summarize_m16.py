"""Assemble the M=16 loop and summarize projected Berry-phase mesh convergence."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "berry_result_m16.json"
CHECKPOINT = ROOT / "berry_result_m16.jsonl"
CONVERGENCE = ROOT / "berry_mesh_convergence.json"


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    metadata = read_json(RESULT)
    with CHECKPOINT.open(encoding="utf-8") as handle:
        links = [json.loads(line) for line in handle if line.strip()]
    links.sort(key=lambda link: link["initial_point"])

    steps = 16
    count = 4 * steps
    assert metadata["steps_per_leg"] == steps
    assert len(links) == count, f"Expected {count} links, found {len(links)}"
    assert [link["initial_point"] for link in links] == list(range(count))
    assert len({link["initial_point"] for link in links}) == count
    for index, link in enumerate(links):
        assert link["final_point"] == (index + 1) % count
        assert link["magnitude"] > 0.0

    total_phase = math.fsum(link["phase"] for link in links)
    wrapped_phase = (total_phase + math.pi) % (2.0 * math.pi) - math.pi
    result = {
        "lattice": metadata["lattice"],
        "parameters": metadata["parameters"],
        "steps_per_leg": steps,
        "number_of_links": count,
        "vertices": metadata["vertices"],
        "minimum_bdg_gap": metadata["minimum_bdg_gap"],
        "parities": metadata["parities"],
        "hamiltonian_endpoint_closure_error": metadata["hamiltonian_endpoint_closure_error"],
        "minimum_link_magnitude": min(link["magnitude"] for link in links),
        "total_unwrapped_link_phase": total_phase,
        "berry_phase_radians": wrapped_phase,
        "berry_phase_over_pi": wrapped_phase / math.pi,
        "checkpoint_file": str(CHECKPOINT),
        "links": links,
    }
    with RESULT.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    meshes = [read_json(ROOT / f"berry_result_m{mesh}.json") for mesh in (2, 4, 8)] + [result]
    convergence_entries = []
    previous = None
    for mesh in meshes:
        phase = mesh["berry_phase_radians"]
        convergence_entries.append({
            "steps_per_leg": mesh["steps_per_leg"],
            "number_of_links": mesh["number_of_links"],
            "berry_phase_radians": phase,
            "berry_phase_over_pi": mesh["berry_phase_over_pi"],
            "change_from_previous_mesh_radians": None if previous is None else phase - previous,
            "minimum_link_magnitude": mesh["minimum_link_magnitude"],
        })
        previous = phase

    # For a second-order discrete Wilson loop, theta(M)=theta_inf+A/M^2+... .
    richardson = result["berry_phase_radians"] + (result["berry_phase_radians"] - meshes[-2]["berry_phase_radians"]) / 3.0
    convergence = {
        "meshes": convergence_entries,
        "second_order_richardson_estimate_radians": richardson,
        "second_order_richardson_estimate_over_pi": richardson / math.pi,
        "note": "The Richardson value is an extrapolation, not an independent exact-enumeration result.",
    }
    with CONVERGENCE.open("w", encoding="utf-8") as handle:
        json.dump(convergence, handle, indent=2)
        handle.write("\n")

    print(f"Validated {count} unique links with closure 63 -> 0")
    print(f"Minimum link magnitude: {result['minimum_link_magnitude']:.12f}")
    print(f"Berry phase: {wrapped_phase:.12f} rad")
    print(f"Berry phase / pi: {result['berry_phase_over_pi']:.12f}")
    print(f"Change from M=8: {convergence_entries[-1]['change_from_previous_mesh_radians']:.12f} rad")
    print(f"Second-order extrapolation: {richardson:.12f} rad")
    print(f"Combined result: {RESULT}")
    print(f"Convergence summary: {CONVERGENCE}")


if __name__ == "__main__":
    main()
