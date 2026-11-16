"""Combine and validate checkpointed links for a complete projected Berry loop."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = [ROOT / "berry_result_m8_part1.jsonl", ROOT / "berry_result_m8_part2.jsonl"]
METADATA = ROOT / "berry_result_m8_part2.json"
OUTPUT = ROOT / "berry_result_m8.json"


def main() -> None:
    links = []
    for checkpoint in CHECKPOINTS:
        with checkpoint.open(encoding="utf-8") as handle:
            links.extend(json.loads(line) for line in handle if line.strip())
    links.sort(key=lambda link: link["initial_point"])

    count = len(links)
    assert count == 32, f"Expected 32 links, found {count}"
    assert [link["initial_point"] for link in links] == list(range(count))
    assert len({link["initial_point"] for link in links}) == count
    for index, link in enumerate(links):
        assert link["final_point"] == (index + 1) % count
        assert link["magnitude"] > 0.0

    with METADATA.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["steps_per_leg"] == 8

    total_phase = math.fsum(link["phase"] for link in links)
    wrapped_phase = (total_phase + math.pi) % (2.0 * math.pi) - math.pi
    result = {
        "lattice": metadata["lattice"],
        "parameters": metadata["parameters"],
        "steps_per_leg": 8,
        "number_of_links": count,
        "vertices": metadata["vertices"],
        "minimum_bdg_gap": metadata["minimum_bdg_gap"],
        "parities": metadata["parities"],
        "hamiltonian_endpoint_closure_error": metadata["hamiltonian_endpoint_closure_error"],
        "minimum_link_magnitude": min(link["magnitude"] for link in links),
        "total_unwrapped_link_phase": total_phase,
        "berry_phase_radians": wrapped_phase,
        "berry_phase_over_pi": wrapped_phase / math.pi,
        "checkpoint_files": [str(path) for path in CHECKPOINTS],
        "links": links,
    }
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    print(f"Validated {count} unique links with closure 31 -> 0")
    print(f"Minimum link magnitude: {result['minimum_link_magnitude']:.12f}")
    print(f"Berry phase: {wrapped_phase:.12f} rad")
    print(f"Berry phase / pi: {result['berry_phase_over_pi']:.12f}")
    print(f"Combined result: {OUTPUT}")


if __name__ == "__main__":
    main()
