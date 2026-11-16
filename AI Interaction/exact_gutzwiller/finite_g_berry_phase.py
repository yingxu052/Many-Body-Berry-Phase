"""Scan the exact projected Berry phase from no projection (g=1) to full (g=0)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from berry_phase import twist_loop
from finite_g_overlap import finite_g_overlap
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-leg", type=int, default=16)
    parser.add_argument("--g-values", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    g_values = [float(value) for value in args.g_values.split(",")]
    output = args.output or ROOT / f"finite_g_berry_scan_m{args.steps_per_leg}.json"
    # Do not resume checkpoints generated with the former complex-pairing path.
    checkpoint = output.with_suffix(".cosine.jsonl")
    model = TwoOrbitalModel(4, 4, ModelParameters(m=1.5))
    vertices, points = twist_loop(args.steps_per_leg)
    states = [gaussian_bcs_state(model.bdg_hamiltonian(*point)) for point in points]
    references = {state.reference_mask for state in states}
    assert len(references) == 1
    reference = next(iter(references))

    completed = {}
    if checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    completed[(record["g"], record["initial_point"])] = record

    results = []
    for g in g_values:
        links = []
        for index, state in enumerate(states):
            key = (g, index)
            if key in completed:
                record = completed[key]
                links.append(record)
                continue
            next_index = (index + 1) % len(states)
            overlap_result = finite_g_overlap(
                state.z, states[next_index].z, model.n_cells, g,
                reference_mask=reference, workers=args.workers,
            )
            overlap = overlap_result["normalized_overlap"]
            record = {
                "g": g,
                "initial_point": index,
                "final_point": next_index,
                "normalized_overlap": {"real": overlap.real, "imag": overlap.imag},
                "magnitude": abs(overlap),
                "phase": float(np.angle(overlap)),
                "norm1_fraction": overlap_result["norm1_fraction"],
                "norm2_fraction": overlap_result["norm2_fraction"],
                "elapsed_seconds": overlap_result["elapsed_seconds"],
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            completed[key] = record
            links.append(record)

        links.sort(key=lambda link: link["initial_point"])
        assert len(links) == len(states)
        assert [link["initial_point"] for link in links] == list(range(len(states)))
        total_phase = math.fsum(link["phase"] for link in links)
        berry_phase = float(np.angle(np.exp(1j * total_phase)))
        result = {
            "g": g,
            "number_of_links": len(links),
            "berry_phase_radians": berry_phase,
            "berry_phase_over_pi": berry_phase / np.pi,
            "minimum_link_magnitude": min(link["magnitude"] for link in links),
            "minimum_norm_fraction": min(
                min(link["norm1_fraction"], link["norm2_fraction"]) for link in links
            ),
            "links": links,
        }
        results.append(result)
        print(
            f"g={g:.2f}: theta={berry_phase:+.12f} rad, "
            f"theta/pi={berry_phase / np.pi:+.12f}, "
            f"min|O|={result['minimum_link_magnitude']:.10f}",
            flush=True,
        )

    endpoint_checks = {}
    if 0.0 in g_values:
        exact_full = read_json(ROOT / f"berry_result_m{args.steps_per_leg}.json")
        computed = next(result["berry_phase_radians"] for result in results if result["g"] == 0.0)
        endpoint_checks["g0_difference_from_base3_enumeration_radians"] = computed - exact_full["berry_phase_radians"]
    if 1.0 in g_values:
        exact_gaussian = read_json(ROOT / f"gaussian_berry_result_m{args.steps_per_leg}.json")
        computed = next(result["berry_phase_radians"] for result in results if result["g"] == 1.0)
        endpoint_checks["g1_difference_from_gaussian_radians"] = computed - exact_gaussian["berry_phase_radians"]

    payload = {
        "lattice": [4, 4],
        "parameters": vars(model.params),
        "projection_convention": "P(g)=product_i[1-(1-g)n_ic n_id]",
        "steps_per_leg": args.steps_per_leg,
        "vertices": vertices.tolist(),
        "minimum_bdg_gap": min(state.gap for state in states),
        "reference_mask": reference,
        "endpoint_checks": endpoint_checks,
        "results": results,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Endpoint checks: {endpoint_checks}")
    print(f"Result file: {output.resolve()}")


if __name__ == "__main__":
    main()
