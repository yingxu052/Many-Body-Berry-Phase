"""Exact double-occupancy-suppression Berry scan with batched g evaluation.

P(g)=product_i[1-(1-g)n_ic n_id]: g=1 is unprojected and g=0 is fully projected.
Each subset Pfaffian is evaluated once per link and reused for every g value.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from berry_phase import twist_loop
from finite_g_overlap import finite_g_overlaps
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-leg", type=int, default=16)
    parser.add_argument("--g-values", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--m", type=float, default=-2.0)
    parser.add_argument("--reference-mask", type=lambda value: int(value, 0), default=0x5057F515)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps_per_leg < 1 or args.workers < 1:
        raise ValueError("steps-per-leg and workers must be positive")
    g_values = [float(value) for value in args.g_values.split(",")]
    if any(not 0.0 <= value <= 1.0 for value in g_values):
        raise ValueError("Every g must lie between 0 and 1")
    output = args.output or ROOT / f"finite_g_double_occupancy_m{args.m:g}_cosine_M{args.steps_per_leg}.json"
    checkpoint = output.with_suffix(".jsonl")

    model = TwoOrbitalModel(4, 4, ModelParameters(m=args.m))
    vertices, points = twist_loop(args.steps_per_leg)
    states = [
        gaussian_bcs_state(model.bdg_hamiltonian(*point), reference_mask=args.reference_mask)
        for point in points
    ]
    print(
        f"Computing {len(points)} links for {len(g_values)} g values; "
        f"minimum BdG gap={min(state.gap for state in states):.10f}",
        flush=True,
    )

    completed = {}
    if checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    completed[record["initial_point"]] = record
        print(f"Resumed {len(completed)} links from {checkpoint}", flush=True)

    links = []
    for index, state in enumerate(states):
        next_index = (index + 1) % len(states)
        if index in completed:
            record = completed[index]
        else:
            overlaps = finite_g_overlaps(
                state.z, states[next_index].z, model.n_cells, g_values,
                reference_mask=args.reference_mask, workers=args.workers,
            )
            record = {
                "initial_point": index,
                "final_point": next_index,
                "phi_initial": points[index].tolist(),
                "phi_final": points[next_index].tolist(),
                "values": [{
                    "g": item["g"],
                    "normalized_overlap": {
                        "real": item["normalized_overlap"].real,
                        "imag": item["normalized_overlap"].imag,
                    },
                    "norm1_fraction": item["norm1_fraction"],
                    "norm2_fraction": item["norm2_fraction"],
                } for item in overlaps],
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            completed[index] = record
        links.append(record)
        print(f"Completed link {index:03d}->{next_index:03d}", flush=True)

    links.sort(key=lambda record: record["initial_point"])
    results = []
    for g_index, g in enumerate(g_values):
        overlaps = []
        norm_fractions = []
        for record in links:
            value = record["values"][g_index]
            if value["g"] != g:
                raise RuntimeError("Checkpoint g ordering is inconsistent")
            overlaps.append(complex(**value["normalized_overlap"]))
            norm_fractions.extend((value["norm1_fraction"], value["norm2_fraction"]))
        total_phase = math.fsum(float(np.angle(overlap)) for overlap in overlaps)
        phase = float(np.angle(np.exp(1j * total_phase)))
        result = {
            "g": g,
            "berry_phase_radians": phase,
            "berry_phase_over_pi": phase / np.pi,
            "minimum_link_magnitude": float(min(abs(overlap) for overlap in overlaps)),
            "minimum_norm_fraction": float(min(norm_fractions)),
            "total_unwrapped_link_phase": total_phase,
        }
        results.append(result)
        print(
            f"g={g:.2f}: theta={phase:+.12f} rad = {phase / np.pi:+.12f} pi; "
            f"min|O|={result['minimum_link_magnitude']:.10f}",
            flush=True,
        )

    payload = {
        "lattice": [4, 4],
        "parameters": vars(model.params),
        "pairing_twist": "cosine",
        "projection_convention": "P(g)=product_i[1-(1-g)n_ic n_id]; g=0 full, g=1 unprojected",
        "steps_per_leg": args.steps_per_leg,
        "vertices": vertices.tolist(),
        "reference_mask": args.reference_mask,
        "minimum_bdg_gap": min(state.gap for state in states),
        "results": results,
        "links": links,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Result file: {output.resolve()}")


if __name__ == "__main__":
    main()
