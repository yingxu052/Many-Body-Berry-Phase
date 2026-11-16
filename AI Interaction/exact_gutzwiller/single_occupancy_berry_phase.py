"""Exact Berry phase after projection to the one-parton-per-cell spin Hilbert space."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from berry_phase import twist_loop
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state
from single_occupancy_overlap import single_occupancy_overlap


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-leg", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--m", type=float, default=1.5)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--delta-p", type=float, default=1.0)
    parser.add_argument("--delta-s", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps_per_leg < 1:
        raise ValueError("steps-per-leg must be positive")
    params = ModelParameters(
        t=args.t, t1=args.t1, m=args.m, delta_p=args.delta_p, delta_s=args.delta_s,
    )
    output = args.output or ROOT / f"single_occupancy_berry_m{args.steps_per_leg}.json"
    # Do not resume checkpoints generated with the former complex-pairing path.
    checkpoint = output.with_suffix(".cosine.jsonl")
    model = TwoOrbitalModel(4, 4, params)
    vertices, points = twist_loop(args.steps_per_leg)
    states = [gaussian_bcs_state(model.bdg_hamiltonian(*point)) for point in points]
    print(
        f"Computing {len(points)} links in a {2**model.n_cells}-dimensional spin Hilbert space; "
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
        print(f"Resumed {len(completed)} completed links from {checkpoint}", flush=True)

    links = []
    for index, state in enumerate(states):
        next_index = (index + 1) % len(states)
        if index in completed:
            record = completed[index]
            overlap = complex(
                record["normalized_overlap"]["real"], record["normalized_overlap"]["imag"],
            )
        else:
            result = single_occupancy_overlap(
                state.z, states[next_index].z, model.n_cells,
                reference_mask1=state.reference_mask,
                reference_mask2=states[next_index].reference_mask,
                workers=args.workers,
            )
            overlap = result["normalized_overlap"]
            record = {
                "initial_point": index,
                "final_point": next_index,
                "phi_initial": points[index].tolist(),
                "phi_final": points[next_index].tolist(),
                "normalized_overlap": {"real": overlap.real, "imag": overlap.imag},
                "magnitude": abs(overlap),
                "phase": float(np.angle(overlap)),
                "norm1_fraction": result["norm1_fraction"],
                "norm2_fraction": result["norm2_fraction"],
                "counted_configurations": result["counted_configurations"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        links.append(record)
        print(
            f"link {index:02d}->{next_index:02d}: |O|={abs(overlap):.10f}, "
            f"phase={np.angle(overlap):+.10f}, "
            f"projected weight={record['norm1_fraction']:.8e}",
            flush=True,
        )

    links.sort(key=lambda link: link["initial_point"])
    if len(links) != len(points):
        raise RuntimeError("The loop is incomplete")
    total_phase = math.fsum(link["phase"] for link in links)
    berry_phase = float(np.angle(np.exp(1j * total_phase)))
    endpoint_error = float(np.linalg.norm(
        model.bdg_hamiltonian(2 * np.pi, 2 * np.pi) - model.bdg_hamiltonian(0.0, 0.0)
    ))
    sector_labels = ["(0,0)", "(pi,0)", "(pi,pi)", "(2pi,pi)"]
    sectors = []
    for sector, label in enumerate(sector_labels):
        index = sector * args.steps_per_leg
        sectors.append({
            "sector": label,
            "point_index": index,
            "twist": points[index].tolist(),
            "projected_norm_fraction": links[index]["norm1_fraction"],
            "bdg_gap": states[index].gap,
            "gaussian_parity": states[index].parity,
            "reference_mask": states[index].reference_mask,
        })

    payload = {
        "lattice": [4, 4],
        "parameters": vars(params),
        "projection": "P_G=prod_i n_i(2-n_i), with n_i=n_ic+n_id",
        "physical_hilbert_space_dimension": 2**model.n_cells,
        "steps_per_leg": args.steps_per_leg,
        "number_of_links": len(links),
        "vertices": vertices.tolist(),
        "minimum_bdg_gap": min(state.gap for state in states),
        "gaussian_parities": [state.parity for state in states],
        "hamiltonian_endpoint_closure_error": endpoint_error,
        "minimum_link_magnitude": min(link["magnitude"] for link in links),
        "minimum_projected_norm_fraction": min(link["norm1_fraction"] for link in links),
        "total_unwrapped_link_phase": total_phase,
        "berry_phase_radians": berry_phase,
        "berry_phase_over_pi": berry_phase / np.pi,
        "twist_sectors": sectors,
        "links": links,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Berry phase: {berry_phase:+.12f} rad = {berry_phase / np.pi:+.12f} pi")
    print(f"Minimum link magnitude: {payload['minimum_link_magnitude']:.10f}")
    print(f"Result file: {output.resolve()}")


if __name__ == "__main__":
    main()
