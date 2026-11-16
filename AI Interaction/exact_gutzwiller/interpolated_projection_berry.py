"""Berry phase for exact linear interpolation into the spin-projected state.

|Psi_lambda> proportional to (1-lambda)|Psi_G> + lambda P_G|Psi_G>.
The endpoints are the unprojected Gaussian state (lambda=0) and the physical
one-parton-per-cell projected state (lambda=1).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from berry_phase import twist_loop
from gaussian_overlap import normalized_gaussian_overlap
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state
from single_occupancy_overlap import single_occupancy_overlap


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-leg", type=int, default=16)
    parser.add_argument("--lambda-values", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reference-mask", type=lambda value: int(value, 0), default=0xAAC8DA33)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lambdas = [float(value) for value in args.lambda_values.split(",")]
    if any(not 0.0 <= value <= 1.0 for value in lambdas):
        raise ValueError("Every interpolation parameter must lie between 0 and 1")
    output = args.output or ROOT / f"interpolated_projection_m-2_M{args.steps_per_leg}.json"
    # Do not resume checkpoints generated with the former complex-pairing path.
    checkpoint = output.with_suffix(".cosine.jsonl")
    model = TwoOrbitalModel(4, 4, ModelParameters(m=-2.0))
    vertices, points = twist_loop(args.steps_per_leg)
    states = [
        gaussian_bcs_state(model.bdg_hamiltonian(*point), reference_mask=args.reference_mask)
        for point in points
    ]
    print(
        f"Computing {len(points)} base links; minimum BdG gap={min(s.gap for s in states):.10f}; "
        f"maximum reference condition={max(s.reference_condition for s in states):.3e}",
        flush=True,
    )

    completed = {}
    if checkpoint.exists():
        with checkpoint.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    completed[record["initial_point"]] = record
        print(f"Resumed {len(completed)} base links from {checkpoint}", flush=True)

    base_links = []
    for index, state in enumerate(states):
        next_index = (index + 1) % len(states)
        if index in completed:
            record = completed[index]
        else:
            next_state = states[next_index]
            gaussian = normalized_gaussian_overlap(state.z, next_state.z)
            projected = single_occupancy_overlap(
                state.z, next_state.z, model.n_cells,
                reference_mask1=args.reference_mask,
                reference_mask2=args.reference_mask,
                workers=args.workers,
            )
            # The input Gaussian states are normalized here. If f_a=N_P(a)/N_G(a),
            # then <Psi_G(a)|P_G|Psi_G(b)> = O_P(a,b)*sqrt(f_a*f_b).
            projected_raw = projected["normalized_overlap"] * np.sqrt(
                projected["norm1_fraction"] * projected["norm2_fraction"]
            )
            record = {
                "initial_point": index,
                "final_point": next_index,
                "gaussian_overlap": {"real": gaussian.real, "imag": gaussian.imag},
                "projected_raw_overlap": {"real": projected_raw.real, "imag": projected_raw.imag},
                "projected_normalized_overlap": {
                    "real": projected["normalized_overlap"].real,
                    "imag": projected["normalized_overlap"].imag,
                },
                "norm1_fraction": projected["norm1_fraction"],
                "norm2_fraction": projected["norm2_fraction"],
                "elapsed_seconds": projected["elapsed_seconds"],
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            completed[index] = record
        base_links.append(record)
        print(f"Completed base link {index:03d}->{next_index:03d}", flush=True)

    results = []
    for lam in lambdas:
        overlaps = []
        norm_factors = []
        for record in base_links:
            gaussian = complex(**record["gaussian_overlap"])
            projected = complex(**record["projected_raw_overlap"])
            weight = lam * (2.0 - lam)
            numerator = (1.0 - lam) ** 2 * gaussian + weight * projected
            norm1 = (1.0 - lam) ** 2 + weight * record["norm1_fraction"]
            norm2 = (1.0 - lam) ** 2 + weight * record["norm2_fraction"]
            overlaps.append(numerator / np.sqrt(norm1 * norm2))
            norm_factors.append(norm1)
        phases = [float(np.angle(value)) for value in overlaps]
        total_phase = math.fsum(phases)
        berry_phase = float(np.angle(np.exp(1j * total_phase)))
        results.append({
            "lambda": lam,
            "berry_phase_radians": berry_phase,
            "berry_phase_over_pi": berry_phase / np.pi,
            "minimum_link_magnitude": float(min(abs(value) for value in overlaps)),
            "minimum_norm_fraction": float(min(norm_factors)),
            "total_unwrapped_link_phase": total_phase,
        })
        print(
            f"lambda={lam:.2f}: theta={berry_phase:+.12f} rad = "
            f"{berry_phase / np.pi:+.12f} pi; min|O|={min(abs(value) for value in overlaps):.10f}",
            flush=True,
        )

    payload = {
        "lattice": [4, 4],
        "parameters": vars(model.params),
        "interpolation": "Psi_lambda proportional to (1-lambda)Psi_G + lambda P_G Psi_G",
        "projection": "P_G=prod_i n_i(2-n_i), with n_i=n_ic+n_id",
        "steps_per_leg": args.steps_per_leg,
        "reference_mask": args.reference_mask,
        "vertices": vertices.tolist(),
        "minimum_bdg_gap": min(state.gap for state in states),
        "results": results,
        "base_links": base_links,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Result file: {output.resolve()}")


if __name__ == "__main__":
    main()
