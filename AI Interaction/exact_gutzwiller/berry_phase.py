"""Compute the fully Gutzwiller-projected Berry phase around the manuscript's twist loop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state
from overlap import projected_overlap


def twist_loop(steps_per_leg: int):
    vertices = np.array(
        [[0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi], [2 * np.pi, np.pi], [2 * np.pi, 2 * np.pi]]
    )
    points = []
    for leg in range(4):
        for step in range(steps_per_leg):
            fraction = step / steps_per_leg
            points.append((1.0 - fraction) * vertices[leg] + fraction * vertices[leg + 1])
    return vertices, np.array(points)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-leg", type=int, default=2)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--m", type=float, default=1.5)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--delta-p", type=float, default=1.0)
    parser.add_argument("--delta-s", type=float, default=0.5)
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--link-start", type=int, default=0)
    parser.add_argument("--link-stop", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("berry_result.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps_per_leg < 1:
        raise ValueError("steps-per-leg must be positive")
    params = ModelParameters(
        t=args.t, t1=args.t1, m=args.m, delta_p=args.delta_p, delta_s=args.delta_s,
    )
    model = TwoOrbitalModel(4, 4, params)
    vertices, points = twist_loop(args.steps_per_leg)
    states = []

    print(f"Scanning {len(points)} twist points on the 4 x 4 lattice")
    for index, (phi_x, phi_y) in enumerate(points):
        state = gaussian_bcs_state(model.bdg_hamiltonian(phi_x, phi_y))
        states.append(state)
        print(
            f"  point {index:02d}: Phi=({phi_x:.6f},{phi_y:.6f}), gap={state.gap:.8f}, "
            f"parity={state.parity}, reference={state.reference_mask}, "
            f"cond={state.reference_condition:.3e}"
        )

    gaps = [state.gap for state in states]
    parities = [state.parity for state in states]
    endpoint_error = np.linalg.norm(
        model.bdg_hamiltonian(2 * np.pi, 2 * np.pi) - model.bdg_hamiltonian(0.0, 0.0)
    )
    print(f"Minimum BdG gap on the loop: {min(gaps):.8f}")
    print(f"Hamiltonian endpoint closure error: {endpoint_error:.3e}")
    if args.diagnose_only:
        return

    link_start = args.link_start
    link_stop = len(points) if args.link_stop < 0 else args.link_stop
    if not (0 <= link_start < link_stop <= len(points)):
        raise ValueError(f"Invalid link range [{link_start}, {link_stop}) for {len(points)} links")

    # Do not resume checkpoints generated with the former complex-pairing path.
    checkpoint_path = args.output.with_suffix(".cosine.jsonl")
    links = []
    if checkpoint_path.exists():
        for line in checkpoint_path.read_text().splitlines():
            link = json.loads(line)
            value = link["normalized_overlap"]
            link["normalized_overlap"] = complex(value["real"], value["imag"])
            if link_start <= link["initial_point"] < link_stop:
                links.append(link)
        if links:
            print(f"Resumed {len(links)} completed links from {checkpoint_path}")
    completed = {link["initial_point"] for link in links}

    for index in range(link_start, link_stop):
        if index in completed:
            continue
        next_index = (index + 1) % len(points)
        result = projected_overlap(
            states[index].z, states[next_index].z, model.n_cells,
            parity=0,
            reference_mask1=states[index].reference_mask,
            reference_mask2=states[next_index].reference_mask,
            workers=args.workers,
        )
        overlap = result["normalized_overlap"]
        link = {
            "initial_point": index,
            "final_point": next_index,
            "phi_initial": points[index].tolist(),
            "phi_final": points[next_index].tolist(),
            "normalized_overlap": overlap,
            "magnitude": abs(overlap),
            "phase": float(np.angle(overlap)),
            "elapsed_seconds": result["elapsed_seconds"],
        }
        links.append(link)
        checkpoint_link = dict(link)
        checkpoint_link["normalized_overlap"] = {"real": overlap.real, "imag": overlap.imag}
        with checkpoint_path.open("a") as checkpoint_file:
            checkpoint_file.write(json.dumps(checkpoint_link) + "\n")
        print(
            f"  link {index:02d}->{next_index:02d}: |O|={abs(overlap):.10f}, "
            f"phase={np.angle(overlap):+.10f}, time={result['elapsed_seconds']:.3f} s",
            flush=True,
        )

    links.sort(key=lambda link: link["initial_point"])
    overlaps = np.array([link["normalized_overlap"] for link in links])
    total_unwrapped = float(np.sum(np.angle(overlaps)))
    berry_phase = float(np.angle(np.exp(1j * total_unwrapped)))
    result = {
        "lattice": [4, 4],
        "parameters": vars(params),
        "steps_per_leg": args.steps_per_leg,
        "number_of_links": len(links),
        "link_start": link_start,
        "link_stop": link_stop,
        "vertices": vertices.tolist(),
        "minimum_bdg_gap": min(gaps),
        "parities": parities,
        "hamiltonian_endpoint_closure_error": endpoint_error,
        "minimum_link_magnitude": min(link["magnitude"] for link in links),
        "total_unwrapped_link_phase": total_unwrapped,
        "berry_phase_radians": berry_phase,
        "berry_phase_over_pi": berry_phase / np.pi,
        "links": links,
    }
    for link in result["links"]:
        value = link["normalized_overlap"]
        link["normalized_overlap"] = {"real": value.real, "imag": value.imag}
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    print(f"Minimum link magnitude: {result['minimum_link_magnitude']:.10f}")
    print(f"Total unwrapped link phase: {total_unwrapped:+.12f}")
    print(f"Projected Berry phase: {berry_phase:+.12f} rad")
    print(f"Projected Berry phase / pi: {berry_phase / np.pi:+.12f}")
    print(f"Result file: {args.output.resolve()}")


if __name__ == "__main__":
    main()
