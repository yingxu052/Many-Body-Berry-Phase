"""First-order response of parton-projected Gaussian-state overlaps."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import PARAMETERS, twist_loop
from gaussian_overlap import normalized_gaussian_overlap
from model import TwoOrbitalModel, gaussian_bcs_state
from single_occupancy_overlap import single_occupancy_overlap


REFERENCE_MASK = 0xAAC8DA33


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-per-leg", type=int, default=4)
    parser.add_argument("--derivative-step", type=float, default=1.0e-4)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    steps = args.steps_per_leg
    h = args.derivative_step
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    points = twist_loop(steps)
    checkpoint = Path(__file__).resolve().with_name(
        f"projected_pairing_response_m{steps}_h{h:g}.jsonl"
    )
    completed = {}
    if checkpoint.exists():
        for line in checkpoint.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                completed[(record["deformation"], record["sign"], record["index"])] = record
        print(f"Resumed {len(completed)} records from {checkpoint}", flush=True)

    def hamiltonian(point, deformation, parameter):
        if deformation == "imaginary":
            return model.bdg_hamiltonian(
                *point, pairing_twist="linear", pairing_imag=parameter,
            )
        linear = model.bdg_hamiltonian(*point, pairing_twist="linear")
        cosine = model.bdg_hamiltonian(*point, pairing_twist="cosine")
        center = 0.5
        return (1.0 - center - parameter) * linear + (center + parameter) * cosine

    states_cache = {}
    for deformation in ("imaginary", "real_profile"):
        for sign in (-1, 1):
            parameter = sign * h
            states = [
                gaussian_bcs_state(
                    hamiltonian(point, deformation, parameter),
                    reference_mask=REFERENCE_MASK,
                )
                for point in points
            ]
            states_cache[(deformation, sign)] = states
            for index, state in enumerate(states):
                key = (deformation, sign, index)
                if key in completed:
                    continue
                next_index = (index + 1) % len(states)
                next_state = states[next_index]
                gaussian = normalized_gaussian_overlap(state.z, next_state.z)
                start = time.perf_counter()
                projected_result = single_occupancy_overlap(
                    state.z, next_state.z, model.n_cells,
                    reference_mask1=REFERENCE_MASK,
                    reference_mask2=REFERENCE_MASK,
                    workers=args.workers,
                )
                projected = projected_result["normalized_overlap"]
                record = {
                    "deformation": deformation,
                    "sign": sign,
                    "index": index,
                    "gaussian_overlap": [gaussian.real, gaussian.imag],
                    "projected_overlap": [projected.real, projected.imag],
                    "projected_magnitude": abs(projected),
                    "minimum_norm_fraction": min(
                        projected_result["norm1_fraction"],
                        projected_result["norm2_fraction"],
                    ),
                    "elapsed_seconds": time.perf_counter() - start,
                }
                with checkpoint.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                completed[key] = record
                print(
                    f"{deformation} sign={sign:+d} link {index:03d}: "
                    f"|P|={abs(projected):.8f}, time={record['elapsed_seconds']:.3f}s",
                    flush=True,
                )

    output_results = []
    for deformation in ("imaginary", "real_profile"):
        gaussian_links = []
        projected_links = []
        for index in range(len(points)):
            minus = completed[(deformation, -1, index)]
            plus = completed[(deformation, 1, index)]
            gaussian_minus = complex(*minus["gaussian_overlap"])
            gaussian_plus = complex(*plus["gaussian_overlap"])
            projected_minus = complex(*minus["projected_overlap"])
            projected_plus = complex(*plus["projected_overlap"])
            gaussian_links.append(
                (np.angle(gaussian_plus) - np.angle(gaussian_minus)) / (2.0 * h)
            )
            projected_links.append(
                (np.angle(projected_plus) - np.angle(projected_minus)) / (2.0 * h)
            )
        gaussian_links = np.asarray(gaussian_links)
        projected_links = np.asarray(projected_links)
        result = {
            "deformation": deformation,
            "steps_per_leg": steps,
            "derivative_step": h,
            "gaussian_leg_coefficients": [
                math.fsum(gaussian_links[leg * steps:(leg + 1) * steps])
                for leg in range(4)
            ],
            "projected_leg_coefficients": [
                math.fsum(projected_links[leg * steps:(leg + 1) * steps])
                for leg in range(4)
            ],
            "gaussian_total_coefficient": math.fsum(gaussian_links),
            "projected_total_coefficient": math.fsum(projected_links),
            "maximum_projected_gaussian_link_difference": float(
                np.max(np.abs(projected_links - gaussian_links))
            ),
            "minimum_projected_link_magnitude": min(
                completed[(deformation, sign, index)]["projected_magnitude"]
                for sign in (-1, 1) for index in range(len(points))
            ),
            "minimum_projected_norm_fraction": min(
                completed[(deformation, sign, index)]["minimum_norm_fraction"]
                for sign in (-1, 1) for index in range(len(points))
            ),
            "projected_link_coefficients": projected_links.tolist(),
            "gaussian_link_coefficients": gaussian_links.tolist(),
        }
        output_results.append(result)
        print("SUMMARY " + json.dumps({
            k: v for k, v in result.items()
            if k not in ("projected_link_coefficients", "gaussian_link_coefficients")
        }, indent=2), flush=True)

    output = Path(__file__).resolve().with_name(f"projected_pairing_response_m{steps}_h{h:g}.json")
    output.write_text(json.dumps({
        "projection": "P_G=product_i n_i(2-n_i), one parton per unit cell",
        "reference_mask": REFERENCE_MASK,
        "results": output_results,
    }, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
