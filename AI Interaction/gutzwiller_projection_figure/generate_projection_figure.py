#!/usr/bin/env python3
"""Berry phases for the two Gutzwiller projections below Eq. (7).

The physical system is fixed to 4 x 4 unit cells. Both curves use the common
projection-strength convention g=0 (unprojected) and g=1 (fully projected):

    P_parton(g) = prod_r [1 - g Q_r],
    Q_r = 1 - n_r (2 - n_r),  n_r = n_{r,c} + n_{r,d},
    P_2occ(g) = prod_r [1 - g n_{r,c} n_{r,d}].

Every overlap is evaluated exactly. The double-occupancy calculation reuses the
validated Gaussian-Wick enumerator in ../exact_gutzwiller. For the parton curve,

    Q_r = [1 + (1 - 2 n_{r,c})(1 - 2 n_{r,d})] / 2,

so its overlap also becomes a 2^N subset sum of Pfaffians for N=16 cells. This
is the cell-product operator in Eq. (6), not the different global interpolation
(1-g)|Psi> + g P_parton(1)|Psi>.

Example:
    python generate_projection_figure.py --m -2 --steps-per-leg 16

All model parameters can be tuned on the command line. Different parameter sets
receive different default output names, and each link is checkpointed.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
EXACT_CODE = ROOT.parent / "exact_gutzwiller"
if str(EXACT_CODE) not in sys.path:
    sys.path.insert(0, str(EXACT_CODE))

from berry_phase import twist_loop  # noqa: E402
from finite_g_overlap import (  # noqa: E402
    double_occupancy_gamma,
    finite_g_overlaps,
)
from gaussian_overlap import normalized_gaussian_overlap  # noqa: E402
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state  # noqa: E402
from overlap import _FLOAT_ARRAY, _load_library  # noqa: E402
from validate_gaussian_overlap import fock_vector  # noqa: E402

LX = LY = 4
DEFAULT_STRENGTHS = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"


def parse_strengths(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one projection strength is required")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Every projection strength g must lie between 0 and 1")
    if len(set(values)) != len(values):
        raise ValueError("Projection strengths must not contain duplicates")
    return values


def complex_record(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def majorana_gamma(z1: np.ndarray, z2: np.ndarray, reference_mask: int) -> np.ndarray:
    """Transition Wick matrix for Majoranas ordered (a_c,b_c,a_d,b_d) per cell.

    a=c+c^dag and b=i(c-c^dag), hence 1-2n=iab. The transformation is applied
    without complex conjugation because it mixes operators, not wavefunction
    coefficients. A common Slater reference is required for z1 and z2.

    double_occupancy_gamma stores ordered <c^dag c> contractions. Majorana
    Pfaffians instead need (1/2)<[A,B]>. For a creator/annihilator pair on the
    same mode this differs by (1/2)(2<c^dag c>-1), so the canonical
    anticommutator must be subtracted before the Majorana transformation.
    """
    base = double_occupancy_gamma(z1, z2, reference_mask)
    n_modes = z1.shape[0]
    symmetrized = base.copy()
    for mode in range(n_modes):
        creator, annihilator = 2 * mode, 2 * mode + 1
        symmetrized[creator, annihilator] -= 0.5
        symmetrized[annihilator, creator] += 0.5
    transform = np.zeros((2 * n_modes, 2 * n_modes), dtype=np.complex128)
    for mode in range(n_modes):
        row = 2 * mode
        column = 2 * mode
        # a = c^dag + c; b = -i c^dag + i c.
        transform[row, column] = 1.0
        transform[row, column + 1] = 1.0
        transform[row + 1, column] = -1j
        transform[row + 1, column + 1] = 1j
    gamma = transform @ symmetrized @ transform.T
    return 0.5 * (gamma - gamma.T)


def subset_factors(
    gamma1: np.ndarray,
    gamma2: np.ndarray,
    gamma12: np.ndarray,
    positive_weights: np.ndarray,
    workers: int,
) -> np.ndarray:
    """Evaluate subset sums sum_S weight^|S| Pf(Gamma_S) in compiled code."""
    n_cells = gamma1.shape[0] // 4
    arrays: list[np.ndarray] = []
    for gamma in (gamma1, gamma2, gamma12):
        arrays.extend((
            np.ascontiguousarray(gamma.real),
            np.ascontiguousarray(gamma.imag),
        ))
    # finite_g_factors_batch uses (-lambda)^|S|. Negative lambdas therefore
    # implement the positive weights needed for the parton convention.
    lambdas = np.ascontiguousarray(-positive_weights, dtype=np.float64)
    output = np.zeros(6 * len(lambdas), dtype=np.float64)
    library = _load_library()
    library.finite_g_factors_batch.argtypes = [
        _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY,
        _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY,
        ctypes.c_int, _FLOAT_ARRAY, ctypes.c_int, ctypes.c_int, _FLOAT_ARRAY,
    ]
    library.finite_g_factors_batch.restype = None
    library.finite_g_factors_batch(
        *arrays, n_cells, lambdas, len(lambdas), workers, output,
    )
    return output


def parton_overlaps(
    z1: np.ndarray,
    z2: np.ndarray,
    n_cells: int,
    strengths: list[float] | np.ndarray,
    reference_mask: int,
    workers: int,
) -> list[dict]:
    """Exact overlaps for the cell-product parton projection in Eq. (6).

    Define V_r=(1-2n_c)(1-2n_d), a=1-g/2, and b=g/2. Then
    P(g)=prod_r(a-bV_r) and P(g)^2=prod_r(A+B V_r), where
    A=a^2+b^2 and B=-2ab. With eta=-B/A, the common factor A^N cancels
    from normalized overlaps. Since <prod_{r in S}V_r>=(-1)^|S| Pf(Gamma_S),
    the required subset weight is +eta^|S|.
    """
    g = np.asarray(strengths, dtype=np.float64)
    a_factor = 1.0 - g + 0.5 * g * g
    eta = (g - 0.5 * g * g) / a_factor

    start = time.perf_counter()
    base_overlap = normalized_gaussian_overlap(z1, z2)
    gamma1 = majorana_gamma(z1, z1, reference_mask)
    gamma2 = majorana_gamma(z2, z2, reference_mask)
    gamma12 = majorana_gamma(z1, z2, reference_mask)
    raw = subset_factors(gamma1, gamma2, gamma12, eta, workers)
    elapsed = time.perf_counter() - start

    results = []
    for index, strength in enumerate(g):
        values = raw[6 * index:6 * index + 6]
        sum1 = complex(values[0], values[1])
        sum2 = complex(values[2], values[3])
        cross = complex(values[4], values[5])
        if sum1.real <= 0.0 or sum2.real <= 0.0:
            raise ValueError(f"Zero parton-projected norm at g={strength}")
        normalized = base_overlap * cross / np.sqrt(sum1 * sum2)
        scale = float(a_factor[index] ** n_cells)
        results.append({
            "g": float(strength),
            "normalized_overlap": complex(normalized),
            "norm1_fraction": float(scale * sum1.real),
            "norm2_fraction": float(scale * sum2.real),
            "norm1_imaginary_residual": float(sum1.imag),
            "norm2_imaginary_residual": float(sum2.imag),
            "unprojected_overlap": base_overlap,
            "elapsed_seconds": elapsed,
        })
    return results


def direct_projected_overlap(
    vector1: np.ndarray,
    vector2: np.ndarray,
    n_cells: int,
    strength: float,
    projection: str,
) -> tuple[complex, float, float]:
    """Independent Fock-space reference used only on a 2 x 2 test system."""
    weights = np.ones(len(vector1), dtype=np.float64)
    for state in range(len(vector1)):
        for cell in range(n_cells):
            occupied = int(bool(state & (1 << (2 * cell)))) + int(
                bool(state & (1 << (2 * cell + 1)))
            )
            if projection == "parton" and occupied != 1:
                weights[state] *= 1.0 - strength
            elif projection == "double_occupancy" and occupied == 2:
                weights[state] *= 1.0 - strength
    projected1 = weights * vector1
    projected2 = weights * vector2
    norm1 = float(np.vdot(projected1, projected1).real)
    norm2 = float(np.vdot(projected2, projected2).real)
    full1 = float(np.vdot(vector1, vector1).real)
    full2 = float(np.vdot(vector2, vector2).real)
    overlap = np.vdot(projected1, projected2) / np.sqrt(norm1 * norm2)
    return complex(overlap), norm1 / full1, norm2 / full2


def validate_against_fock_space(workers: int = 2) -> float:
    """Check both finite-g implementations against explicit 2 x 2 Fock vectors."""
    model = TwoOrbitalModel(2, 2, ModelParameters(m=1.5))
    state1 = gaussian_bcs_state(model.bdg_hamiltonian(0.23, -0.17))
    state2 = gaussian_bcs_state(
        model.bdg_hamiltonian(0.24, -0.17), reference_mask=state1.reference_mask,
    )
    vector1 = fock_vector(state1.z, state1.reference_mask)
    vector2 = fock_vector(state2.z, state2.reference_mask)
    largest_error = 0.0
    for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
        parton = parton_overlaps(
            state1.z, state2.z, model.n_cells, [strength],
            state1.reference_mask, workers,
        )[0]
        double = finite_g_overlaps(
            state1.z, state2.z, model.n_cells, [1.0 - strength],
            reference_mask=state1.reference_mask, workers=workers,
        )[0]
        for projection, obtained in (("parton", parton), ("double_occupancy", double)):
            expected = direct_projected_overlap(
                vector1, vector2, model.n_cells, strength, projection,
            )
            errors = (
                abs(expected[0] - obtained["normalized_overlap"]),
                abs(expected[1] - obtained["norm1_fraction"]),
                abs(expected[2] - obtained["norm2_fraction"]),
            )
            largest_error = max(largest_error, *errors)
    if largest_error > 1e-10:
        raise RuntimeError(f"Fock-space validation failed; maximum error={largest_error:.3e}")
    print(f"Fock-space validation passed (maximum error {largest_error:.3e})", flush=True)
    return float(largest_error)


def run_signature(configuration: dict) -> str:
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint(path: Path, signature: str) -> dict[int, dict]:
    completed: dict[int, dict] = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_signature") != signature:
                raise ValueError(
                    f"Checkpoint {path} belongs to different parameters. "
                    "Use another --output path or move the old checkpoint."
                )
            completed[int(record["initial_point"])] = record
    print(f"Resumed {len(completed)} links from {path}", flush=True)
    return completed


def serialize_values(parton: dict, double: dict) -> dict:
    return {
        "g": parton["g"],
        "parton_overlap": complex_record(parton["normalized_overlap"]),
        "parton_norm1_fraction": parton["norm1_fraction"],
        "parton_norm2_fraction": parton["norm2_fraction"],
        "double_occupancy_overlap": complex_record(double["normalized_overlap"]),
        "double_occupancy_norm1_fraction": double["norm1_fraction"],
        "double_occupancy_norm2_fraction": double["norm2_fraction"],
    }


def summarize(links: list[dict], strengths: list[float], projection: str) -> list[dict]:
    key = "parton" if projection == "parton" else "double_occupancy"
    results = []
    for g_index, strength in enumerate(strengths):
        overlaps = []
        norm_fractions = []
        for link in links:
            value = link["values"][g_index]
            if value["g"] != strength:
                raise RuntimeError("Checkpoint projection-strength ordering is inconsistent")
            overlaps.append(complex(**value[f"{key}_overlap"]))
            norm_fractions.extend((
                value[f"{key}_norm1_fraction"], value[f"{key}_norm2_fraction"],
            ))
        if min(abs(value) for value in overlaps) < 1e-12:
            raise ValueError(f"A normalized {key} link overlap is zero at g={strength}")
        total_phase = math.fsum(float(np.angle(value)) for value in overlaps)
        phase = float(np.angle(np.exp(1j * total_phase)))
        results.append({
            "g": strength,
            "berry_phase_radians": phase,
            "berry_phase_over_pi": phase / np.pi,
            "minimum_link_magnitude": float(min(abs(value) for value in overlaps)),
            "minimum_norm_fraction": float(min(norm_fractions)),
            "total_unwrapped_link_phase": total_phase,
        })
    return results


def save_csv(path: Path, strengths: list[float], parton: list[dict], double: list[dict]) -> None:
    fields = [
        "g",
        "parton_berry_phase_radians", "parton_berry_phase_over_pi",
        "double_occupancy_berry_phase_radians", "double_occupancy_berry_phase_over_pi",
        "parton_minimum_link_magnitude", "double_occupancy_minimum_link_magnitude",
        "parton_minimum_norm_fraction", "double_occupancy_minimum_norm_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p_item, d_item in zip(parton, double):
            writer.writerow({
                "g": p_item["g"],
                "parton_berry_phase_radians": p_item["berry_phase_radians"],
                "parton_berry_phase_over_pi": p_item["berry_phase_over_pi"],
                "double_occupancy_berry_phase_radians": d_item["berry_phase_radians"],
                "double_occupancy_berry_phase_over_pi": d_item["berry_phase_over_pi"],
                "parton_minimum_link_magnitude": p_item["minimum_link_magnitude"],
                "double_occupancy_minimum_link_magnitude": d_item["minimum_link_magnitude"],
                "parton_minimum_norm_fraction": p_item["minimum_norm_fraction"],
                "double_occupancy_minimum_norm_fraction": d_item["minimum_norm_fraction"],
            })


def save_figure(
    path: Path,
    parameters: ModelParameters,
    strengths: list[float],
    parton: list[dict],
    double: list[dict],
    steps_per_leg: int,
) -> None:
    parton_phase = [item["berry_phase_over_pi"] for item in parton]
    double_phase = [item["berry_phase_over_pi"] for item in double]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.plot(strengths, parton_phase, "o-", color="#1676a3", lw=1.9, ms=5.5,
            label="Parton projection")
    ax.plot(strengths, double_phase, "x--", color="#dc7d21", lw=1.6, ms=6.5,
            label="Double-occupancy suppression")
    ax.set_xlabel("Projection strength $g$ (0 = unprojected, 1 = full)")
    ax.set_ylabel(r"Many-body Berry phase $\theta_{\mathrm{BP}}/\pi$")
    ax.set_xlim(-0.025, 1.025)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, loc="best")
    all_phases = np.asarray(parton_phase + double_phase)
    center = float(np.mean(all_phases))
    span = float(np.ptp(all_phases))
    margin = max(0.01, 0.12 * span)
    lower = min(float(np.min(all_phases)) - margin, center - 0.005)
    upper = max(float(np.max(all_phases)) + margin, center + 0.005)
    ax.set_ylim(lower, upper)
    ax.set_title(
        "Exact Gutzwiller-projected Berry phases: $4\\times4$ unit cells\n"
        f"$t={parameters.t:g}$, $t_1={parameters.t1:g}$, $m={parameters.m:g}$, "
        f"$\\Delta_p={parameters.delta_p:g}$, $\\Delta_s={parameters.delta_s:g}$; "
        f"{steps_per_leg} steps/leg",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--t", type=float, default=1.0, help="intraorbital hopping")
    parser.add_argument("--t1", type=float, default=1.0, help="interorbital hopping")
    parser.add_argument("--m", type=float, default=-2.0, help="orbital mass parameter")
    parser.add_argument("--delta-p", type=float, default=1.0, help="p-wave pairing amplitude")
    parser.add_argument("--delta-s", type=float, default=0.5, help="interorbital s-wave pairing")
    parser.add_argument("--steps-per-leg", type=int, default=16, help="links on each of four legs")
    parser.add_argument("--g-values", default=DEFAULT_STRENGTHS,
                        help="comma-separated strengths; 0=unprojected, 1=full")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--pairing-twist", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--reference-mask", type=lambda value: int(value, 0), default=None,
                        help="optional common Slater reference, e.g. 0x5057F515")
    parser.add_argument("--output", type=Path, default=None,
                        help="output JSON path; default includes a parameter fingerprint")
    parser.add_argument("--validate-only", action="store_true",
                        help="run the independent 2x2 Fock-space validation and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps_per_leg < 1 or args.workers < 1:
        raise ValueError("steps-per-leg and workers must be positive")
    strengths = parse_strengths(args.g_values)
    validation_error = validate_against_fock_space(workers=min(2, args.workers))
    if args.validate_only:
        return

    parameters = ModelParameters(
        t=args.t, t1=args.t1, m=args.m, delta_p=args.delta_p, delta_s=args.delta_s,
    )
    model = TwoOrbitalModel(LX, LY, parameters)
    vertices, points = twist_loop(args.steps_per_leg)

    hamiltonians = [
        model.bdg_hamiltonian(*point, pairing_twist=args.pairing_twist)
        for point in points
    ]
    if args.reference_mask is not None:
        reference_mask = args.reference_mask
        states = [gaussian_bcs_state(h, reference_mask=reference_mask) for h in hamiltonians]
    else:
        # A reference that is regular at one vertex need not remain regular on
        # every leg. Select one common reference by minimizing the worst
        # condition number over the entire loop, not merely at its first point.
        automatic_states = [gaussian_bcs_state(h) for h in hamiltonians]
        candidate_set = {state.reference_mask for state in automatic_states}
        if model.n_modes == 32:
            # Two additional starting points that were independently validated
            # for the existing 4 x 4 calculations. They are only candidates;
            # the loop-wide condition number still determines the final choice.
            candidate_set.update((0x5057F515, 0xAAC8DA33))
        candidates = sorted(candidate_set)
        best = None
        for candidate in candidates:
            try:
                candidate_states = [
                    gaussian_bcs_state(h, reference_mask=candidate) for h in hamiltonians
                ]
            except ValueError:
                continue
            score = max(state.reference_condition for state in candidate_states)
            if best is None or score < best[0]:
                best = (score, candidate, candidate_states)
        if best is None:
            raise ValueError(
                "No automatically selected Slater reference is regular on the whole loop; "
                "specify --reference-mask explicitly."
            )
        _, reference_mask, states = best
    if any(state.parity != 0 for state in states):
        raise ValueError("The common-reference BCS construction requires even Gaussian parity")
    minimum_gap = min(state.gap for state in states)
    maximum_condition = max(state.reference_condition for state in states)
    print(
        f"4 x 4 cells, {len(points)} links, {len(strengths)} projection strengths; "
        f"minimum BdG gap={minimum_gap:.8e}, maximum reference condition={maximum_condition:.3e}",
        flush=True,
    )

    configuration = {
        "lattice": [LX, LY],
        "parameters": asdict(parameters),
        "steps_per_leg": args.steps_per_leg,
        "g_values": strengths,
        "pairing_twist": args.pairing_twist,
        "reference_mask": reference_mask,
        "parton_projection": "prod_r[1-g Q_r], Q_r=1-n_r(2-n_r)",
        "double_occupancy_projection": "prod_r[1-g n_rc n_rd]",
    }
    signature = run_signature(configuration)
    output = args.output or ROOT / f"projection_berry_{signature[:12]}.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".jsonl")
    completed = load_checkpoint(checkpoint, signature)

    links = []
    for index, state in enumerate(states):
        next_index = (index + 1) % len(states)
        if index in completed:
            record = completed[index]
        else:
            next_state = states[next_index]
            parton = parton_overlaps(
                state.z, next_state.z, model.n_cells, strengths,
                reference_mask, args.workers,
            )
            # finite_g_overlaps uses the opposite endpoint convention for its
            # input: original_g=1 is unprojected and original_g=0 is full.
            double = finite_g_overlaps(
                state.z, next_state.z, model.n_cells, [1.0 - g for g in strengths],
                reference_mask=reference_mask, workers=args.workers,
            )
            values = [serialize_values(p, d) for p, d in zip(parton, double)]
            record = {
                "run_signature": signature,
                "initial_point": index,
                "final_point": next_index,
                "phi_initial": points[index].tolist(),
                "phi_final": points[next_index].tolist(),
                "values": values,
            }
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            completed[index] = record
        links.append(record)
        print(f"Completed exact link {index:03d}->{next_index:03d}", flush=True)

    links.sort(key=lambda record: record["initial_point"])
    if len(links) != len(points):
        raise RuntimeError("The twist loop is incomplete")
    parton_results = summarize(links, strengths, "parton")
    double_results = summarize(links, strengths, "double_occupancy")
    for p_item, d_item in zip(parton_results, double_results):
        print(
            f"g={p_item['g']:.3f}: parton={p_item['berry_phase_over_pi']:+.12f} pi, "
            f"double={d_item['berry_phase_over_pi']:+.12f} pi",
            flush=True,
        )

    phase_difference = max(abs(float(np.angle(np.exp(1j * (
        p["berry_phase_radians"] - d["berry_phase_radians"]
    ))))) for p, d in zip(parton_results, double_results))
    endpoint_error = float(np.linalg.norm(
        model.bdg_hamiltonian(2 * np.pi, 2 * np.pi, pairing_twist=args.pairing_twist)
        - model.bdg_hamiltonian(0.0, 0.0, pairing_twist=args.pairing_twist)
    ))
    payload = dict(configuration)
    payload.update({
        "run_signature": signature,
        "method": "exact Gaussian-Wick subset enumeration (2^16 subsets per overlap)",
        "small_fock_space_validation_maximum_error": validation_error,
        "vertices": vertices.tolist(),
        "minimum_bdg_gap": minimum_gap,
        "maximum_reference_condition": maximum_condition,
        "gaussian_parities": [state.parity for state in states],
        "hamiltonian_endpoint_closure_error": endpoint_error,
        "maximum_parton_double_phase_difference_radians": phase_difference,
        "parton_results": parton_results,
        "double_occupancy_results": double_results,
        "links": links,
    })
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    save_csv(csv_path, strengths, parton_results, double_results)
    save_figure(figure_path, parameters, strengths, parton_results, double_results, args.steps_per_leg)
    print(f"JSON: {output}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
