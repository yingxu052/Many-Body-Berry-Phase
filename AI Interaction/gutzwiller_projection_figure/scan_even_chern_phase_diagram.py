#!/usr/bin/env python3
"""Exact projected Berry-phase diagram restricted to even Majorana Chern numbers.

For every (m, Delta) point, this driver first computes the occupied-BdG Chern
number and uses nu_M=-C_BdG, consistent with nu_M=+2 at m=-2. Odd-nu_M,
gapless, and unclassified points are recorded but are never sent to the exact
projected-overlap enumerator. Even-nu_M points are evaluated for both

    P_parton(g) = prod_r [1-g(1-n_r(2-n_r))],
    P_2occ(g)   = prod_r [1-g n_{r,c} n_{r,d}],

on 4 x 4 unit cells. The default is full projection (g=1). At g=1 the parton
calculation uses direct enumeration of the 2^16 one-parton-per-cell Fock
configurations; the double-occupancy calculation uses the equivalent exact
2^16 Gaussian-Wick subset sum. Other g values use the validated subset sums
for both projections.

Delta sets delta_p=Delta and delta_s=Delta/2, as in the manuscript. The plotted
phase is theta_BP modulo pi; raw wrapped phases are retained in JSON and CSV.

Example:
    python scan_even_chern_phase_diagram.py \
        --m-min -4.75 --m-max 4.75 --m-points 20 \
        --delta-values 0.25,0.75,1,1.5 --steps-per-leg 2

Every completed grid point is checkpointed. Repeating the same command resumes
the scan without recomputing finished points.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

import generate_projection_figure as projected
from majorana_chern import occupied_chern
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state
from overlap import projected_overlap
from single_occupancy_overlap import _unprojected_norm, single_occupancy_overlap

ROOT = Path(__file__).resolve().parent
LX = LY = 4


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Expected a nonempty comma-separated list of finite numbers")
    if len(set(values)) != len(values):
        raise ValueError("Grid values must not contain duplicates")
    return values


def select_common_states(
    hamiltonians: list[np.ndarray],
    n_modes: int,
    requested_reference: int | None,
):
    """Choose one Slater reference that is regular throughout the twist loop."""
    if requested_reference is not None:
        states = [gaussian_bcs_state(h, reference_mask=requested_reference) for h in hamiltonians]
        return requested_reference, states

    automatic_states = [gaussian_bcs_state(h) for h in hamiltonians]
    candidates = {state.reference_mask for state in automatic_states}
    if n_modes == 32:
        candidates.update((0x5057F515, 0xAAC8DA33, 0x08ABB8DB))
    best = None
    for candidate in sorted(candidates):
        try:
            states = [gaussian_bcs_state(h, reference_mask=candidate) for h in hamiltonians]
        except ValueError:
            continue
        score = max(state.reference_condition for state in states)
        if best is None or score < best[0]:
            best = (score, candidate, states)
    if best is None:
        raise ValueError("No regular common Slater reference found on the twist loop")
    return best[1], best[2]


def berry_result(overlaps: list[complex], norm_fractions: list[float]) -> dict:
    magnitudes = [abs(value) for value in overlaps]
    if any(not math.isfinite(value) for value in magnitudes):
        raise ValueError("A normalized link overlap is not finite")
    if min(magnitudes) == 0.0:
        raise ValueError("A normalized link overlap is exactly zero")
    total_phase = math.fsum(float(np.angle(value)) for value in overlaps)
    phase = float(np.angle(np.exp(1j * total_phase)))
    return {
        "berry_phase_radians": phase,
        "berry_phase_over_pi": phase / np.pi,
        "berry_phase_mod_pi_radians": phase % np.pi,
        "berry_phase_mod_pi_over_pi": (phase % np.pi) / np.pi,
        "total_unwrapped_link_phase": total_phase,
        "minimum_link_magnitude": float(min(abs(value) for value in overlaps)),
        "minimum_norm_fraction": float(min(norm_fractions)),
    }


def compute_even_point(
    params: ModelParameters,
    strength: float,
    steps_per_leg: int,
    pairing_twist: str,
    requested_reference: int | None,
    workers: int,
) -> dict:
    """Compute both exact projected Wilson loops at one even-nu_M grid point."""
    model = TwoOrbitalModel(LX, LY, params)
    _, points = projected.twist_loop(steps_per_leg)
    hamiltonians = [
        model.bdg_hamiltonian(*point, pairing_twist=pairing_twist) for point in points
    ]
    reference, states = select_common_states(hamiltonians, model.n_modes, requested_reference)
    if any(state.parity != 0 for state in states):
        raise ValueError("Only even-parity Gaussian ground states are supported")

    parton_overlaps = []
    double_overlaps = []
    parton_norms = []
    double_norms = []
    direct_double_fallbacks = 0
    for index, state in enumerate(states):
        next_state = states[(index + 1) % len(states)]
        if strength == 0.0:
            overlap = projected.normalized_gaussian_overlap(state.z, next_state.z)
            parton = double = {
                "normalized_overlap": overlap,
                "norm1_fraction": 1.0,
                "norm2_fraction": 1.0,
            }
        else:
            if strength == 1.0:
                parton = single_occupancy_overlap(
                    state.z, next_state.z, model.n_cells,
                    reference_mask1=reference, reference_mask2=reference,
                    workers=workers,
                )
            else:
                parton = projected.parton_overlaps(
                    state.z, next_state.z, model.n_cells, [strength],
                    reference, workers,
                )[0]
            # The reusable finite-g routine has the opposite endpoint convention.
            double = projected.finite_g_overlaps(
                state.z, next_state.z, model.n_cells, [1.0 - strength],
                reference_mask=reference, workers=workers,
            )[0]
            if strength == 1.0 and not np.all(np.isfinite([
                double["normalized_overlap"].real, double["normalized_overlap"].imag,
            ])):
                # Near a weak Gaussian link, cross_factor/base_overlap suffers
                # catastrophic cancellation. Direct base-3 Fock enumeration
                # needs no unprojected-overlap division and remains finite.
                direct = projected_overlap(
                    state.z, next_state.z, model.n_cells, parity=0,
                    reference_mask1=reference, reference_mask2=reference,
                    workers=workers,
                )
                double = {
                    "normalized_overlap": direct["normalized_overlap"],
                    "norm1_fraction": direct["norm1"] / _unprojected_norm(state.z),
                    "norm2_fraction": direct["norm2"] / _unprojected_norm(next_state.z),
                }
                direct_double_fallbacks += 1
        parton_overlaps.append(parton["normalized_overlap"])
        double_overlaps.append(double["normalized_overlap"])
        parton_norms.extend((parton["norm1_fraction"], parton["norm2_fraction"]))
        double_norms.extend((double["norm1_fraction"], double["norm2_fraction"]))

    parton_result = berry_result(parton_overlaps, parton_norms)
    double_result = berry_result(double_overlaps, double_norms)
    endpoint_error = float(np.linalg.norm(
        model.bdg_hamiltonian(2 * np.pi, 2 * np.pi, pairing_twist=pairing_twist)
        - model.bdg_hamiltonian(0.0, 0.0, pairing_twist=pairing_twist)
    ))
    return {
        "status": "computed",
        "projection_strength": strength,
        "reference_mask": reference,
        "minimum_twist_bdg_gap": min(state.gap for state in states),
        "maximum_reference_condition": max(state.reference_condition for state in states),
        "hamiltonian_endpoint_closure_error": endpoint_error,
        "direct_double_occupancy_fallback_links": direct_double_fallbacks,
        "parton": parton_result,
        "double_occupancy": double_result,
        "parton_double_phase_difference_radians": float(np.angle(np.exp(1j * (
            parton_result["berry_phase_radians"] - double_result["berry_phase_radians"]
        )))),
    }


def load_checkpoint(path: Path, signature: str) -> dict[tuple[int, int], dict]:
    records = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_signature") != signature:
                raise ValueError(
                    f"Checkpoint {path} belongs to a different scan. "
                    "Use another --output path or move the old checkpoint."
                )
            records[(record["delta_index"], record["m_index"])] = record
    if records:
        print(f"Resumed {len(records)} grid points from {path}", flush=True)
    return records


def save_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "m", "delta", "nu_M", "occupied_bdg_chern", "zero_twist_gap",
        "minimum_twist_bdg_gap", "projection_strength", "reliable_for_plot",
        "parton_berry_phase_radians", "parton_berry_phase_mod_pi_over_pi",
        "double_occupancy_berry_phase_radians", "double_occupancy_berry_phase_mod_pi_over_pi",
        "parton_double_phase_difference_radians",
        "parton_minimum_link_magnitude", "double_occupancy_minimum_link_magnitude",
        "parton_minimum_norm_fraction", "double_occupancy_minimum_norm_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if record["status"] != "computed":
                continue
            row = {key: record.get(key) for key in fields}
            for projection_name in ("parton", "double_occupancy"):
                result = record[projection_name]
                row[f"{projection_name}_berry_phase_radians"] = result["berry_phase_radians"]
                row[f"{projection_name}_berry_phase_mod_pi_over_pi"] = result["berry_phase_mod_pi_over_pi"]
                row[f"{projection_name}_minimum_link_magnitude"] = result["minimum_link_magnitude"]
                row[f"{projection_name}_minimum_norm_fraction"] = result["minimum_norm_fraction"]
            writer.writerow(row)


def save_figure(
    path: Path,
    m_values: list[float],
    delta_values: list[float],
    records: list[dict],
    configuration: dict,
) -> None:
    shape = (len(delta_values), len(m_values))
    nu_grid = np.full(shape, np.nan)
    parton_grid = np.full(shape, np.nan)
    double_grid = np.full(shape, np.nan)
    for record in records:
        i, j = record["delta_index"], record["m_index"]
        if record["status"] == "computed" and record.get("reliable_for_plot", True):
            nu_grid[i, j] = record["nu_M"]
            parton_grid[i, j] = record["parton"]["berry_phase_mod_pi_over_pi"]
            double_grid[i, j] = record["double_occupancy"]["berry_phase_mod_pi_over_pi"]

    fig, axes = plt.subplots(
        1, 3, figsize=(15.2, 4.8), sharex=True, sharey=True, constrained_layout=True,
    )
    even_values = sorted({int(value) for value in nu_grid[np.isfinite(nu_grid)]})
    if even_values:
        colors = plt.cm.coolwarm(np.linspace(0.12, 0.88, len(even_values)))
        nu_cmap = ListedColormap(colors)
        index_grid = np.full(shape, np.nan)
        for index, value in enumerate(even_values):
            index_grid[nu_grid == value] = index
        boundaries = np.arange(-0.5, len(even_values) + 0.5, 1.0)
        nu_norm = BoundaryNorm(boundaries, nu_cmap.N)
        nu_plot = axes[0].pcolormesh(
            m_values, delta_values, np.ma.masked_invalid(index_grid),
            shading="nearest", cmap=nu_cmap, norm=nu_norm,
        )
        nu_cbar = fig.colorbar(nu_plot, ax=axes[0], ticks=np.arange(len(even_values)), pad=0.02)
        nu_cbar.ax.set_yticklabels([str(value) for value in even_values])
    else:
        axes[0].text(0.5, 0.5, "No reliable even-$\nu_M$ points", ha="center", va="center",
                     transform=axes[0].transAxes)
    axes[0].set_title(r"Majorana Chern number $\nu_M$")
    axes[0].set_ylabel(r"Pairing scale $\Delta$")

    phase_cmap = plt.get_cmap("viridis").copy()
    phase_cmap.set_bad("#e6e6e6")
    phase_images = []
    for ax, grid, title in zip(
        axes[1:], (parton_grid, double_grid),
        ("Parton projection", "Double-occupancy suppression"),
    ):
        image = ax.pcolormesh(
            m_values, delta_values, np.ma.masked_invalid(grid), shading="nearest",
            cmap=phase_cmap, vmin=0.0, vmax=1.0,
        )
        phase_images.append(image)
        ax.set_title(title)
    phase_cbar = fig.colorbar(phase_images[-1], ax=axes[1:], pad=0.02, fraction=0.04)
    phase_cbar.set_label(r"$\theta_{\mathrm{BP}}/\pi\;(\mathrm{mod}\;\pi)$")
    phase_cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    for ax in axes:
        ax.set_xlabel("Mass parameter $m$")
        ax.set_facecolor("#e6e6e6")
    fig.text(
        0.5, -0.015,
        "Grey: odd, gapless, failed, or weak-link points are not plotted",
        ha="center", va="top", fontsize=10,
    )
    fig.suptitle(
        f"Exact Gutzwiller-projected Berry phases in even-$\\nu_M$ regions\n"
        f"$4\\times4$ cells, $g={configuration['projection_strength']:g}$, "
        f"$t={configuration['t']:g}$, $t_1={configuration['t1']:g}$, "
        f"{configuration['steps_per_leg']} steps/leg; "
        r"$\Delta_p=\Delta$, $\Delta_s=\Delta/2$",
        fontsize=13,
    )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--m-min", type=float, default=-4.75)
    parser.add_argument("--m-max", type=float, default=4.75)
    parser.add_argument("--m-points", type=int, default=20)
    parser.add_argument("--m-values", default=None,
                        help="optional comma-separated masses; overrides m-min/max/points")
    parser.add_argument("--delta-values", default="0.25,0.75,1,1.5",
                        help="Delta sets delta_p=Delta and delta_s=Delta/2")
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--projection-strength", type=float, default=1.0)
    parser.add_argument("--steps-per-leg", type=int, default=2)
    parser.add_argument("--pairing-twist", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--chern-grid", type=int, default=40,
                        help="Fukui mesh used to classify nu_M")
    parser.add_argument("--minimum-chern-gap", type=float, default=1e-5,
                        help="smaller zero-twist gaps are masked as gapless")
    parser.add_argument("--minimum-reliable-link", type=float, default=1e-8,
                        help="smaller links remain in JSON/CSV but are masked in the figure")
    parser.add_argument("--reference-mask", type=lambda value: int(value, 0), default=None,
                        help="optional fixed Slater reference for every computed point")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.m_points < 1 or args.chern_grid < 4 or args.steps_per_leg < 1 or args.workers < 1:
        raise ValueError("Grid sizes, steps-per-leg, and workers must be positive")
    if not 0.0 <= args.projection_strength <= 1.0:
        raise ValueError("projection-strength must lie between 0 and 1")
    m_values = (parse_float_list(args.m_values) if args.m_values is not None else
                np.linspace(args.m_min, args.m_max, args.m_points).tolist())
    delta_values = parse_float_list(args.delta_values)
    if any(delta <= 0.0 for delta in delta_values):
        raise ValueError("Use positive Delta values; Delta=0 requires separate gapless-line handling")

    validation_error = projected.validate_against_fock_space(workers=min(2, args.workers))
    configuration = {
        "lattice": [LX, LY],
        "m_values": m_values,
        "delta_values": delta_values,
        "t": args.t,
        "t1": args.t1,
        "pairing_convention": "delta_p=Delta, delta_s=Delta/2",
        "projection_strength": args.projection_strength,
        "steps_per_leg": args.steps_per_leg,
        "pairing_twist": args.pairing_twist,
        "chern_grid": args.chern_grid,
        "minimum_chern_gap": args.minimum_chern_gap,
        "reference_mask": args.reference_mask,
        "nu_sign_convention": "nu_M=-occupied_BdG_Chern; m=-2 gives nu_M=+2",
    }
    signature = projected.run_signature(configuration)
    output = (args.output or ROOT / f"even_chern_phase_diagram_{signature[:12]}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".jsonl")
    completed = load_checkpoint(checkpoint, signature)

    total = len(m_values) * len(delta_values)
    for delta_index, delta in enumerate(delta_values):
        for m_index, mass in enumerate(m_values):
            key = (delta_index, m_index)
            if key in completed:
                continue
            params = ModelParameters(
                t=args.t, t1=args.t1, m=float(mass),
                delta_p=float(delta), delta_s=float(delta) / 2.0,
            )
            occupied_c, zero_gap = occupied_chern(params, grid=args.chern_grid)
            nearest_chern = round(occupied_c)
            chern_residual = abs(occupied_c - nearest_chern)
            nu_m = int(-nearest_chern)
            record = {
                "run_signature": signature,
                "delta_index": delta_index,
                "m_index": m_index,
                "m": float(mass),
                "delta": float(delta),
                "parameters": asdict(params),
                "occupied_bdg_chern": float(occupied_c),
                "chern_integer_residual": float(chern_residual),
                "nu_M": nu_m,
                "zero_twist_gap": float(zero_gap),
            }
            number = delta_index * len(m_values) + m_index + 1
            if zero_gap < args.minimum_chern_gap or chern_residual > 1e-5:
                record["status"] = "gapless_or_unclassified"
                print(f"[{number:03d}/{total}] m={mass:+.4f}, Delta={delta:.4f}: masked gapless/unclassified", flush=True)
            elif nu_m % 2 != 0:
                record["status"] = "odd_nu_M"
                print(f"[{number:03d}/{total}] m={mass:+.4f}, Delta={delta:.4f}: odd nu_M={nu_m:+d}, skipped", flush=True)
            else:
                print(f"[{number:03d}/{total}] m={mass:+.4f}, Delta={delta:.4f}: even nu_M={nu_m:+d}, computing", flush=True)
                try:
                    result = compute_even_point(
                        params, args.projection_strength, args.steps_per_leg,
                        args.pairing_twist, args.reference_mask, args.workers,
                    )
                    record.update(result)
                    print(
                        f"    parton={record['parton']['berry_phase_mod_pi_over_pi']:.9f} pi, "
                        f"double={record['double_occupancy']['berry_phase_mod_pi_over_pi']:.9f} pi "
                        "(both modulo pi)",
                        flush=True,
                    )
                except (ValueError, np.linalg.LinAlgError) as error:
                    record["status"] = "failed_even_nu_M"
                    record["error"] = str(error)
                    print(f"    FAILED: {error}", flush=True)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            completed[key] = record

    records = [completed[(i, j)] for i in range(len(delta_values)) for j in range(len(m_values))]
    counts = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    computed = [record for record in records if record["status"] == "computed"]
    for record in computed:
        minimum_link = min(
            record["parton"]["minimum_link_magnitude"],
            record["double_occupancy"]["minimum_link_magnitude"],
        )
        record["reliable_for_plot"] = bool(minimum_link >= args.minimum_reliable_link)
        for projection_name in ("parton", "double_occupancy"):
            result = record[projection_name]
            # -pi and 0 are identical modulo pi; avoid artificial endpoint
            # colors when roundoff places a trivial phase at +pi.
            if abs(result["berry_phase_mod_pi_over_pi"] - 1.0) < 1e-10:
                result["berry_phase_mod_pi_radians"] = 0.0
                result["berry_phase_mod_pi_over_pi"] = 0.0
    reliable = [record for record in computed if record["reliable_for_plot"]]
    payload = dict(configuration)
    payload.update({
        "run_signature": signature,
        "small_fock_space_validation_maximum_error": validation_error,
        "minimum_reliable_link_for_plot": args.minimum_reliable_link,
        "status_counts": counts,
        "computed_even_points": len(computed),
        "reliable_computed_even_points": len(reliable),
        "points": records,
    })
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    csv_path = output.with_suffix(".csv")
    figure_path = output.with_suffix(".png")
    save_csv(csv_path, records)
    save_figure(figure_path, m_values, delta_values, records, configuration)
    print(f"Status counts: {counts}")
    print(f"JSON: {output}")
    print(f"CSV: {csv_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
