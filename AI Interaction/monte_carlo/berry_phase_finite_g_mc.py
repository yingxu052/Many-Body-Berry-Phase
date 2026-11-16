"""Finite-g Monte Carlo Berry phase for interpolated parton-projected states.

The interpolated state is
    |Psi_g> proportional to [(1-g) + g P_G] |G>.
Consequently its Fock-space amplitude is b_G(S) on physical one-parton-per-cell
configurations and (1-g) b_G(S) on all other configurations.  Sampling this
amplitude directly avoids estimating exponentially small projected norms.

For 0<g<1 the Markov chain samples the full fixed-parity Fock space.  Hops and
pair creation/annihilation are supplemented by physical-space flips and swaps.
At g=1 only the physical space survives; at g=0 Gaussian overlaps are exact.

Example
-------
python monte_carlo/berry_phase_finite_g_mc.py --g 0.5 --steps-per-leg 2 \
    --chains 8 --burn-in-sweeps 50 --measurement-sweeps 250
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server_ready.berry_phase import (  # noqa: E402
    ModelParameters,
    TwoOrbitalModel,
    _batched_pfaffian,
    _choose_reference,
    _make_state,
    _positive_blocks,
    normalized_gaussian_overlap,
    twist_loop,
)
from monte_carlo.berry_phase_mc import (  # noqa: E402
    _mean_and_autocorrelation_error,
    _original_occupations,
    _phase_error,
    _reference_occupations,
    _transformation_signs,
)


def gaussian_amplitudes(
    z: np.ndarray, occupations: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Gaussian amplitudes for arbitrary fixed-parity Fock configurations."""
    transformed = occupations != reference[None, :]
    counts = np.sum(transformed, axis=1)
    signs = _transformation_signs(occupations, reference)
    amplitudes = np.empty(len(occupations), dtype=np.complex128)
    for count in np.unique(counts):
        rows = np.flatnonzero(counts == count)
        if count == 0:
            amplitudes[rows] = signs[rows]
            continue
        modes = np.empty((len(rows), int(count)), dtype=np.int64)
        for local, row in enumerate(rows):
            modes[local] = np.flatnonzero(transformed[row])
        matrices = z[modes[:, :, None], modes[:, None, :]]
        amplitudes[rows] = signs[rows] * _batched_pfaffian(matrices)
    return amplitudes


def physical_configurations(occupations: np.ndarray) -> np.ndarray:
    n_cells = occupations.shape[1] // 2
    occupancy = occupations.reshape(len(occupations), n_cells, 2).sum(axis=2)
    return np.all(occupancy == 1, axis=1)


def interpolated_amplitudes(
    z: np.ndarray,
    occupations: np.ndarray,
    reference: np.ndarray,
    g: float,
) -> np.ndarray:
    amplitudes = gaussian_amplitudes(z, occupations, reference)
    if g != 0:
        physical = physical_configurations(occupations)
        amplitudes[~physical] *= 1.0 - g
    return amplitudes


def _metropolis_finite_g(
    z: np.ndarray,
    z_next: np.ndarray,
    reference: np.ndarray,
    g: float,
    *,
    chains: int,
    burn_in_sweeps: int,
    measurement_sweeps: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n_modes = z.shape[0]
    n_cells = n_modes // 2
    spins = rng.integers(0, 2, size=(chains, n_cells), dtype=np.int8)
    occupations = _original_occupations(spins)
    amplitudes = interpolated_amplitudes(z, occupations, reference, g)
    rows = np.arange(chains)
    samples = np.empty((measurement_sweeps, chains), dtype=np.complex128)
    accepted = 0
    proposals = 0

    for sweep in range(burn_in_sweeps + measurement_sweeps):
        for _ in range(n_cells):
            move = rng.random()
            first = rng.integers(0, n_modes, size=chains)
            second = rng.integers(0, n_modes, size=chains)
            cell1, cell2 = first // 2, second // 2
            occ1 = occupations[rows, first]
            occ2 = occupations[rows, second]

            if move < 0.25:
                # A particle hop; its reverse is proposed with the same probability.
                active = rows[(occ1 == 1) & (occ2 == 0)]
                i, j = first[active], second[active]
                trial = occupations[active].copy()
                local = np.arange(len(active))
                trial[local, i] = 0
                trial[local, j] = 1
            elif move < 0.50:
                # Pair creation/annihilation changes particle number by +-2.
                active = rows[(first != second) & (occ1 == occ2)]
                i, j = first[active], second[active]
                trial = occupations[active].copy()
                local = np.arange(len(active))
                trial[local, i] = 1 - trial[local, i]
                trial[local, j] = 1 - trial[local, j]
            else:
                physical1 = (
                    occupations[rows, 2 * cell1] + occupations[rows, 2 * cell1 + 1] == 1
                )
                physical2 = (
                    occupations[rows, 2 * cell2] + occupations[rows, 2 * cell2 + 1] == 1
                )
                if move < 0.65:
                    # A physical c<->d flip at one cell.
                    active = rows[physical1]
                    trial = occupations[active].copy()
                    c = 2 * cell1[active]
                    local = np.arange(len(active))
                    old_c = trial[local, c].copy()
                    trial[local, c] = trial[local, c + 1]
                    trial[local, c + 1] = old_c
                elif move < 0.85:
                    # Simultaneous physical flips at two cells (Delta-N_d=+-2 or 0).
                    active = rows[(cell1 != cell2) & physical1 & physical2]
                    trial = occupations[active].copy()
                    local = np.arange(len(active))
                    for cell in (cell1[active], cell2[active]):
                        c = 2 * cell
                        old_c = trial[local, c].copy()
                        trial[local, c] = trial[local, c + 1]
                        trial[local, c + 1] = old_c
                else:
                    # Exchange opposite physical orbitals at two cells.
                    orbital1 = occupations[rows, 2 * cell1]
                    orbital2 = occupations[rows, 2 * cell2]
                    active = rows[
                        (cell1 != cell2) & physical1 & physical2 & (orbital1 != orbital2)
                    ]
                    trial = occupations[active].copy()
                    local = np.arange(len(active))
                    c1, c2 = 2 * cell1[active], 2 * cell2[active]
                    old1 = trial[local, c1].copy()
                    old2 = trial[local, c2].copy()
                    trial[local, c1] = old2
                    trial[local, c1 + 1] = 1 - old2
                    trial[local, c2] = old1
                    trial[local, c2 + 1] = 1 - old1

            proposals += chains
            if not len(active):
                continue
            trial_amplitudes = interpolated_amplitudes(z, trial, reference, g)
            probability = np.abs(trial_amplitudes / amplitudes[active]) ** 2
            accept = rng.random(len(active)) < np.minimum(1.0, probability)
            if np.any(accept):
                chosen = active[accept]
                occupations[chosen] = trial[accept]
                amplitudes[chosen] = trial_amplitudes[accept]
                accepted += int(np.count_nonzero(accept))

        if sweep >= burn_in_sweeps:
            next_amplitudes = interpolated_amplitudes(z_next, occupations, reference, g)
            samples[sweep - burn_in_sweeps] = next_amplitudes / amplitudes

    return {
        "samples": samples,
        "acceptance_rate": accepted / proposals,
        "physical_fraction": float(np.mean(physical_configurations(occupations))),
        "zero_amplitude_fraction": float(np.mean(amplitudes == 0)),
    }


def compute_finite_g_mc_berry_phase(
    g: float,
    *,
    m: float = -2.0,
    t: float = 1.0,
    t1: float = 1.0,
    delta_p: float = 1.0,
    delta_s: float = 0.5,
    steps_per_leg: int = 2,
    lx: int = 4,
    ly: int = 4,
    reference_mask: Optional[int] = None,
    chains: int = 8,
    burn_in_sweeps: int = 50,
    measurement_sweeps: int = 250,
    seed: int = 20261102,
) -> dict:
    if not 0 <= g <= 1:
        raise ValueError("g must lie in [0,1]")
    started = time.perf_counter()
    model = TwoOrbitalModel(lx, ly, ModelParameters(t, t1, m, delta_p, delta_s))
    points = twist_loop(steps_per_leg)
    blocks = [_positive_blocks(model.bdg_hamiltonian(*point)) for point in points]
    reference_mask = _choose_reference(blocks, model.n_cells, reference_mask)
    states = [_make_state(u, v, gap, reference_mask) for u, v, gap in blocks]

    if g == 0:
        overlaps = [
            normalized_gaussian_overlap(state.z, states[(i + 1) % len(states)].z)
            for i, state in enumerate(states)
        ]
        phases = [float(np.angle(overlap)) for overlap in overlaps]
        return {
            "method": "exact Gaussian overlap",
            "g": g,
            "berry_phase": float(np.angle(np.exp(1j * math.fsum(phases)))),
            "statistical_error": 0.0,
            "elapsed_seconds": time.perf_counter() - started,
            "reference_mask": reference_mask,
        }

    reference = _reference_occupations(reference_mask, model.n_modes)
    links = []
    for index, state in enumerate(states):
        next_state = states[(index + 1) % len(states)]
        forward = _metropolis_finite_g(
            state.z, next_state.z, reference, g,
            chains=chains, burn_in_sweeps=burn_in_sweeps,
            measurement_sweeps=measurement_sweeps, seed=seed + 2 * index,
        )
        backward = _metropolis_finite_g(
            next_state.z, state.z, reference, g,
            chains=chains, burn_in_sweeps=burn_in_sweeps,
            measurement_sweeps=measurement_sweeps, seed=seed + 2 * index + 1,
        )
        a = _mean_and_autocorrelation_error(forward["samples"])
        b = _mean_and_autocorrelation_error(backward["samples"])
        links.append({
            "link": index,
            "phase": float(np.angle(a["mean"])),
            "phase_error": _phase_error(a),
            "overlap_magnitude": math.sqrt(abs(a["mean"] * b["mean"])),
            "forward_mean": [a["mean"].real, a["mean"].imag],
            "backward_mean": [b["mean"].real, b["mean"].imag],
            "forward_acceptance": forward["acceptance_rate"],
            "backward_acceptance": backward["acceptance_rate"],
            "forward_physical_fraction": forward["physical_fraction"],
            "backward_physical_fraction": backward["physical_fraction"],
        })
        print(
            f"g={g:g}, link {index + 1:02d}/{len(states)}: "
            f"phase={links[-1]['phase']:+.8f} +/- {links[-1]['phase_error']:.8f}, "
            f"|O|={links[-1]['overlap_magnitude']:.6f}",
            flush=True,
        )

    total = math.fsum(link["phase"] for link in links)
    error = math.sqrt(math.fsum(link["phase_error"] ** 2 for link in links))
    return {
        "method": "two-sided finite-g full-Fock-space Metropolis Monte Carlo",
        "g": g,
        "parameters": {
            "m": m, "t": t, "t1": t1, "delta_p": delta_p, "delta_s": delta_s,
            "lx": lx, "ly": ly, "steps_per_leg": steps_per_leg,
            "chains_per_direction": chains, "burn_in_sweeps": burn_in_sweeps,
            "measurement_sweeps": measurement_sweeps, "seed": seed,
            "reference_mask": reference_mask,
        },
        "berry_phase": float(np.angle(np.exp(1j * total))),
        "statistical_error": error,
        "elapsed_seconds": time.perf_counter() - started,
        "mean_acceptance_rate": float(np.mean([
            (link["forward_acceptance"] + link["backward_acceptance"]) / 2
            for link in links
        ])),
        "mean_physical_fraction": float(np.mean([
            (link["forward_physical_fraction"] + link["backward_physical_fraction"]) / 2
            for link in links
        ])),
        "links": links,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g", type=float, required=True)
    parser.add_argument("--m", type=float, default=-2.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--delta-p", type=float, default=1.0)
    parser.add_argument("--delta-s", type=float, default=0.5)
    parser.add_argument("--lx", type=int, default=4)
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--steps-per-leg", type=int, default=2)
    parser.add_argument("--reference-mask", type=lambda x: int(x, 0), default=None)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--burn-in-sweeps", type=int, default=50)
    parser.add_argument("--measurement-sweeps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = compute_finite_g_mc_berry_phase(
        args.g, m=args.m, t=args.t, t1=args.t1, delta_p=args.delta_p,
        delta_s=args.delta_s, steps_per_leg=args.steps_per_leg, lx=args.lx, ly=args.ly,
        reference_mask=args.reference_mask, chains=args.chains,
        burn_in_sweeps=args.burn_in_sweeps,
        measurement_sweeps=args.measurement_sweeps, seed=args.seed,
    )
    output = args.output or Path(__file__).resolve().parent / (
        f"finite_g_mc_g{args.g:g}_L{args.lx}x{args.ly}_M{args.steps_per_leg}_"
        f"C{args.chains}_S{args.measurement_sweeps}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "links"}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
