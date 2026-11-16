"""Metropolis Monte Carlo Berry phase of a fully parton-projected Gaussian state.

The physical Hilbert space has exactly one parton in every unit cell.  Symmetric
single-cell flips, double-cell flips, and c/d swaps preserve the projection and
connect all 2^N spin configurations.  Only NumPy is required.

For projected amplitudes b_i(S), sampled from p_i(S) proportional to |b_i(S)|^2,
    A = E_{p_1}[b_2/b_1] = N_12/N_11,
    B = E_{p_2}[b_1/b_2] = N_21/N_22.
Consequently |O_12| = sqrt(|A B|) and arg(O_12) = arg(A), without knowing the
individual projected norms N_11 and N_22.

Example
-------
python monte_carlo/berry_phase_mc.py --steps-per-leg 2 --chains 16 \
    --burn-in-sweeps 20 --measurement-sweeps 80
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

# Allow execution either as a script or as a module from the workspace root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server_ready.berry_phase import (  # noqa: E402
    ModelParameters,
    TwoOrbitalModel,
    _batched_pfaffian,
    _choose_reference,
    _make_state,
    _positive_blocks,
    twist_loop,
)


def _reference_occupations(reference: int, n_modes: int) -> np.ndarray:
    return np.array([(reference >> mode) & 1 for mode in range(n_modes)], dtype=np.int8)


def _original_occupations(spins: np.ndarray) -> np.ndarray:
    """Map spin values 0/1 to occupied c/d fermionic modes."""
    chains, n_cells = spins.shape
    original = np.zeros((chains, 2 * n_cells), dtype=np.int8)
    rows = np.arange(chains)[:, None]
    cells = np.arange(n_cells)[None, :]
    original[rows, 2 * cells + spins] = 1
    return original


def _transformation_signs(original: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Fermionic signs for excitations relative to a common Slater reference."""
    chains, n_modes = original.shape
    state = np.zeros_like(original)
    signs = np.ones(chains, dtype=np.float64)

    for mode in range(n_modes):
        if reference[mode]:
            parity = np.sum(state[:, :mode], axis=1) % 2
            signs *= 1.0 - 2.0 * parity
            state[:, mode] = 1

    for mode in range(n_modes - 1, -1, -1):
        parity = np.sum(state[:, :mode], axis=1) % 2
        active = (original[:, mode] != 0) & (parity != 0)
        signs[active] *= -1.0
        if reference[mode]:
            state[:, mode] = np.where(original[:, mode] != 0, 0, state[:, mode])
        else:
            state[:, mode] = np.where(original[:, mode] != 0, 1, state[:, mode])
    return signs


def amplitudes_for_spins(
    z: np.ndarray, spins: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Projected Gaussian amplitudes for a batch of spin configurations.

    Spins are stored as an integer array rather than a fixed-width bit mask, so
    the same routine also applies beyond 64 unit cells.  An arbitrary common
    normalization of the amplitudes is allowed and cancels in all MC ratios.
    """
    original = _original_occupations(spins)
    transformed = (original != reference[None, :])
    counts = np.sum(transformed, axis=1)
    signs = _transformation_signs(original, reference)
    amplitudes = np.empty(len(spins), dtype=np.complex128)

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


def _metropolis_ensemble(
    z: np.ndarray,
    z_next: np.ndarray,
    reference: np.ndarray,
    *,
    chains: int,
    burn_in_sweeps: int,
    measurement_sweeps: int,
    seed: int,
) -> dict:
    """Sample one state and measure ratios to its following state."""
    rng = np.random.default_rng(seed)
    n_cells = z.shape[0] // 2
    # Random initial configurations are valid once single-cell flips make the
    # dynamics ergodic; independent chains and burn-in remove initialization bias.
    spins = rng.integers(0, 2, size=(chains, n_cells), dtype=np.int8)
    amplitudes = amplitudes_for_spins(z, spins, reference)
    accepted = 0
    proposals = 0
    samples = np.empty((measurement_sweeps, chains), dtype=np.complex128)
    rows = np.arange(chains)

    for sweep in range(burn_in_sweeps + measurement_sweeps):
        for _ in range(n_cells):
            # Single flips change parity, double flips make the often-important
            # Delta-N_d=+-2 transitions, and swaps improve spatial diffusion.
            # All three proposals are symmetric; single flips ensure ergodicity.
            move = rng.random()
            first = rng.integers(0, n_cells, size=chains)
            second = rng.integers(0, n_cells, size=chains)
            if move < 1.0 / 3.0:
                active = rows
                trial = spins.copy()
                trial[rows, first] = 1 - trial[rows, first]
            elif move < 2.0 / 3.0:
                active = rows[first != second]
                selected = np.arange(len(active))
                trial = spins[active].copy()
                i, j = first[first != second], second[first != second]
                trial[selected, i] = 1 - trial[selected, i]
                trial[selected, j] = 1 - trial[selected, j]
            else:
                movable = spins[rows, first] != spins[rows, second]
                active = rows[movable]
                selected = np.arange(len(active))
                trial = spins[active].copy()
                i, j = first[movable], second[movable]
                trial[selected, i], trial[selected, j] = (
                    spins[active, j], spins[active, i]
                )
            proposals += chains
            if not len(active):
                continue
            trial_amplitudes = amplitudes_for_spins(z, trial, reference)
            probability = np.abs(trial_amplitudes / amplitudes[active]) ** 2
            accept = rng.random(len(active)) < np.minimum(1.0, probability)
            if np.any(accept):
                chosen = active[accept]
                spins[chosen] = trial[accept]
                amplitudes[chosen] = trial_amplitudes[accept]
                accepted += int(np.count_nonzero(accept))

        if sweep >= burn_in_sweeps:
            next_amplitudes = amplitudes_for_spins(z_next, spins, reference)
            samples[sweep - burn_in_sweeps] = next_amplitudes / amplitudes

    return {
        "samples": samples,
        "acceptance_rate": accepted / proposals,
        "zero_amplitude_fraction": float(np.mean(amplitudes == 0)),
    }


def _mean_and_autocorrelation_error(samples: np.ndarray) -> dict:
    """Complex mean and initial-positive-sequence autocorrelation errors.

    The time series is the independent-chain average at each sweep.  Its
    autocorrelation therefore includes both chain-to-chain and sweep-to-sweep
    fluctuations.  The returned errors are for the mean over all sweeps/chains.
    """
    series = np.mean(samples, axis=1)
    mean = complex(np.mean(series))
    n = len(series)
    errors = []
    taus = []
    for values in (series.real, series.imag):
        centered = values - np.mean(values)
        variance = float(np.mean(centered * centered))
        if n < 4 or variance <= 1e-30:
            errors.append(float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
            taus.append(0.5)
            continue
        autocovariance = np.correlate(centered, centered, mode="full")[n - 1:] / n
        rho = autocovariance / autocovariance[0]
        tau = 0.5
        for lag in range(1, n - 1, 2):
            pair = float(rho[lag] + rho[lag + 1])
            if pair <= 0:
                break
            tau += pair
        # Avoid an unstable error estimate when the series is too short.
        tau = min(tau, n / 4.0)
        errors.append(math.sqrt(max(0.0, 2.0 * tau * variance / n)))
        taus.append(float(tau))
    return {
        "mean": mean,
        "real_error": errors[0],
        "imag_error": errors[1],
        "real_tau": taus[0],
        "imag_tau": taus[1],
    }


def _phase_error(a: dict) -> float:
    value = a["mean"]
    magnitude_squared = abs(value) ** 2
    if magnitude_squared <= 1e-30:
        return float("inf")
    variance = (
        value.imag ** 2 * a["real_error"] ** 2
        + value.real ** 2 * a["imag_error"] ** 2
    ) / magnitude_squared ** 2
    return math.sqrt(variance)


def compute_mc_berry_phase(
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
    chains: int = 16,
    burn_in_sweeps: int = 20,
    measurement_sweeps: int = 80,
    seed: int = 20261102,
) -> dict:
    """Return the MC Berry phase, statistical error, and per-link diagnostics."""
    if min(chains, burn_in_sweeps, measurement_sweeps) < 1:
        raise ValueError("chains and sweep counts must be positive integers")
    started = time.perf_counter()
    model = TwoOrbitalModel(lx, ly, ModelParameters(t, t1, m, delta_p, delta_s))
    points = twist_loop(steps_per_leg)
    blocks = [_positive_blocks(model.bdg_hamiltonian(*point)) for point in points]
    reference_mask = _choose_reference(blocks, model.n_cells, reference_mask)
    states = [_make_state(u, v, gap, reference_mask) for u, v, gap in blocks]
    reference = _reference_occupations(reference_mask, model.n_modes)

    links = []
    for index, state in enumerate(states):
        next_state = states[(index + 1) % len(states)]
        forward = _metropolis_ensemble(
            state.z, next_state.z, reference,
            chains=chains, burn_in_sweeps=burn_in_sweeps,
            measurement_sweeps=measurement_sweeps,
            seed=seed + 2 * index,
        )
        backward = _metropolis_ensemble(
            next_state.z, state.z, reference,
            chains=chains, burn_in_sweeps=burn_in_sweeps,
            measurement_sweeps=measurement_sweeps,
            seed=seed + 2 * index + 1,
        )
        a = _mean_and_autocorrelation_error(forward["samples"])
        b = _mean_and_autocorrelation_error(backward["samples"])
        magnitude = math.sqrt(abs(a["mean"] * b["mean"]))
        links.append({
            "link": index,
            "twist_1": points[index].tolist(),
            "twist_2": points[(index + 1) % len(points)].tolist(),
            "phase": float(np.angle(a["mean"])),
            "phase_error": _phase_error(a),
            "overlap_magnitude": magnitude,
            "forward_mean": [a["mean"].real, a["mean"].imag],
            "backward_mean": [b["mean"].real, b["mean"].imag],
            "forward_real_error": a["real_error"],
            "forward_imag_error": a["imag_error"],
            "forward_real_tau": a["real_tau"],
            "forward_imag_tau": a["imag_tau"],
            "forward_acceptance": forward["acceptance_rate"],
            "backward_acceptance": backward["acceptance_rate"],
            "forward_zero_amplitude_fraction": forward["zero_amplitude_fraction"],
            "backward_zero_amplitude_fraction": backward["zero_amplitude_fraction"],
        })
        print(
            f"link {index + 1:02d}/{len(states)}: "
            f"phase={links[-1]['phase']:+.8f} +/- {links[-1]['phase_error']:.8f}, "
            f"|O|={magnitude:.6f}",
            flush=True,
        )

    total = math.fsum(link["phase"] for link in links)
    error = math.sqrt(math.fsum(link["phase_error"] ** 2 for link in links))
    return {
        "method": "two-sided Metropolis Monte Carlo",
        "parameters": {
            "m": m, "t": t, "t1": t1, "delta_p": delta_p, "delta_s": delta_s,
            "lx": lx, "ly": ly, "steps_per_leg": steps_per_leg,
            "chains_per_direction": chains,
            "burn_in_sweeps": burn_in_sweeps,
            "measurement_sweeps": measurement_sweeps,
            "seed": seed,
            "reference_mask": reference_mask,
        },
        "berry_phase": float(np.angle(np.exp(1j * total))),
        "statistical_error": error,
        "elapsed_seconds": time.perf_counter() - started,
        "minimum_overlap_magnitude": min(link["overlap_magnitude"] for link in links),
        "mean_acceptance_rate": float(np.mean([
            (link["forward_acceptance"] + link["backward_acceptance"]) / 2
            for link in links
        ])),
        "links": links,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=float, default=-2.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--delta-p", type=float, default=1.0)
    parser.add_argument("--delta-s", type=float, default=0.5)
    parser.add_argument("--lx", type=int, default=4)
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--steps-per-leg", type=int, default=2)
    parser.add_argument("--reference-mask", type=lambda x: int(x, 0), default=None)
    parser.add_argument("--chains", type=int, default=16)
    parser.add_argument("--burn-in-sweeps", type=int, default=20)
    parser.add_argument("--measurement-sweeps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = compute_mc_berry_phase(
        m=args.m, t=args.t, t1=args.t1, delta_p=args.delta_p, delta_s=args.delta_s,
        lx=args.lx, ly=args.ly, steps_per_leg=args.steps_per_leg,
        reference_mask=args.reference_mask, chains=args.chains,
        burn_in_sweeps=args.burn_in_sweeps,
        measurement_sweeps=args.measurement_sweeps, seed=args.seed,
    )
    output = args.output
    if output is None:
        output = Path(__file__).resolve().parent / (
            f"mc_m{args.m:g}_L{args.lx}x{args.ly}_M{args.steps_per_leg}_"
            f"C{args.chains}_S{args.measurement_sweeps}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "links"}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
