#!/usr/bin/env python3
"""Many-body Berry phase versus Gutzwiller projection strength.

Self-contained Monte Carlo driver requiring NumPy for computation.  JSON, CSV,
and SVG output use only the Python standard library.  If Matplotlib is installed,
a PNG error-bar plot is also written; Matplotlib is not needed for the calculation.

The interpolated state is |Psi_g> = [(1-g) + g P_G] |G>, up to normalization.
For every nonzero g, two-sided Metropolis sampling estimates each normalized
link overlap without explicitly calculating projected norm fractions.

Example
-------
python gutzwiller_mc_sweep_parallel.py \
    --g-values 0 0.25 0.5 0.75 0.9 1 \
    --lx 4 --ly 4 --chains 8 --samples 250 --burn-in 50 \
    --steps-per-leg 2 --m -2 --t 1 --t1 1 --delta-p 1 --delta-s 0.5 \
    --link-jobs 8 --output-prefix results/berry_sweep

Here --samples is the number of measurement sweeps per chain.  One sweep makes
N=lx*ly proposals per chain, and every link samples in both directions.
--link-jobs N parallelizes the 2*(4*steps_per_leg) independent ensembles for
each g value.  The parent process combines results and writes all output files.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path
import time
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Gaussian model and common Slater reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelParameters:
    t: float = 1.0
    t1: float = 1.0
    m: float = -2.0
    delta_p: float = 1.0
    delta_s: float = 0.5


@dataclass
class GaussianState:
    z: np.ndarray
    gap: float


class TwoOrbitalModel:
    def __init__(self, lx: int, ly: int, params: ModelParameters):
        if lx < 1 or ly < 1:
            raise ValueError("Lattice dimensions must be positive")
        self.lx, self.ly, self.params = int(lx), int(ly), params
        self.n_cells = self.lx * self.ly
        self.n_modes = 2 * self.n_cells

    def cell(self, x: int, y: int) -> int:
        return (y % self.ly) * self.lx + (x % self.lx)

    def mode(self, x: int, y: int, orbital: int) -> int:
        return 2 * self.cell(x, y) + orbital

    def bdg_hamiltonian(self, phi_x: float = 0.0, phi_y: float = 0.0) -> np.ndarray:
        p = self.params
        n = self.n_modes
        h = np.zeros((n, n), dtype=np.complex128)
        delta = np.zeros((n, n), dtype=np.complex128)

        def normal(i: int, j: int, amplitude: complex) -> None:
            h[i, j] += amplitude
            h[j, i] += np.conjugate(amplitude)

        def pair(i: int, j: int, amplitude: complex) -> None:
            delta[i, j] += amplitude
            delta[j, i] -= amplitude

        for y in range(self.ly):
            for x in range(self.lx):
                c, d = self.mode(x, y, 0), self.mode(x, y, 1)
                h[c, c] -= p.m
                h[d, d] += p.m
                for direction in ("x", "y"):
                    xp, yp = (x + 1, y) if direction == "x" else (x, y + 1)
                    cx, dx = self.mode(xp, yp, 0), self.mode(xp, yp, 1)
                    phi = phi_x if direction == "x" else phi_y
                    twisted = (x == 0) if direction == "x" else (y == 0)
                    phase = np.exp(1j * phi) if twisted else 1.0
                    # Eqs. (A5)-(A6): summing c_i^dagger c_j^dagger and
                    # c_j^dagger c_i^dagger gives pairing proportional to cos(phi).
                    pair_phase = np.cos(phi) if twisted else 1.0
                    normal(c, cx, -p.t * phase)
                    normal(d, dx, p.t * phase)
                    interorbital = -1j * p.t1 if direction == "x" else -p.t1
                    normal(c, dx, interorbital * phase)
                    reverse_sign = 1.0 if direction == "x" else -1.0
                    normal(d, cx, reverse_sign * interorbital * phase)
                    if direction == "x":
                        pair(c, cx, -1j * p.delta_p * pair_phase / 4)
                        pair(cx, c, 1j * p.delta_p * pair_phase / 4)
                        pair(d, dx, 1j * p.delta_p * pair_phase / 4)
                        pair(dx, d, -1j * p.delta_p * pair_phase / 4)
                    else:
                        pair(c, cx, -p.delta_p * pair_phase / 4)
                        pair(cx, c, p.delta_p * pair_phase / 4)
                        pair(d, dx, p.delta_p * pair_phase / 4)
                        pair(dx, d, -p.delta_p * pair_phase / 4)
                pair(c, d, -p.delta_s / 2)
                pair(d, c, p.delta_s / 2)
        h_bdg = np.block([[h, delta], [-delta.conj(), -h.T]])
        return 0.5 * (h_bdg + h_bdg.conj().T)


def twist_loop(steps_per_leg: int) -> np.ndarray:
    if steps_per_leg < 1:
        raise ValueError("steps_per_leg must be positive")
    vertices = np.array([
        [0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi],
        [2 * np.pi, np.pi], [2 * np.pi, 2 * np.pi],
    ])
    return np.array([
        (1.0 - step / steps_per_leg) * vertices[leg]
        + (step / steps_per_leg) * vertices[leg + 1]
        for leg in range(4) for step in range(steps_per_leg)
    ])


def positive_blocks(h_bdg: np.ndarray, tolerance: float = 1e-8):
    n = h_bdg.shape[0] // 2
    energies, vectors = np.linalg.eigh(h_bdg)
    positive = energies > tolerance
    if np.count_nonzero(positive) != n:
        raise ValueError("The BdG spectrum is gapless; Berry phase is undefined")
    vectors = vectors[:, positive]
    u, v = vectors[:n], vectors[n:]
    gap = float(min(np.min(energies[positive]), -np.max(energies[~positive])))
    return u.copy(), v.copy(), gap


def bit_count(value: int) -> int:
    return bin(int(value)).count("1")


def reference_condition(mask: int, blocks) -> float:
    worst = 0.0
    n_modes = blocks[0][0].shape[0]
    for u, v, _ in blocks:
        ud = u.copy()
        for mode in range(n_modes):
            if mask & (1 << mode):
                ud[mode] = v[mode]
        try:
            worst = max(worst, float(np.linalg.cond(ud)))
        except np.linalg.LinAlgError:
            return float("inf")
        if worst > 1e10:
            return worst
    return worst


def choose_reference(blocks, n_cells: int, requested: Optional[int]) -> int:
    n_modes = 2 * n_cells
    if requested is not None:
        mask = int(requested)
        if not 0 <= mask < (1 << n_modes):
            raise ValueError("reference_mask is inconsistent with the lattice")
        if bit_count(mask) % 2 != n_cells % 2:
            raise ValueError("reference_mask has incompatible fermion parity")
        condition = reference_condition(mask, blocks)
        if condition > 1e10:
            raise ValueError(f"reference_mask is ill-conditioned: {condition:.3e}")
        return mask

    full = (1 << n_modes) - 1
    candidates = [0, full]
    candidates += [sum(1 << i for i in range(parity, n_modes, 2)) for parity in (0, 1)]
    if n_modes == 32:
        candidates += [0xAAC8DA33, 0x5057F515, 0x08ABB8DB]
    candidates += [
        sum(1 << i for i in range(n_modes) if (v @ v.conj().T)[i, i].real > 0.5)
        for _, v, _ in blocks[:1]
    ]
    parity = n_cells % 2
    candidates = list(dict.fromkeys(mask for mask in candidates if bit_count(mask) % 2 == parity))
    for mask in candidates:
        if reference_condition(mask, blocks) < 1e4:
            return mask

    rng = np.random.default_rng(20261030)
    nbytes = (n_modes + 7) // 8
    best_mask, best_condition = None, float("inf")
    for _ in range(512):
        mask = int.from_bytes(rng.bytes(nbytes), "little") & full
        if bit_count(mask) % 2 != parity:
            continue
        condition = reference_condition(mask, blocks)
        if condition < best_condition:
            best_mask, best_condition = mask, condition
    if best_mask is None or best_condition > 1e10:
        raise ValueError("No regular common Slater reference was found")
    return best_mask


def make_state(u: np.ndarray, v: np.ndarray, gap: float, mask: int) -> GaussianState:
    ud, vd = u.copy(), v.copy()
    for mode in range(u.shape[0]):
        if mask & (1 << mode):
            ud[mode], vd[mode] = v[mode].copy(), u[mode].copy()
    z = -np.linalg.solve(ud.conj().T, vd.conj().T)
    return GaussianState(0.5 * (z - z.T), gap)


def normalized_gaussian_overlap(z1: np.ndarray, z2: np.ndarray) -> complex:
    identity = np.eye(z1.shape[0], dtype=np.complex128)
    determinant1 = np.linalg.det(identity + z1.conj().T @ z1)
    determinant2 = np.linalg.det(identity + z2.conj().T @ z2)
    cross = np.linalg.det(identity + z1.conj().T @ z2)
    norm1 = float(np.sqrt(determinant1.real))
    norm2 = float(np.sqrt(determinant2.real))
    return complex(np.sqrt(cross) / np.sqrt(norm1 * norm2))


# ---------------------------------------------------------------------------
# Fock-configuration amplitudes
# ---------------------------------------------------------------------------


def batched_pfaffian(matrices: np.ndarray) -> np.ndarray:
    """Skew-symmetric Pfaffians with batch-specific LTL pivoting."""
    a = np.array(matrices, dtype=np.complex128, copy=True)
    batch, n, _ = a.shape
    if n == 0:
        return np.ones(batch, dtype=np.complex128)
    if n % 2:
        return np.zeros(batch, dtype=np.complex128)
    if n == 2:
        return a[:, 0, 1].copy()
    result = np.ones(batch, dtype=np.complex128)
    rows = np.arange(batch)
    for k in range(0, n - 1, 2):
        pivots = k + 1 + np.argmax(np.abs(a[:, k, k + 1:]), axis=1)
        nonzero = np.abs(a[rows, k, pivots]) >= 1e-30
        result[~nonzero] = 0.0
        swapped = pivots != k + 1
        if np.any(swapped):
            old_rows = a[rows, k + 1, :].copy()
            a[rows, k + 1, :] = a[rows, pivots, :]
            a[rows, pivots, :] = old_rows
            old_columns = a[rows, :, k + 1].copy()
            a[rows, :, k + 1] = a[rows, :, pivots]
            a[rows, :, pivots] = old_columns
            result[swapped] *= -1
        pivot_value = a[:, k, k + 1].copy()
        result *= pivot_value
        inverse = np.zeros_like(pivot_value)
        np.divide(1.0, pivot_value, out=inverse, where=nonzero)
        x = a[:, k, k + 2:].copy()
        y = a[:, k + 1, k + 2:].copy()
        for i_local in range(n - k - 2):
            i = k + 2 + i_local
            for j in range(i + 1, n):
                j_local = j - (k + 2)
                updated = a[:, i, j] + (
                    y[:, i_local] * x[:, j_local] - x[:, i_local] * y[:, j_local]
                ) * inverse
                a[:, i, j] = updated
                a[:, j, i] = -updated
    return result


def reference_occupations(reference: int, n_modes: int) -> np.ndarray:
    return np.array([(reference >> mode) & 1 for mode in range(n_modes)], dtype=np.int8)


def original_occupations(spins: np.ndarray) -> np.ndarray:
    chains, n_cells = spins.shape
    original = np.zeros((chains, 2 * n_cells), dtype=np.int8)
    rows = np.arange(chains)[:, None]
    cells = np.arange(n_cells)[None, :]
    original[rows, 2 * cells + spins] = 1
    return original


def transformation_signs(original: np.ndarray, reference: np.ndarray) -> np.ndarray:
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


def gaussian_amplitudes(z, occupations, reference) -> np.ndarray:
    transformed = occupations != reference[None, :]
    counts = np.sum(transformed, axis=1)
    signs = transformation_signs(occupations, reference)
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
        amplitudes[rows] = signs[rows] * batched_pfaffian(matrices)
    return amplitudes


def physical_configurations(occupations: np.ndarray) -> np.ndarray:
    n_cells = occupations.shape[1] // 2
    occupancy = occupations.reshape(len(occupations), n_cells, 2).sum(axis=2)
    return np.all(occupancy == 1, axis=1)


def interpolated_amplitudes(z, occupations, reference, g: float) -> np.ndarray:
    amplitudes = gaussian_amplitudes(z, occupations, reference)
    if g != 0:
        physical = physical_configurations(occupations)
        amplitudes[~physical] *= 1.0 - g
    return amplitudes


# ---------------------------------------------------------------------------
# Metropolis sampling and statistical analysis
# ---------------------------------------------------------------------------


def metropolis_finite_g(
    z, z_next, reference, g: float, *, chains: int, burn_in: int,
    samples: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n_modes = z.shape[0]
    n_cells = n_modes // 2
    spins = rng.integers(0, 2, size=(chains, n_cells), dtype=np.int8)
    occupations = original_occupations(spins)
    amplitudes = interpolated_amplitudes(z, occupations, reference, g)
    rows = np.arange(chains)
    ratios = np.empty((samples, chains), dtype=np.complex128)
    accepted = 0
    proposals = 0

    for sweep in range(burn_in + samples):
        for _ in range(n_cells):
            move = rng.random()
            first = rng.integers(0, n_modes, size=chains)
            second = rng.integers(0, n_modes, size=chains)
            cell1, cell2 = first // 2, second // 2
            occ1 = occupations[rows, first]
            occ2 = occupations[rows, second]

            if move < 0.25:
                active = rows[(occ1 == 1) & (occ2 == 0)]
                i, j = first[active], second[active]
                trial = occupations[active].copy()
                local = np.arange(len(active))
                trial[local, i] = 0
                trial[local, j] = 1
            elif move < 0.50:
                active = rows[(first != second) & (occ1 == occ2)]
                i, j = first[active], second[active]
                trial = occupations[active].copy()
                local = np.arange(len(active))
                trial[local, i] = 1 - trial[local, i]
                trial[local, j] = 1 - trial[local, j]
            else:
                physical1 = occupations[rows, 2 * cell1] + occupations[rows, 2 * cell1 + 1] == 1
                physical2 = occupations[rows, 2 * cell2] + occupations[rows, 2 * cell2 + 1] == 1
                if move < 0.65:
                    active = rows[physical1]
                    trial = occupations[active].copy()
                    c = 2 * cell1[active]
                    local = np.arange(len(active))
                    old_c = trial[local, c].copy()
                    trial[local, c] = trial[local, c + 1]
                    trial[local, c + 1] = old_c
                elif move < 0.85:
                    active = rows[(cell1 != cell2) & physical1 & physical2]
                    trial = occupations[active].copy()
                    local = np.arange(len(active))
                    for cell in (cell1[active], cell2[active]):
                        c = 2 * cell
                        old_c = trial[local, c].copy()
                        trial[local, c] = trial[local, c + 1]
                        trial[local, c + 1] = old_c
                else:
                    orbital1 = occupations[rows, 2 * cell1]
                    orbital2 = occupations[rows, 2 * cell2]
                    active = rows[(cell1 != cell2) & physical1 & physical2 & (orbital1 != orbital2)]
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

        if sweep >= burn_in:
            next_amplitudes = interpolated_amplitudes(z_next, occupations, reference, g)
            ratios[sweep - burn_in] = next_amplitudes / amplitudes

    return {
        "samples": ratios,
        "acceptance_rate": accepted / proposals,
        "physical_fraction": float(np.mean(physical_configurations(occupations))),
    }


def mean_and_autocorrelation_error(samples: np.ndarray) -> dict:
    series = np.mean(samples, axis=1)
    mean = complex(np.mean(series))
    n = len(series)
    errors, taus = [], []
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
        tau = min(tau, n / 4.0)
        errors.append(math.sqrt(max(0.0, 2.0 * tau * variance / n)))
        taus.append(float(tau))
    return {"mean": mean, "real_error": errors[0], "imag_error": errors[1],
            "real_tau": taus[0], "imag_tau": taus[1]}


def phase_error(a: dict) -> float:
    value = a["mean"]
    magnitude_squared = abs(value) ** 2
    if magnitude_squared <= 1e-30:
        return float("inf")
    variance = (
        value.imag ** 2 * a["real_error"] ** 2
        + value.real ** 2 * a["imag_error"] ** 2
    ) / magnitude_squared ** 2
    return math.sqrt(variance)


def sampling_worker(task: tuple) -> tuple:
    """Run one independent link/direction ensemble in a separate process."""
    index, direction, z, z_next, reference, g, chains, burn_in, samples, seed = task
    result = metropolis_finite_g(
        z, z_next, reference, g, chains=chains,
        burn_in=burn_in, samples=samples, seed=seed,
    )
    return index, direction, result


def compute_berry_phase(
    g: float, states, points, reference_mask: int, *, chains: int,
    burn_in: int, samples: int, seed: int, link_jobs: int = 1,
) -> dict:
    """Compute one Berry phase, optionally parallelizing 2*len(states) ensembles."""
    if link_jobs < 1:
        raise ValueError("link_jobs must be a positive integer")
    started = time.perf_counter()
    if g == 0:
        overlaps = [
            normalized_gaussian_overlap(state.z, states[(i + 1) % len(states)].z)
            for i, state in enumerate(states)
        ]
        total = math.fsum(float(np.angle(overlap)) for overlap in overlaps)
        return {
            "g": g, "berry_phase": float(np.angle(np.exp(1j * total))),
            "statistical_error": 0.0, "elapsed_seconds": time.perf_counter() - started,
            "method": "exact Gaussian overlap",
        }

    reference = reference_occupations(reference_mask, states[0].z.shape[0])
    tasks = []
    for index, state in enumerate(states):
        next_state = states[(index + 1) % len(states)]
        tasks.append((
            index, 0, state.z, next_state.z, reference, g,
            chains, burn_in, samples, seed + 2 * index,
        ))
        tasks.append((
            index, 1, next_state.z, state.z, reference, g,
            chains, burn_in, samples, seed + 2 * index + 1,
        ))

    print(f"  Sampling {len(tasks)} independent ensembles with {link_jobs} workers", flush=True)
    if link_jobs == 1:
        sampling_results = [sampling_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=link_jobs) as executor:
            sampling_results = list(executor.map(sampling_worker, tasks))
    ensembles = {(index, direction): result for index, direction, result in sampling_results}

    links = []
    for index in range(len(states)):
        forward = ensembles[(index, 0)]
        backward = ensembles[(index, 1)]
        a = mean_and_autocorrelation_error(forward["samples"])
        b = mean_and_autocorrelation_error(backward["samples"])
        links.append({
            "link": index,
            "phase": float(np.angle(a["mean"])),
            "phase_error": phase_error(a),
            "overlap_magnitude": math.sqrt(abs(a["mean"] * b["mean"])),
            "forward_acceptance": forward["acceptance_rate"],
            "backward_acceptance": backward["acceptance_rate"],
            "forward_physical_fraction": forward["physical_fraction"],
            "backward_physical_fraction": backward["physical_fraction"],
        })
        print(
            f"  link {index + 1:02d}/{len(states)}: "
            f"phase={links[-1]['phase']:+.7f} +/- {links[-1]['phase_error']:.7f}",
            flush=True,
        )

    total = math.fsum(link["phase"] for link in links)
    error = math.sqrt(math.fsum(link["phase_error"] ** 2 for link in links))
    return {
        "g": g,
        "berry_phase": float(np.angle(np.exp(1j * total))),
        "statistical_error": error,
        "elapsed_seconds": time.perf_counter() - started,
        "mean_acceptance_rate": float(np.mean([
            (link["forward_acceptance"] + link["backward_acceptance"]) / 2 for link in links
        ])),
        "mean_physical_fraction": float(np.mean([
            (link["forward_physical_fraction"] + link["backward_physical_fraction"]) / 2
            for link in links
        ])),
        "method": f"two-sided full-Fock-space Metropolis Monte Carlo ({link_jobs} link workers)",
        "links": links,
    }


# ---------------------------------------------------------------------------
# Output and plotting
# ---------------------------------------------------------------------------


def save_results(prefix: Path, parameters: dict, results: list, complete: bool) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    payload = {"parameters": parameters, "complete": complete, "results": results}
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    fields = ["g", "berry_phase", "statistical_error", "elapsed_seconds",
              "mean_acceptance_rate", "mean_physical_fraction", "method"]
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    save_svg(prefix.with_suffix(".svg"), results)


def save_svg(path: Path, results: list) -> None:
    """Dependency-free error-bar plot."""
    if not results:
        return
    width, height = 800, 560
    left, right, top, bottom = 90, 30, 55, 75
    plot_w, plot_h = width - left - right, height - top - bottom
    x = np.array([r["g"] for r in results], dtype=float)
    y = np.array([r["berry_phase"] for r in results], dtype=float)
    e = np.array([r["statistical_error"] for r in results], dtype=float)
    xmin, xmax = min(0.0, float(x.min())), max(1.0, float(x.max()))
    ymin, ymax = float(np.min(y - e)), float(np.max(y + e))
    padding = max(0.05 * (ymax - ymin), 0.01)
    ymin, ymax = ymin - padding, ymax + padding

    def sx(value):
        return left + (value - xmin) / (xmax - xmin) * plot_w

    def sy(value):
        return top + (ymax - value) / (ymax - ymin) * plot_h

    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="400" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Many-body Berry phase vs projection strength</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="black"/>',
        '<text x="400" y="535" text-anchor="middle" font-family="sans-serif" font-size="16">Gutzwiller projection strength g</text>',
        '<text x="22" y="280" text-anchor="middle" font-family="sans-serif" font-size="16" transform="rotate(-90 22 280)">Berry phase (rad)</text>',
    ]
    for tick in np.linspace(xmin, xmax, 6):
        tx = sx(tick)
        items += [f'<line x1="{tx:.2f}" y1="{top + plot_h}" x2="{tx:.2f}" y2="{top + plot_h + 6}" stroke="black"/>',
                  f'<text x="{tx:.2f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="sans-serif" font-size="13">{tick:.2f}</text>']
    for tick in np.linspace(ymin, ymax, 6):
        ty = sy(tick)
        items += [f'<line x1="{left - 6}" y1="{ty:.2f}" x2="{left}" y2="{ty:.2f}" stroke="black"/>',
                  f'<text x="{left - 10}" y="{ty + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{tick:.3f}</text>',
                  f'<line x1="{left}" y1="{ty:.2f}" x2="{left + plot_w}" y2="{ty:.2f}" stroke="#dddddd"/>']
    points = []
    for xv, yv, ev in zip(x, y, e):
        px, py = sx(xv), sy(yv)
        points.append(f"{px:.2f},{py:.2f}")
        items += [f'<line x1="{px:.2f}" y1="{sy(yv - ev):.2f}" x2="{px:.2f}" y2="{sy(yv + ev):.2f}" stroke="#1f77b4" stroke-width="2"/>',
                  f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="#1f77b4"/>']
    if points:
        items.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1f77b4" stroke-width="1.5"/>')
    items.append('</svg>')
    path.write_text("\n".join(items) + "\n")


def save_png(path: Path, results: list) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    g = [r["g"] for r in results]
    phase = [r["berry_phase"] for r in results]
    error = [r["statistical_error"] for r in results]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(g, phase, yerr=error, fmt="o-", capsize=4, color="#1f77b4")
    ax.set_xlabel("Gutzwiller projection strength $g$")
    ax.set_ylabel("Many-body Berry phase (rad)")
    ax.set_title("Berry phase versus projection strength")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Command-line driver
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--g-values", type=float, nargs="+", required=True)
    parser.add_argument("--lx", type=int, default=4)
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--chains", type=int, default=8, help="Independent chains per direction")
    parser.add_argument("--samples", type=int, default=250, help="Measurement sweeps per chain")
    parser.add_argument("--burn-in", type=int, default=50, help="Burn-in sweeps per chain")
    parser.add_argument("--steps-per-leg", type=int, default=2)
    parser.add_argument("--m", type=float, default=-2.0)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--delta-p", type=float, default=1.0)
    parser.add_argument("--delta-s", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--reference-mask", type=lambda x: int(x, 0), default=None)
    parser.add_argument("--link-jobs", type=int, default=1,
                        help="Worker processes for independent link/direction ensembles")
    parser.add_argument("--output-prefix", type=Path, default=Path("berry_sweep"))
    args = parser.parse_args()

    if min(args.chains, args.samples, args.burn_in, args.steps_per_leg, args.link_jobs) < 1:
        parser.error("chains, samples, burn-in, steps-per-leg, and link-jobs must be positive")
    g_values = np.asarray(args.g_values, dtype=float)
    if np.any(~np.isfinite(g_values)) or np.any((g_values < 0) | (g_values > 1)):
        parser.error("all g values must be finite numbers in [0,1]")

    model_parameters = ModelParameters(args.t, args.t1, args.m, args.delta_p, args.delta_s)
    model = TwoOrbitalModel(args.lx, args.ly, model_parameters)
    points = twist_loop(args.steps_per_leg)
    blocks = [positive_blocks(model.bdg_hamiltonian(*point)) for point in points]
    reference = choose_reference(blocks, model.n_cells, args.reference_mask)
    states = [make_state(u, v, gap, reference) for u, v, gap in blocks]
    parameters = {
        "model": asdict(model_parameters), "lx": args.lx, "ly": args.ly,
        "steps_per_leg": args.steps_per_leg, "chains_per_direction": args.chains,
        "measurement_sweeps_per_chain": args.samples, "burn_in_sweeps": args.burn_in,
        "seed": args.seed, "reference_mask": reference, "g_values": g_values.tolist(),
        "link_worker_processes": args.link_jobs,
    }

    results = []
    save_results(args.output_prefix, parameters, results, complete=False)
    for index, g in enumerate(g_values):
        print(f"Starting g={g:g} ({index + 1}/{len(g_values)})", flush=True)
        result = compute_berry_phase(
            float(g), states, points, reference, chains=args.chains,
            burn_in=args.burn_in, samples=args.samples, seed=args.seed + 1000 * index,
            link_jobs=args.link_jobs,
        )
        results.append(result)
        save_results(args.output_prefix, parameters, results, complete=False)
        print(
            f"g={g:g}: phase={result['berry_phase']:+.9f} "
            f"+/- {result['statistical_error']:.9f}; "
            f"elapsed={result['elapsed_seconds']:.1f} s",
            flush=True,
        )

    save_results(args.output_prefix, parameters, results, complete=True)
    png_written = save_png(args.output_prefix.with_suffix(".png"), results)
    print(f"Wrote {args.output_prefix.with_suffix('.json')}")
    print(f"Wrote {args.output_prefix.with_suffix('.csv')}")
    print(f"Wrote {args.output_prefix.with_suffix('.svg')}")
    if png_written:
        print(f"Wrote {args.output_prefix.with_suffix('.png')}")
    else:
        print("Matplotlib not installed: SVG plot is available; PNG was skipped.")


if __name__ == "__main__":
    main()
