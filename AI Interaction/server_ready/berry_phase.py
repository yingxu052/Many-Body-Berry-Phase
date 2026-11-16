"""Many-body Berry phases for interpolated Gutzwiller-projected Gaussian states.

Only NumPy is required. The physical projection is
    P_G = prod_i n_i (2 - n_i),  n_i = n_ic + n_id,
and the interpolation is
    |Psi(g)> proportional to (1-g) |Psi_G> + g P_G |Psi_G>.
Thus g=0 is unprojected and g=1 is fully projected.

Main API
--------
compute_berry_phases(g_values, m=-2, t=1, t1=1, delta_p=1, delta_s=0.5,
                     steps_per_leg=8, lx=4, ly=4)

The returned NumPy array has the same shape as ``g_values``. The Hamiltonian
uses Eqs. (A1)--(A6): normal hopping is complex on twisted seams, while summing
both pairing-operator orderings gives pairing proportional to cos(phi).
Exact enumeration is performed in the 2**(lx*ly) physical spin Hilbert space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


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
    u: np.ndarray
    v: np.ndarray
    gap: float
    full_norm: float


class TwoOrbitalModel:
    def __init__(self, lx: int, ly: int, params: ModelParameters):
        if lx < 1 or ly < 1:
            raise ValueError("The lattice dimensions must be positive integers")
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
        raise ValueError("steps_per_leg must be a positive integer")
    vertices = np.array([
        [0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi],
        [2 * np.pi, np.pi], [2 * np.pi, 2 * np.pi],
    ])
    return np.array([
        (1.0 - step / steps_per_leg) * vertices[leg]
        + (step / steps_per_leg) * vertices[leg + 1]
        for leg in range(4) for step in range(steps_per_leg)
    ])


def _positive_blocks(h_bdg: np.ndarray, tolerance: float = 1e-8):
    n = h_bdg.shape[0] // 2
    energies, vectors = np.linalg.eigh(h_bdg)
    positive = energies > tolerance
    if np.count_nonzero(positive) != n:
        raise ValueError("The BdG spectrum is gapless; Berry phase is undefined")
    vectors = vectors[:, positive]
    u, v = vectors[:n], vectors[n:]
    # The common Slater reference, rather than an arbitrary even-parity vacuum,
    # determines the represented Gaussian parity.  Odd-parity states are needed
    # for projected lattices with an odd number of unit cells.
    gap = float(min(np.min(energies[positive]), -np.max(energies[~positive])))
    return u.copy(), v.copy(), gap


def _reference_condition(mask: int, blocks) -> float:
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


def _bit_count(value: int) -> int:
    """Python-version-independent population count."""
    return bin(int(value)).count("1")


def _choose_reference(blocks, n_cells: int, requested: Optional[int]) -> int:
    n_modes = 2 * n_cells
    if requested is not None:
        mask = int(requested)
        if not 0 <= mask < (1 << n_modes):
            raise ValueError("reference_mask is inconsistent with the lattice")
        if _bit_count(mask) % 2 != n_cells % 2:
            raise ValueError("reference_mask has incompatible fermion parity")
        condition = _reference_condition(mask, blocks)
        if condition > 1e10:
            raise ValueError(f"reference_mask is ill-conditioned: cond={condition:.3e}")
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
    candidates = list(dict.fromkeys(mask for mask in candidates if _bit_count(mask) % 2 == parity))

    # Use a known good reference immediately when it is well conditioned.
    for mask in candidates:
        condition = _reference_condition(mask, blocks)
        if condition < 1e4:
            return mask

    if n_modes <= 12:
        search_masks = range(1 << n_modes)
    else:
        rng = np.random.default_rng(20261030)
        search_masks = (int(x) for x in rng.integers(0, 1 << n_modes, size=512, dtype=np.uint64))
    best_mask, best_condition = None, float("inf")
    for mask in search_masks:
        if _bit_count(mask) % 2 != parity:
            continue
        condition = _reference_condition(mask, blocks)
        if condition < best_condition:
            best_mask, best_condition = mask, condition
    if best_mask is None or best_condition > 1e10:
        raise ValueError("No regular common Slater reference was found")
    return best_mask


def _make_state(u: np.ndarray, v: np.ndarray, gap: float, mask: int) -> GaussianState:
    n_modes = u.shape[0]
    ud, vd = u.copy(), v.copy()
    for mode in range(n_modes):
        if mask & (1 << mode):
            ud[mode], vd[mode] = v[mode].copy(), u[mode].copy()
    z = -np.linalg.solve(ud.conj().T, vd.conj().T)
    z = 0.5 * (z - z.T)
    identity = np.eye(n_modes, dtype=np.complex128)
    full_norm = float(np.sqrt(np.linalg.det(identity + z.conj().T @ z).real))
    if not np.isfinite(full_norm) or full_norm <= 0:
        raise ValueError("Invalid Gaussian norm")
    return GaussianState(z, u, v, gap, full_norm)


def normalized_gaussian_overlap(z1: np.ndarray, z2: np.ndarray) -> complex:
    identity = np.eye(z1.shape[0], dtype=np.complex128)
    determinant1 = np.linalg.det(identity + z1.conj().T @ z1)
    determinant2 = np.linalg.det(identity + z2.conj().T @ z2)
    cross = np.linalg.det(identity + z1.conj().T @ z2)
    norm1 = float(np.sqrt(determinant1.real))
    norm2 = float(np.sqrt(determinant2.real))
    return complex(np.sqrt(cross) / np.sqrt(norm1 * norm2))


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int8)


def _popcount32(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.uint64)
    return sum(_POPCOUNT[((value >> shift) & 0xFF).astype(np.uint8)] for shift in (0, 8, 16, 24))


def _spin_configurations(n_cells: int, reference: int):
    configurations = np.arange(1 << n_cells, dtype=np.uint64)
    original = np.zeros(1 << n_cells, dtype=np.uint64)
    for cell in range(n_cells):
        c_bit = np.uint64(1 << (2 * cell))
        d_bit = np.uint64(1 << (2 * cell + 1))
        choose_d = ((configurations >> cell) & 1) != 0
        original = np.where(choose_d, (original & ~c_bit) | d_bit, original | c_bit)
    n_modes = 2 * n_cells
    state = np.zeros_like(original)
    signs = np.ones(len(original), dtype=np.int8)

    def create(mode: int) -> None:
        nonlocal state, signs
        bit = np.uint64(1 << mode)
        signs *= np.where(_popcount32(state & (bit - np.uint64(1))) % 2, -1, 1).astype(np.int8)
        state |= bit

    def annihilate(mode: int) -> None:
        nonlocal state, signs
        bit = np.uint64(1 << mode)
        signs *= np.where(_popcount32(state & (bit - np.uint64(1))) % 2, -1, 1).astype(np.int8)
        state &= ~bit

    for mode in range(n_modes):
        if reference & (1 << mode):
            create(mode)
    for mode in range(n_modes - 1, -1, -1):
        occupied = (original & np.uint64(1 << mode)) != 0
        if reference & (1 << mode):
            # All configurations are valid, so this update has no zero branch.
            bit = np.uint64(1 << mode)
            signs *= np.where(
                occupied & (_popcount32(state & (bit - np.uint64(1))) % 2 != 0), -1, 1,
            ).astype(np.int8)
            state = np.where(occupied, state & ~bit, state)
        else:
            bit = np.uint64(1 << mode)
            signs *= np.where(
                occupied & (_popcount32(state & (bit - np.uint64(1))) % 2 != 0), -1, 1,
            ).astype(np.int8)
            state = np.where(occupied, state | bit, state)

    transformed = original ^ np.uint64(reference)
    modes = np.zeros((len(original), n_modes), dtype=np.int64)
    counts = np.zeros(len(original), dtype=np.int64)
    for mode in range(n_modes):
        rows = (transformed & np.uint64(1 << mode)) != 0
        modes[rows, counts[rows]] = mode
        counts[rows] += 1
    if np.any(counts % 2):
        raise ValueError("Reference parity is incompatible with the spin Hilbert space")
    return modes, counts, signs.astype(np.float64)


def _batched_pfaffian(matrices: np.ndarray) -> np.ndarray:
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


def projected_amplitudes(z: np.ndarray, modes: np.ndarray, counts: np.ndarray,
                         signs: np.ndarray, chunk_size: int = 8192) -> np.ndarray:
    amplitudes = np.empty(modes.shape[0], dtype=np.complex128)
    for start in range(0, len(amplitudes), chunk_size):
        stop = min(start + chunk_size, len(amplitudes))
        for count in np.unique(counts[start:stop]):
            relative = np.flatnonzero(counts[start:stop] == count)
            absolute = start + relative
            if count == 0:
                amplitudes[absolute] = signs[absolute]
                continue
            selected = modes[absolute, :count]
            matrices = z[selected[:, :, None], selected[:, None, :]]
            amplitudes[absolute] = signs[absolute] * _batched_pfaffian(matrices)
    return amplitudes


def _base_links(states, reference: int, n_cells: int, need_projection: bool):
    links = []
    if not need_projection:
        for index, state in enumerate(states):
            next_state = states[(index + 1) % len(states)]
            links.append({
                "u": normalized_gaussian_overlap(state.z, next_state.z),
                "v": 0.0j, "f1": 0.0, "f2": 0.0,
            })
        return links

    modes, counts, signs = _spin_configurations(n_cells, reference)

    def amplitudes_for(state):
        amplitude = projected_amplitudes(state.z, modes, counts, signs) / np.sqrt(state.full_norm)
        fraction = float(np.vdot(amplitude, amplitude).real)
        return amplitude, fraction

    first_amplitude, first_fraction = amplitudes_for(states[0])
    previous_amplitude, previous_fraction = first_amplitude, first_fraction
    for index in range(len(states)):
        next_index = (index + 1) % len(states)
        if next_index == 0:
            next_amplitude, next_fraction = first_amplitude, first_fraction
        else:
            next_amplitude, next_fraction = amplitudes_for(states[next_index])
        links.append({
            "u": normalized_gaussian_overlap(states[index].z, states[next_index].z),
            "v": complex(np.vdot(previous_amplitude, next_amplitude)),
            "f1": previous_fraction, "f2": next_fraction,
        })
        previous_amplitude, previous_fraction = next_amplitude, next_fraction
    return links


def compute_berry_phases(
    g_values,
    *,
    m: float = -2.0,
    t: float = 1.0,
    t1: float = 1.0,
    delta_p: float = 1.0,
    delta_s: float = 0.5,
    steps_per_leg: int = 8,
    lx: int = 4,
    ly: int = 4,
    reference_mask: Optional[int] = None,
) -> np.ndarray:
    """Return many-body Berry phases in radians, one for each projection strength.

    Parameters are scalars except ``g_values``, whose shape is preserved in the
    output. Here g=0 is unprojected and g=1 is fully spin projected. For fixed
    model and mesh parameters, all Gaussian/projected overlaps are computed only
    once and then reused for every element of ``g_values``.
    """
    strengths = np.asarray(g_values, dtype=float)
    if np.any(~np.isfinite(strengths)) or np.any((strengths < 0) | (strengths > 1)):
        raise ValueError("All projection strengths must be finite numbers in [0,1]")
    if lx * ly > 16:
        raise ValueError("Exact enumeration requires lx*ly <= 16; use Monte Carlo instead")
    params = ModelParameters(t=t, t1=t1, m=m, delta_p=delta_p, delta_s=delta_s)
    model = TwoOrbitalModel(lx, ly, params)
    points = twist_loop(steps_per_leg)
    blocks = [_positive_blocks(model.bdg_hamiltonian(*point)) for point in points]
    reference = _choose_reference(blocks, model.n_cells, reference_mask)
    states = [_make_state(u, v, gap, reference) for u, v, gap in blocks]
    need_projection = bool(np.any(strengths != 0))
    links = _base_links(states, reference, model.n_cells, need_projection)

    flat_results = []
    for g in strengths.ravel():
        if g == 0:
            overlaps = [link["u"] for link in links]
        elif g == 1:
            overlaps = [
                link["v"] / np.sqrt(link["f1"] * link["f2"]) for link in links
            ]
        else:
            weight = g * (2.0 - g)
            overlaps = [
                ((1.0 - g) ** 2 * link["u"] + weight * link["v"])
                / np.sqrt(
                    ((1.0 - g) ** 2 + weight * link["f1"])
                    * ((1.0 - g) ** 2 + weight * link["f2"])
                )
                for link in links
            ]
        total_phase = math.fsum(float(np.angle(overlap)) for overlap in overlaps)
        flat_results.append(float(np.angle(np.exp(1j * total_phase))))
    return np.asarray(flat_results, dtype=float).reshape(strengths.shape)


if __name__ == "__main__":
    # Example: returns a NumPy array aligned with the input projection strengths.
    strengths = np.array([0.0, 0.5, 0.9, 1.0])
    phases = compute_berry_phases(strengths, m=-2.0, steps_per_leg=4)
    print(phases)
