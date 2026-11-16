"""BdG model and Gaussian-state conventions for the projected-overlap test.

Real-space Hamiltonian convention:
    H = sum_ij h_ij c_i^dag c_j
        + 1/2 sum_ij Delta_ij c_i^dag c_j^dag
        - 1/2 sum_ij Delta_ij^* c_i c_j.
The BCS state is |Z> proportional to exp(1/2 sum_ij Z_ij c_i^dag c_j^dag)|0>.
Modes are ordered by unit cell, with c before d in each cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    t: float = 1.0
    t1: float = 1.0
    m: float = 1.0
    delta_p: float = 1.0
    delta_s: float = 0.5


class TwoOrbitalModel:
    def __init__(self, lx: int, ly: int, params: ModelParameters = ModelParameters()):
        self.lx = int(lx)
        self.ly = int(ly)
        self.params = params
        self.n_cells = self.lx * self.ly
        self.n_modes = 2 * self.n_cells

    def cell(self, x: int, y: int) -> int:
        return (y % self.ly) * self.lx + (x % self.lx)

    def mode(self, x: int, y: int, orbital: int) -> int:
        return 2 * self.cell(x, y) + orbital

    def bdg_hamiltonian(
        self, phi_x: float = 0.0, phi_y: float = 0.0, pairing_twist: str = "cosine",
        pairing_imag: float = 0.0,
    ) -> np.ndarray:
        """Construct the real-space BdG matrix and its boundary twist.

        The default follows Eqs. (A5)--(A6): after summing both ordered pairing
        terms, twisted pairing is proportional to cos(phi). The former complex
        pairing path is retained only for explicit comparison. The "linear"
        option is the closed piecewise-linear interpolation between +Delta_p
        and -Delta_p on successive pi-twist intervals. For the linear option,
        pairing_imag adds i*pairing_imag*sin(phi) on each twisted seam.
        """
        if pairing_twist not in ("complex", "cosine", "linear"):
            raise ValueError("pairing_twist must be 'complex', 'cosine', or 'linear'")
        if pairing_imag != 0.0 and pairing_twist != "linear":
            raise ValueError("pairing_imag is only supported with pairing_twist='linear'")
        p = self.params
        n = self.n_modes
        h = np.zeros((n, n), dtype=np.complex128)
        delta = np.zeros((n, n), dtype=np.complex128)

        def normal(i: int, j: int, amplitude: complex) -> None:
            # amplitude c_i^dag c_j + h.c.
            h[i, j] += amplitude
            h[j, i] += np.conjugate(amplitude)

        def pair(i: int, j: int, amplitude: complex) -> None:
            # amplitude c_i^dag c_j^dag + h.c.
            delta[i, j] += amplitude
            delta[j, i] -= amplitude

        for y in range(self.ly):
            for x in range(self.lx):
                c = self.mode(x, y, 0)
                d = self.mode(x, y, 1)
                h[c, c] -= p.m
                h[d, d] += p.m

                for direction in ("x", "y"):
                    xp, yp = (x + 1, y) if direction == "x" else (x, y + 1)
                    cx = self.mode(xp, yp, 0)
                    dx = self.mode(xp, yp, 1)
                    phi = phi_x if direction == "x" else phi_y
                    # Eqs. (A5)--(A6): twist only the bonds from 0 to 1.
                    twisted = (x == 0) if direction == "x" else (y == 0)
                    phase = np.exp(1j * phi) if twisted else 1.0
                    if not twisted:
                        pair_phase = 1.0
                    elif pairing_twist == "cosine":
                        pair_phase = np.cos(phi)
                    elif pairing_twist == "linear":
                        reduced_phi = phi % (2.0 * np.pi)
                        pair_phase = (
                            1.0 - 2.0 * reduced_phi / np.pi
                            if reduced_phi <= np.pi
                            else 2.0 * reduced_phi / np.pi - 3.0
                        )
                    else:
                        pair_phase = phase
                    if twisted and pairing_twist == "linear" and pairing_imag != 0.0:
                        pair_phase += 1j * pairing_imag * np.sin(phi)

                    normal(c, cx, -p.t * phase)
                    normal(d, dx, p.t * phase)
                    interorbital = -1j * p.t1 if direction == "x" else -p.t1
                    normal(c, dx, interorbital * phase)
                    # Eq. (A2) requires -2*t1*sin(kx) in h_cd. The opposite
                    # sign printed for the second x bond in Eq. (A4) instead
                    # gives -2i*t1*cos(kx) and removes the nu_M=2 phase.
                    reverse_sign = 1.0 if direction == "x" else -1.0
                    normal(d, cx, reverse_sign * interorbital * phase)

                    # Normalize the ordered real-space terms so the Bloch
                    # pairing is exactly Eq. (A3), without extra factors.
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

    def normal_and_pairing_blocks(self, phi_x: float = 0.0, phi_y: float = 0.0):
        h_bdg = self.bdg_hamiltonian(phi_x, phi_y)
        n = self.n_modes
        return h_bdg[:n, :n], h_bdg[:n, n:]


def gaussian_pairing_matrix(h_bdg: np.ndarray, gap_tolerance: float = 1e-8):
    """Return the BCS pairing matrix Z and single-particle gap.

    Positive-energy BdG eigenvectors define quasiparticles annihilated in the
    ground state. If their upper/lower blocks are U/V, then
    C=<c^dag c>=V V^dag and F=<c c>=U V^dag. For
    |Z>=exp(c^dag Z c^dag/2)|0>, Z=-F (1-C)^(-1).
    """
    n = h_bdg.shape[0] // 2
    energies, vectors = np.linalg.eigh(h_bdg)
    positive = energies[energies > gap_tolerance]
    negative = energies[energies < -gap_tolerance]
    if len(positive) != n or len(negative) != n:
        raise ValueError(f"BdG spectrum is gapless or ill-conditioned: {energies}")
    gap = float(min(np.min(positive), -np.max(negative)))
    positive_vectors = vectors[:, energies > gap_tolerance]
    u = positive_vectors[:n, :]
    v = positive_vectors[n:, :]
    c = v @ v.conj().T
    f = u @ v.conj().T
    vacuum_matrix = np.eye(n, dtype=np.complex128) - c
    condition = np.linalg.cond(vacuum_matrix)
    if condition > 1e10:
        raise ValueError(
            "The vacuum-reference BCS representation is ill-conditioned "
            f"(cond(1-C)={condition:.3e}); choose a regular parameter point"
        )
    z = -np.linalg.solve(vacuum_matrix.T, f.T).T
    z = 0.5 * (z - z.T)
    return z, gap


def gaussian_parity(h_bdg: np.ndarray, gap_tolerance: float = 1e-8) -> int:
    """Ground-state fermion parity from det(W): 0 for even and 1 for odd."""
    n = h_bdg.shape[0] // 2
    energies, vectors = np.linalg.eigh(h_bdg)
    positive_vectors = vectors[:, energies > gap_tolerance]
    u = positive_vectors[:n, :]
    v = positive_vectors[n:, :]
    w = np.block([[u, v.conj()], [v, u.conj()]])
    determinant = np.linalg.det(w)
    if abs(determinant.imag) > 1e-8 or abs(abs(determinant.real) - 1.0) > 1e-6:
        raise ValueError(f"Bogoliubov determinant is not a parity sign: {determinant}")
    return int(determinant.real < 0.0)


@dataclass(frozen=True)
class GaussianBCSState:
    z: np.ndarray
    gap: float
    parity: int
    reference_mask: int
    reference_condition: float


def gaussian_bcs_state(
    h_bdg: np.ndarray, gap_tolerance: float = 1e-8, reference_mask: int | None = None,
) -> GaussianBCSState:
    """Construct an even Gaussian state about a fixed or optimized Slater reference.

    If reference_mask is supplied, that common reference is used instead of
    optimizing separate references for different states.

    For modes in reference_mask, the particle-hole transformation d=c^dag is
    used; otherwise d=c. In the d basis, |0_d> is the corresponding Slater
    determinant and |Z>=exp(d^dag Z d^dag/2)|0_d>. The null condition gives
    Z=-U_d^(-dag) V_d^dag. Row swaps between U and V implement particle-hole.
    """
    n = h_bdg.shape[0] // 2
    energies, vectors = np.linalg.eigh(h_bdg)
    positive = energies[energies > gap_tolerance]
    negative = energies[energies < -gap_tolerance]
    if len(positive) != n or len(negative) != n:
        raise ValueError(f"BdG spectrum is gapless or ill-conditioned: {energies}")
    gap = float(min(np.min(positive), -np.max(negative)))
    positive_vectors = vectors[:, energies > gap_tolerance]
    u = positive_vectors[:n, :].copy()
    v = positive_vectors[n:, :].copy()
    w = np.block([[u, v.conj()], [v, u.conj()]])
    parity = int(np.linalg.det(w).real < 0.0)
    if parity != 0:
        raise ValueError("Only even-parity Gaussian states are supported by this BCS representation")

    c = v @ v.conj().T
    base_mask = sum(1 << i for i in range(n) if c[i, i].real > 0.5)
    initial_masks = {
        0,
        (1 << n) - 1,
        base_mask,
        sum(1 << i for i in range(0, n, 2)),
        sum(1 << i for i in range(1, n, 2)),
    }
    for threshold in (0.25, 0.75):
        initial_masks.add(sum(1 << i for i in range(n) if c[i, i].real > threshold))

    cache = {}

    def evaluate(mask: int):
        if mask in cache:
            return cache[mask]
        ud = u.copy()
        vd = v.copy()
        for i in range(n):
            if mask & (1 << i):
                ud[i, :], vd[i, :] = v[i, :], u[i, :]
        try:
            condition = float(np.linalg.cond(ud))
        except np.linalg.LinAlgError:
            condition = float("inf")
        cache[mask] = (condition, ud, vd)
        return cache[mask]

    if reference_mask is not None:
        if not 0 <= reference_mask < (1 << n):
            raise ValueError("reference_mask is inconsistent with the mode count")
        best_mask = reference_mask
        best_condition = evaluate(best_mask)[0]
    else:
        best_mask = base_mask
        best_condition = evaluate(best_mask)[0]
        for initial in initial_masks:
            mask = initial
            condition = evaluate(mask)[0]
            if condition < best_condition:
                best_mask, best_condition = mask, condition
            # Deterministic local search over particle-hole row swaps.
            for _ in range(6):
                candidate_mask = mask
                candidate_condition = condition
                for i in range(n):
                    trial_mask = mask ^ (1 << i)
                    trial_condition = evaluate(trial_mask)[0]
                    if trial_condition < candidate_condition:
                        candidate_mask = trial_mask
                        candidate_condition = trial_condition
                if candidate_mask == mask:
                    break
                mask, condition = candidate_mask, candidate_condition
                if condition < best_condition:
                    best_mask, best_condition = mask, condition

    if best_condition > 1e10:
        raise ValueError(f"No regular Slater-reference BCS representation found; cond={best_condition:.3e}")
    _, ud, vd = evaluate(best_mask)
    z = -np.linalg.solve(ud.conj().T, vd.conj().T)
    z = 0.5 * (z - z.T)
    return GaussianBCSState(z, gap, parity, best_mask, best_condition)


def reference_pfaffian(a: np.ndarray) -> complex:
    """Recursive complex Pfaffian, intended only for small validation matrices."""
    n = a.shape[0]
    if n == 0:
        return 1.0 + 0.0j
    if n % 2:
        return 0.0 + 0.0j
    if n == 2:
        return a[0, 1]
    total = 0.0 + 0.0j
    for j in range(1, n):
        indices = [k for k in range(1, n) if k != j]
        total += (-1) ** (j + 1) * a[0, j] * reference_pfaffian(a[np.ix_(indices, indices)])
    return total


def bcs_amplitude(z: np.ndarray, occupied_modes) -> complex:
    occupied = list(occupied_modes)
    if len(occupied) % 2:
        return 0.0 + 0.0j
    if not occupied:
        return 1.0 + 0.0j
    return reference_pfaffian(z[np.ix_(occupied, occupied)])
