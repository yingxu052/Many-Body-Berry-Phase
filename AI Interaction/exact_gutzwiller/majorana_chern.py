"""Majorana Chern number of the occupied BdG bands using Fukui link variables."""

from __future__ import annotations

import argparse

import numpy as np

from model import ModelParameters


def bloch_bdg(kx: float, ky: float, params: ModelParameters) -> np.ndarray:
    """Fourier transform of TwoOrbitalModel using c_r=sum_k exp(-ikr)c_k."""
    p = params
    ex = np.exp(1j * kx)
    ey = np.exp(1j * ky)
    h = np.zeros((2, 2), dtype=np.complex128)
    h[0, 0] = -p.m - 2 * p.t * (np.cos(kx) + np.cos(ky))
    h[1, 1] = p.m + 2 * p.t * (np.cos(kx) + np.cos(ky))
    h[0, 1] = -2 * p.t1 * np.sin(kx) + 2j * p.t1 * np.sin(ky)
    h[1, 0] = h[0, 1].conjugate()

    # Include both +/- bonds of the full antisymmetric real-space Delta matrix.
    f = -p.delta_p * (np.sin(kx) - 1j * np.sin(ky))
    delta = np.array([[-2 * 0j, 0], [0, 0]], dtype=np.complex128)
    delta[0, 0] = f
    delta[1, 1] = -f
    delta[0, 1] = -p.delta_s
    delta[1, 0] = p.delta_s

    hm = bloch_normal(-kx, -ky, p)
    return np.block([[h, delta], [delta.conj().T, -hm.T]])


def bloch_normal(kx: float, ky: float, params: ModelParameters) -> np.ndarray:
    p = params
    h = np.zeros((2, 2), dtype=np.complex128)
    h[0, 0] = -p.m - 2 * p.t * (np.cos(kx) + np.cos(ky))
    h[1, 1] = p.m + 2 * p.t * (np.cos(kx) + np.cos(ky))
    h[0, 1] = -2 * p.t1 * np.sin(kx) + 2j * p.t1 * np.sin(ky)
    h[1, 0] = h[0, 1].conjugate()
    return h


def occupied_chern(params: ModelParameters, grid: int = 40):
    points = [2 * np.pi * i / grid for i in range(grid)]
    occupied = {}
    minimum_gap = float("inf")
    for ix, kx in enumerate(points):
        for iy, ky in enumerate(points):
            energies, vectors = np.linalg.eigh(bloch_bdg(kx, ky, params))
            minimum_gap = min(minimum_gap, float(energies[2] - energies[1]))
            occupied[ix, iy] = vectors[:, :2]

    def link(a, b):
        value = np.linalg.det(a.conj().T @ b)
        if abs(value) < 1e-14:
            raise ValueError("Singular occupied-band link")
        return value / abs(value)

    curvature = 0.0
    for ix in range(grid):
        for iy in range(grid):
            q = occupied[ix, iy]
            qx = occupied[(ix + 1) % grid, iy]
            qy = occupied[ix, (iy + 1) % grid]
            qxy = occupied[(ix + 1) % grid, (iy + 1) % grid]
            plaquette = link(q, qx) * link(qx, qxy) * np.conjugate(link(qy, qxy)) * np.conjugate(link(q, qy))
            curvature += np.angle(plaquette)
    # Use the manuscript's opposite Berry-curvature orientation: its nu_M=2
    # Chern-insulator regime then has positive, rather than negative, sign.
    return float(-curvature / (2 * np.pi)), minimum_gap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=40)
    parser.add_argument("--m", type=float)
    parser.add_argument("--scan", action="store_true")
    args = parser.parse_args()
    masses = np.linspace(-5, 5, 41) if args.scan else [1.5 if args.m is None else args.m]
    for mass in masses:
        params = ModelParameters(m=float(mass))
        chern, gap = occupied_chern(params, args.grid)
        print(f"m={mass:+.3f}: occupied-BdG Chern={chern:+.6f}, minimum gap={gap:.6e}")


if __name__ == "__main__":
    main()
