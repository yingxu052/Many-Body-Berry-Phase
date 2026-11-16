"""Independent transcription of manuscript Eqs. (A5)-(A6).

Each twisted pairing bond is summed before constructing the BdG matrix.  For
c_i^dagger c_j^dagger coefficient A and c_j^dagger c_i^dagger coefficient B,
the antisymmetric pairing entry is Delta_ij=A-B and Delta_ji=-Delta_ij.
Since A and B carry opposite twist phases, the physical pairing amplitude is
proportional to cos(phi), while normal hopping remains complex.
"""

from __future__ import annotations

import numpy as np


def bdg_manuscript(model, phi_x=0.0, phi_y=0.0, seam="last"):
    p = model.params
    n = model.n_modes
    h = np.zeros((n, n), dtype=np.complex128)
    delta = np.zeros((n, n), dtype=np.complex128)

    def normal(i, j, amplitude):
        h[i, j] += amplitude
        h[j, i] += np.conjugate(amplitude)

    def onsite_pair(i, j, amplitude):
        delta[i, j] += amplitude
        delta[j, i] -= amplitude

    def directed_pair(i, j, a, b):
        # A c_i^dagger c_j^dagger + B c_j^dagger c_i^dagger
        # = (A-B) c_i^dagger c_j^dagger.
        coefficient = a - b
        delta[i, j] += coefficient
        delta[j, i] -= coefficient

    for y in range(model.ly):
        for x in range(model.lx):
            c, d = model.mode(x, y, 0), model.mode(x, y, 1)
            h[c, c] -= p.m
            h[d, d] += p.m
            for direction in ("x", "y"):
                xp, yp = (x + 1, y) if direction == "x" else (x, y + 1)
                cx, dx = model.mode(xp, yp, 0), model.mode(xp, yp, 1)
                phi = phi_x if direction == "x" else phi_y
                if seam == "last":
                    twisted = (x == model.lx - 1) if direction == "x" else (y == model.ly - 1)
                elif seam == "first":
                    twisted = (x == 0) if direction == "x" else (y == 0)
                else:
                    raise ValueError("seam must be 'first' or 'last'")
                forward = np.exp(1j * phi) if twisted else 1.0
                reverse = np.exp(-1j * phi) if twisted else 1.0

                normal(c, cx, -p.t * forward)
                normal(d, dx, p.t * forward)
                interorbital = -1j * p.t1 if direction == "x" else -p.t1
                normal(c, dx, interorbital * forward)
                reverse_sign = 1.0 if direction == "x" else -1.0
                normal(d, cx, reverse_sign * interorbital * forward)

                if direction == "x":
                    directed_pair(c, cx, -1j * p.delta_p * forward / 4,
                                  1j * p.delta_p * reverse / 4)
                    directed_pair(d, dx, 1j * p.delta_p * forward / 4,
                                  -1j * p.delta_p * reverse / 4)
                else:
                    directed_pair(c, cx, -p.delta_p * forward / 4,
                                  p.delta_p * reverse / 4)
                    directed_pair(d, dx, p.delta_p * forward / 4,
                                  -p.delta_p * reverse / 4)
            onsite_pair(c, d, -p.delta_s / 2)
            onsite_pair(d, c, p.delta_s / 2)

    h_bdg = np.block([[h, delta], [-delta.conj(), -h.T]])
    return 0.5 * (h_bdg + h_bdg.conj().T)
