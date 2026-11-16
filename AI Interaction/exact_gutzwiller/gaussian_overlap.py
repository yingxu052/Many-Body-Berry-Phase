"""Normalized overlaps of unprojected BCS states about a common Slater reference."""

from __future__ import annotations

import numpy as np


def normalized_gaussian_overlap(z1: np.ndarray, z2: np.ndarray) -> complex:
    """Return <Z1|Z2>/sqrt(<Z1|Z1><Z2|Z2>).

    The states are exp(d^dag Z d^dag/2)|0_d> with the same reference
    determinant. The principal determinant square root is continuous for
    sufficiently nearby states, as required for a discrete Berry link.
    """
    identity = np.eye(z1.shape[0], dtype=np.complex128)
    determinant1 = np.linalg.det(identity + z1.conj().T @ z1)
    determinant2 = np.linalg.det(identity + z2.conj().T @ z2)
    cross_determinant = np.linalg.det(identity + z1.conj().T @ z2)
    norm1 = float(np.sqrt(determinant1.real))
    norm2 = float(np.sqrt(determinant2.real))
    if norm1 <= 0.0 or norm2 <= 0.0:
        raise ValueError("Encountered a zero-norm BCS state")
    return complex(np.sqrt(cross_determinant) / np.sqrt(norm1 * norm2))
