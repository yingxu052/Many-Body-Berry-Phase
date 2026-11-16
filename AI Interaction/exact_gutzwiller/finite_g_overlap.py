"""Exact finite-Gutzwiller overlaps using Gaussian Wick subset sums."""

from __future__ import annotations

import ctypes
import time

import numpy as np

from gaussian_overlap import normalized_gaussian_overlap
from overlap import _FLOAT_ARRAY, _load_library


def transition_contractions(z1: np.ndarray, z2: np.ndarray):
    """Return <c^dag c^dag>, <cc>, <c^dag c>, and <cc^dag> divided by overlap."""
    n_modes = z1.shape[0]
    identity = np.eye(n_modes, dtype=np.complex128)
    z1_dagger = z1.conj().T
    a = identity + z1_dagger @ z2
    a_inverse = np.linalg.solve(a, identity)
    creators = z1_dagger @ a_inverse.T
    annihilators = z2 @ a_inverse
    normal = z1_dagger @ annihilators
    reverse = identity - normal.T
    return creators, annihilators, normal, reverse


def double_occupancy_gamma(z1: np.ndarray, z2: np.ndarray, reference_mask: int) -> np.ndarray:
    """Transition Wick matrix for ordered physical double-occupancy operators."""
    creators, annihilators, normal, reverse = transition_contractions(z1, z2)
    n_modes = z1.shape[0]
    n_cells = n_modes // 2
    operators = []
    for cell in range(n_cells):
        for mode in (2 * cell, 2 * cell + 1):
            particle_hole = bool(reference_mask & (1 << mode))
            # Original c^dag, c, in that order. Particle-hole swaps their d-basis type.
            operators.append((mode, not particle_hole))
            operators.append((mode, particle_hole))
    # Reorder from [c^dag,c,d^dag,d] branches produced above: this is already
    # the required four-operator sequence for each unit cell.

    gamma = np.zeros((4 * n_cells, 4 * n_cells), dtype=np.complex128)
    for a, (mode_a, creator_a) in enumerate(operators):
        for b in range(a + 1, len(operators)):
            mode_b, creator_b = operators[b]
            if creator_a and creator_b:
                value = creators[mode_a, mode_b]
            elif not creator_a and not creator_b:
                value = annihilators[mode_a, mode_b]
            elif creator_a:
                value = normal[mode_a, mode_b]
            else:
                value = reverse[mode_a, mode_b]
            gamma[a, b] = value
            gamma[b, a] = -value
    return gamma


def finite_g_overlap(
    z1: np.ndarray,
    z2: np.ndarray,
    n_cells: int,
    g: float,
    reference_mask: int = 0,
    workers: int = 8,
):
    """Return the exact normalized overlap after P(g) projection.

    P(g)=product_i[1-(1-g)n_ic n_id]. Since P(g)^2 has coefficient
    lambda=1-g^2 for each double-occupancy projector, 2^n_cells Gaussian
    Pfaffians per norm/overlap replace the full four-state-per-cell sum.
    """
    if not 0.0 <= g <= 1.0:
        raise ValueError("g must lie between 0 and 1")
    if reference_mask >= (1 << (2 * n_cells)):
        raise ValueError("Reference mask is inconsistent with the mode count")
    base_overlap = normalized_gaussian_overlap(z1, z2)
    if g == 1.0:
        return {
            "normalized_overlap": base_overlap,
            "norm1_fraction": 1.0,
            "norm2_fraction": 1.0,
            "cross_factor": 1.0 + 0.0j,
            "unprojected_overlap": base_overlap,
            "elapsed_seconds": 0.0,
        }

    start = time.perf_counter()
    gamma1 = double_occupancy_gamma(z1, z1, reference_mask)
    gamma2 = double_occupancy_gamma(z2, z2, reference_mask)
    gamma12 = double_occupancy_gamma(z1, z2, reference_mask)
    arrays = []
    for gamma in (gamma1, gamma2, gamma12):
        arrays.extend((np.ascontiguousarray(gamma.real), np.ascontiguousarray(gamma.imag)))
    output = np.zeros(6, dtype=np.float64)
    library = _load_library()
    library.finite_g_factors.argtypes = [
        _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY,
        ctypes.c_int, ctypes.c_double, ctypes.c_int, _FLOAT_ARRAY,
    ]
    library.finite_g_factors.restype = None
    library.finite_g_factors(*arrays, n_cells, 1.0 - g * g, workers, output)
    norm1 = complex(output[0], output[1])
    norm2 = complex(output[2], output[3])
    cross = complex(output[4], output[5])
    normalized = base_overlap * cross / np.sqrt(norm1 * norm2)
    return {
        "normalized_overlap": complex(normalized),
        "norm1_fraction": float(norm1.real),
        "norm2_fraction": float(norm2.real),
        "cross_factor": cross,
        "unprojected_overlap": base_overlap,
        "norm1_imaginary_residual": norm1.imag,
        "norm2_imaginary_residual": norm2.imag,
        "elapsed_seconds": time.perf_counter() - start,
    }


def finite_g_overlaps(
    z1: np.ndarray,
    z2: np.ndarray,
    n_cells: int,
    g_values,
    reference_mask: int = 0,
    workers: int = 8,
):
    """Evaluate a whole g scan while reusing every subset Pfaffian.

    The convention is identical to finite_g_overlap:
    P(g)=product_i[1-(1-g)n_ic n_id].
    """
    strengths = np.asarray(g_values, dtype=np.float64)
    if strengths.ndim != 1 or len(strengths) == 0:
        raise ValueError("g_values must be a nonempty one-dimensional array")
    if np.any(~np.isfinite(strengths)) or np.any((strengths < 0.0) | (strengths > 1.0)):
        raise ValueError("Every g must lie between 0 and 1")
    if reference_mask >= (1 << (2 * n_cells)):
        raise ValueError("Reference mask is inconsistent with the mode count")

    start = time.perf_counter()
    base_overlap = normalized_gaussian_overlap(z1, z2)
    gamma1 = double_occupancy_gamma(z1, z1, reference_mask)
    gamma2 = double_occupancy_gamma(z2, z2, reference_mask)
    gamma12 = double_occupancy_gamma(z1, z2, reference_mask)
    arrays = []
    for gamma in (gamma1, gamma2, gamma12):
        arrays.extend((np.ascontiguousarray(gamma.real), np.ascontiguousarray(gamma.imag)))
    lambdas = np.ascontiguousarray(1.0 - strengths * strengths, dtype=np.float64)
    output = np.zeros(6 * len(strengths), dtype=np.float64)
    library = _load_library()
    library.finite_g_factors_batch.argtypes = [
        _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY,
        ctypes.c_int, _FLOAT_ARRAY, ctypes.c_int, ctypes.c_int, _FLOAT_ARRAY,
    ]
    library.finite_g_factors_batch.restype = None
    library.finite_g_factors_batch(
        *arrays, n_cells, lambdas, len(strengths), workers, output,
    )
    elapsed = time.perf_counter() - start

    results = []
    for index, g in enumerate(strengths):
        values = output[6 * index:6 * index + 6]
        norm1 = complex(values[0], values[1])
        norm2 = complex(values[2], values[3])
        cross = complex(values[4], values[5])
        normalized = base_overlap * cross / np.sqrt(norm1 * norm2)
        results.append({
            "g": float(g),
            "normalized_overlap": complex(normalized),
            "norm1_fraction": float(norm1.real),
            "norm2_fraction": float(norm2.real),
            "cross_factor": cross,
            "unprojected_overlap": base_overlap,
            "norm1_imaginary_residual": norm1.imag,
            "norm2_imaginary_residual": norm2.imag,
            "elapsed_seconds": elapsed,
        })
    return results
