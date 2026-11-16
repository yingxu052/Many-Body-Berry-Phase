"""Exact overlaps after projection to one fermionic parton per unit cell."""

from __future__ import annotations

import ctypes
import time

import numpy as np

from overlap import _FLOAT_ARRAY, _load_library


def _unprojected_norm(z: np.ndarray) -> float:
    identity = np.eye(z.shape[0], dtype=np.complex128)
    determinant = float(np.linalg.det(identity + z.conj().T @ z).real)
    if determinant <= 0.0:
        raise ValueError("Encountered a zero-norm BCS state")
    return float(np.sqrt(determinant))


def single_occupancy_overlap(
    z1: np.ndarray,
    z2: np.ndarray,
    n_cells: int,
    reference_mask1: int = 0,
    reference_mask2: int = 0,
    workers: int = 8,
):
    """Return <Z1|P_G|Z2> and norms for P_G=prod_i n_i(2-n_i).

    The BCS states use their unnormalized Pfaffian amplitudes. The normalized
    overlap is therefore the raw projected overlap divided by the square root
    of the two raw projected norms. Norm fractions are relative to the full
    unprojected Gaussian norms.
    """
    z1 = np.ascontiguousarray(z1, dtype=np.complex128)
    z2 = np.ascontiguousarray(z2, dtype=np.complex128)
    expected_shape = (2 * n_cells, 2 * n_cells)
    if z1.shape != expected_shape or z2.shape != expected_shape:
        raise ValueError("Pairing matrices must both have shape (2*n_cells, 2*n_cells)")
    if not 0 <= n_cells <= 16:
        raise ValueError("The exact enumerator supports between 0 and 16 unit cells")

    arrays = [np.ascontiguousarray(z.real) for z in (z1, z2)]
    arrays += [np.ascontiguousarray(z.imag) for z in (z1, z2)]
    z1_real, z2_real, z1_imag, z2_imag = arrays
    output = np.zeros(6, dtype=np.float64)
    library = _load_library()
    library.single_occupancy_overlap.argtypes = [
        _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int, _FLOAT_ARRAY,
    ]
    library.single_occupancy_overlap.restype = None

    start = time.perf_counter()
    library.single_occupancy_overlap(
        z1_real, z1_imag, z2_real, z2_imag,
        2 * n_cells, n_cells, n_cells % 2,
        reference_mask1, reference_mask2, workers, output,
    )
    elapsed = time.perf_counter() - start
    norm1, norm2 = float(output[0]), float(output[1])
    numerator = complex(output[2], output[3])
    if norm1 <= 0.0 or norm2 <= 0.0:
        raise ValueError(f"Projected state has zero norm: ({norm1}, {norm2})")
    normalized = numerator / np.sqrt(norm1 * norm2)
    return {
        "norm1": norm1,
        "norm2": norm2,
        "norm1_fraction": norm1 / _unprojected_norm(z1),
        "norm2_fraction": norm2 / _unprojected_norm(z2),
        "numerator": numerator,
        "normalized_overlap": complex(normalized),
        "counted_configurations": int(output[4]),
        "enumerated_configurations": int(output[5]),
        "elapsed_seconds": elapsed,
    }
