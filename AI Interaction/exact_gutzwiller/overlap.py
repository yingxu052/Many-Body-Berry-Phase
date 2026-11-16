"""Python/ctypes interface to the compiled exact projected-overlap enumerator."""

from __future__ import annotations

import ctypes
import subprocess
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LIBRARY = ROOT / "libenumerator.dylib"
_FLOAT_ARRAY = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")


def build_library(force: bool = False) -> Path:
    source = ROOT / "enumerator.cpp"
    if force or not LIBRARY.exists() or LIBRARY.stat().st_mtime < source.stat().st_mtime:
        command = [
            "g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-pthread",
            str(source), "-o", str(LIBRARY),
        ]
        subprocess.run(command, check=True, cwd=ROOT)
    return LIBRARY


def _load_library():
    build_library()
    library = ctypes.CDLL(str(LIBRARY))
    library.pfaffian_test.argtypes = [_FLOAT_ARRAY, _FLOAT_ARRAY, ctypes.c_int, ctypes.c_int]
    library.pfaffian_test.restype = ctypes.c_double
    library.projected_overlap.argtypes = [
        _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY, _FLOAT_ARRAY,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_uint64, ctypes.c_int, _FLOAT_ARRAY,
    ]
    library.projected_overlap.restype = None
    return library


def compiled_pfaffian(matrix: np.ndarray) -> complex:
    matrix = np.ascontiguousarray(matrix, dtype=np.complex128)
    real = np.ascontiguousarray(matrix.real)
    imag = np.ascontiguousarray(matrix.imag)
    library = _load_library()
    n = matrix.shape[0]
    return complex(
        library.pfaffian_test(real, imag, n, 0),
        library.pfaffian_test(real, imag, n, 1),
    )


def projected_overlap(
    z1: np.ndarray,
    z2: np.ndarray,
    n_cells: int,
    parity: int = 0,
    reference_mask1: int = 0,
    reference_mask2: int = 0,
    limit: int = 0,
    workers: int = 1,
):
    """Return exact or explicitly truncated projected norms and overlap.

    A nonzero limit enumerates only the first `limit` base-3 configurations and
    is intended solely for performance benchmarking, not physical results.
    """
    z1 = np.ascontiguousarray(z1, dtype=np.complex128)
    z2 = np.ascontiguousarray(z2, dtype=np.complex128)
    if z1.shape != (2 * n_cells, 2 * n_cells) or z2.shape != z1.shape:
        raise ValueError("Pairing matrices must both have shape (2*n_cells, 2*n_cells)")
    if n_cells > 16:
        raise ValueError("This quick-test enumerator supports at most 16 unit cells")
    if parity not in (0, 1):
        raise ValueError("Parity must be 0 (even) or 1 (odd)")

    arrays = [np.ascontiguousarray(z.real) for z in (z1, z2)]
    arrays += [np.ascontiguousarray(z.imag) for z in (z1, z2)]
    z1_real, z2_real, z1_imag, z2_imag = arrays
    output = np.zeros(6, dtype=np.float64)
    library = _load_library()

    start = time.perf_counter()
    library.projected_overlap(
        z1_real, z1_imag, z2_real, z2_imag,
        2 * n_cells, n_cells, parity, reference_mask1, reference_mask2,
        limit, workers, output,
    )
    elapsed = time.perf_counter() - start
    norm1, norm2 = output[0], output[1]
    numerator = complex(output[2], output[3])
    normalized = numerator / np.sqrt(norm1 * norm2) if norm1 > 0 and norm2 > 0 else np.nan
    return {
        "norm1": float(norm1),
        "norm2": float(norm2),
        "numerator": numerator,
        "normalized_overlap": normalized,
        "counted_configurations": int(output[4]),
        "enumerated_configurations": int(output[5]),
        "elapsed_seconds": elapsed,
        "truncated": bool(limit and int(output[5]) < 3**n_cells),
    }
