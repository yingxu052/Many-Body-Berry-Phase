"""Benchmark one exact 4 x 4 finite-g normalized overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finite_g_overlap import finite_g_overlap
from model import ModelParameters, TwoOrbitalModel, gaussian_bcs_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("finite_g_benchmark.json"))
    args = parser.parse_args()

    model = TwoOrbitalModel(4, 4, ModelParameters(m=1.5))
    state1 = gaussian_bcs_state(model.bdg_hamiltonian(0.23, -0.17))
    state2 = gaussian_bcs_state(model.bdg_hamiltonian(0.24, -0.17))
    assert state1.reference_mask == state2.reference_mask
    result = finite_g_overlap(
        state1.z, state2.z, model.n_cells, args.g,
        reference_mask=state1.reference_mask, workers=args.workers,
    )
    result["g"] = args.g
    result["lattice"] = [4, 4]
    result["reference_mask"] = state1.reference_mask
    args.output.write_text(json.dumps(result, indent=2, default=lambda value: {"real": value.real, "imag": value.imag}) + "\n")
    print(f"g={args.g:.3f}")
    print(f"Normalized overlap: {result['normalized_overlap']}")
    print(f"Norm fractions: {result['norm1_fraction']:.12f}, {result['norm2_fraction']:.12f}")
    print(f"Elapsed: {result['elapsed_seconds']:.3f} s")
    print(f"Result file: {args.output.resolve()}")


if __name__ == "__main__":
    main()
