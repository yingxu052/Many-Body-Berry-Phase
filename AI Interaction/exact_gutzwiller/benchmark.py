"""Run one exact fully Gutzwiller-projected norm/overlap benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from model import ModelParameters, TwoOrbitalModel, gaussian_pairing_matrix, gaussian_parity
from overlap import build_library, projected_overlap


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=4)
    parser.add_argument("--ly", type=int, default=4)
    parser.add_argument("--m", type=float, default=1.5)
    parser.add_argument("--t", type=float, default=1.0)
    parser.add_argument("--t1", type=float, default=1.0)
    parser.add_argument("--delta-p", type=float, default=1.0)
    parser.add_argument("--delta-s", type=float, default=0.5)
    parser.add_argument("--phi-x", type=float, default=0.23)
    parser.add_argument("--phi-y", type=float, default=-0.17)
    parser.add_argument("--phi2-x", type=float, default=0.24)
    parser.add_argument("--phi2-y", type=float, default=-0.17)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=0, help="Partial speed test only; 0 means exact")
    parser.add_argument("--identical", action="store_true", help="Use the same state on both sides")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("benchmark_result.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    build_library(force=args.rebuild)
    params = ModelParameters(
        t=args.t, t1=args.t1, m=args.m, delta_p=args.delta_p, delta_s=args.delta_s,
    )
    model = TwoOrbitalModel(args.lx, args.ly, params)
    h1 = model.bdg_hamiltonian(args.phi_x, args.phi_y)
    h2 = h1 if args.identical else model.bdg_hamiltonian(args.phi2_x, args.phi2_y)
    parity1 = gaussian_parity(h1)
    parity2 = gaussian_parity(h2)
    if parity1 != 0 or parity2 != 0:
        raise ValueError(
            f"Both states must be even parity for the exponential BCS representation; "
            f"got {parity1} and {parity2}. Choose an even-parity gapped parameter set."
        )
    z1, gap1 = gaussian_pairing_matrix(h1)
    z2, gap2 = gaussian_pairing_matrix(h2)
    result = projected_overlap(
        z1, z2, model.n_cells, parity=0, limit=args.limit, workers=args.workers,
    )
    result.update({
        "lattice": [args.lx, args.ly],
        "parameters": vars(params),
        "phi1": [args.phi_x, args.phi_y],
        "phi2": [args.phi_x, args.phi_y] if args.identical else [args.phi2_x, args.phi2_y],
        "bdg_gaps": [gap1, gap2],
        "parities": [parity1, parity2],
        "workers": args.workers,
    })
    serializable = dict(result)
    for key in ("numerator", "normalized_overlap"):
        value = serializable[key]
        serializable[key] = {"real": value.real, "imag": value.imag}
    args.output.write_text(json.dumps(serializable, indent=2) + "\n")

    print(f"Lattice: {args.lx} x {args.ly}; workers: {args.workers}")
    print(f"BdG gaps: {gap1:.8f}, {gap2:.8f}; parities: {parity1}, {parity2}")
    print(f"Configurations enumerated: {result['enumerated_configurations']:,}")
    print(f"Configurations in selected parity sector: {result['counted_configurations']:,}")
    print(f"Projected norm 1: {result['norm1']:.12e}")
    print(f"Projected norm 2: {result['norm2']:.12e}")
    print(f"Unnormalized overlap: {result['numerator']:.12e}")
    print(f"Normalized overlap: {result['normalized_overlap']:.12e}")
    print(f"Elapsed: {result['elapsed_seconds']:.3f} s")
    print(f"Truncated speed test: {result['truncated']}")
    print(f"Result file: {args.output.resolve()}")


if __name__ == "__main__":
    main()
