"""Diagnostics for the real phase relation between Gaussian and projected overlaps."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    for deformation in ("real_profile", "imaginary"):
        records = []
        path = HERE / "projected_pairing_response_m128_h0.0001.jsonl"
        with path.open() as handle:
            for line in handle:
                record = json.loads(line)
                if record["deformation"] == deformation:
                    records.append(record)
        ratios = []
        phase_differences = []
        for record in records:
            gaussian = complex(*record["gaussian_overlap"])
            projected = complex(*record["projected_overlap"])
            ratio = projected / gaussian
            ratios.append(ratio)
            phase_differences.append(np.angle(projected / gaussian))
        ratios = np.asarray(ratios)
        phase_differences = np.asarray(phase_differences)
        relative_imaginary = np.abs(ratios.imag) / np.abs(ratios.real)
        print(deformation)
        print(f"  records: {len(records)}")
        print(f"  minimum real ratio: {ratios.real.min():.16e}")
        print(f"  maximum real ratio: {ratios.real.max():.16e}")
        print(f"  maximum |Im ratio|/|Re ratio|: {relative_imaginary.max():.3e}")
        print(f"  maximum |arg(projected/gaussian)|: {np.abs(phase_differences).max():.3e}")
        print(f"  ratio magnitude range: [{np.abs(ratios).min():.12f}, {np.abs(ratios).max():.12f}]")


if __name__ == "__main__":
    main()
