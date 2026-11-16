"""Combine the M=8 and M=16 finite-Gutzwiller Berry-phase scans."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    scan8 = read_json(ROOT / "finite_g_berry_scan_m8.json")
    scan16 = read_json(ROOT / "finite_g_berry_scan_m16.json")
    assert [item["g"] for item in scan8["results"]] == [item["g"] for item in scan16["results"]]
    entries = []
    for coarse, fine in zip(scan8["results"], scan16["results"]):
        entries.append({
            "g": fine["g"],
            "phase_m8_radians": coarse["berry_phase_radians"],
            "phase_m8_over_pi": coarse["berry_phase_over_pi"],
            "phase_m16_radians": fine["berry_phase_radians"],
            "phase_m16_over_pi": fine["berry_phase_over_pi"],
            "m16_minus_m8_radians": fine["berry_phase_radians"] - coarse["berry_phase_radians"],
            "minimum_link_magnitude_m16": fine["minimum_link_magnitude"],
            "minimum_norm_fraction_m16": fine["minimum_norm_fraction"],
        })
    output = ROOT / "finite_g_mesh_comparison.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump({
            "projection_convention": scan16["projection_convention"],
            "endpoint_checks_m8": scan8["endpoint_checks"],
            "endpoint_checks_m16": scan16["endpoint_checks"],
            "entries": entries,
        }, handle, indent=2)
        handle.write("\n")
    for entry in entries:
        print(
            f"g={entry['g']:.1f}: M8={entry['phase_m8_over_pi']:+.9f} pi, "
            f"M16={entry['phase_m16_over_pi']:+.9f} pi, "
            f"difference={entry['m16_minus_m8_radians']:+.9f} rad"
        )
    print(f"Comparison file: {output}")


if __name__ == "__main__":
    main()
