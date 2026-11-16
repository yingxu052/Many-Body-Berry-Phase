"""Combine corrected parton and double-occupancy exact Berry-phase scans."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "exact_gutzwiller"


def read(name):
    return json.loads((DATA / name).read_text())


def aligned_entries(mesh):
    parton = read(f"interpolated_projection_m-2_cosine_M{mesh}.json")
    double = read(f"finite_g_double_occupancy_m-2_cosine_M{mesh}.json")
    assert parton["parameters"] == double["parameters"]
    double_by_strength = {round(1.0 - item["g"], 12): item for item in double["results"]}
    entries = []
    for parton_item in parton["results"]:
        strength = parton_item["lambda"]
        double_item = double_by_strength[strength]
        entries.append({
            "projection_strength": strength,
            "parton_lambda": parton_item["lambda"],
            "double_occupancy_original_g": double_item["g"],
            "parton_phase_radians": parton_item["berry_phase_radians"],
            "parton_phase_over_pi": parton_item["berry_phase_over_pi"],
            "double_occupancy_phase_radians": double_item["berry_phase_radians"],
            "double_occupancy_phase_over_pi": double_item["berry_phase_over_pi"],
            "parton_minus_double_radians": float(np.angle(np.exp(1j * (
                parton_item["berry_phase_radians"] - double_item["berry_phase_radians"]
            )))),
            "parton_minimum_link_magnitude": parton_item["minimum_link_magnitude"],
            "double_occupancy_minimum_link_magnitude": double_item["minimum_link_magnitude"],
        })
    return parton, double, entries


parton16, double16, entries16 = aligned_entries(16)
parton32, double32, entries32 = aligned_entries(32)
for coarse, fine in zip(entries16, entries32):
    assert coarse["projection_strength"] == fine["projection_strength"]
    coarse["parton_m32_minus_m16_radians"] = float(np.angle(np.exp(1j * (
        fine["parton_phase_radians"] - coarse["parton_phase_radians"]
    ))))
    coarse["double_m32_minus_m16_radians"] = float(np.angle(np.exp(1j * (
        fine["double_occupancy_phase_radians"] - coarse["double_occupancy_phase_radians"]
    ))))

payload = {
    "lattice": [4, 4],
    "parameters": parton32["parameters"],
    "pairing_twist": "cosine",
    "common_projection_strength": "0=unprojected, 1=fully projected",
    "parton_convention": parton32["interpolation"],
    "double_occupancy_convention": double32["projection_convention"],
    "maximum_parton_double_difference_m32_radians": max(
        abs(item["parton_minus_double_radians"]) for item in entries32
    ),
    "maximum_mesh_difference_radians": max(
        max(abs(item["parton_m32_minus_m16_radians"]), abs(item["double_m32_minus_m16_radians"]))
        for item in entries16
    ),
    "results_m16_with_mesh_differences": entries16,
    "results_m32": entries32,
}
json_path = DATA / "projection_comparison_m-2_cosine.json"
json_path.write_text(json.dumps(payload, indent=2) + "\n")

csv_path = DATA / "projection_comparison_m-2_cosine.csv"
with csv_path.open("w", newline="") as handle:
    fields = [
        "projection_strength", "parton_phase_radians", "parton_phase_over_pi",
        "double_occupancy_phase_radians", "double_occupancy_phase_over_pi",
        "parton_minus_double_radians", "parton_minimum_link_magnitude",
        "double_occupancy_minimum_link_magnitude",
    ]
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(entries32)

strengths = [item["projection_strength"] for item in entries32]
parton_phase = [item["parton_phase_over_pi"] for item in entries32]
double_phase = [item["double_occupancy_phase_over_pi"] for item in entries32]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.plot(strengths, parton_phase, "o-", lw=1.8, ms=5, label="Parton projection")
ax.plot(strengths, double_phase, "x--", lw=1.4, ms=6, label="Double-occupancy suppression")
ax.axhline(-0.5, color="0.35", lw=0.8, ls=":")
ax.set(xlabel="Projection strength (0 = unprojected, 1 = full)", ylabel=r"Berry phase / $\pi$",
       ylim=(-0.53, -0.47))
ax.grid(alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()
png_path = DATA / "projection_comparison_m-2_cosine.png"
fig.savefig(png_path, dpi=180)
plt.close(fig)

print(f"Maximum parton/double difference (M32): {payload['maximum_parton_double_difference_m32_radians']:.3e} rad")
print(f"Maximum M32-M16 difference: {payload['maximum_mesh_difference_radians']:.3e} rad")
for item in entries32:
    print(
        f"g={item['projection_strength']:.1f}: parton={item['parton_phase_over_pi']:+.12f} pi, "
        f"double={item['double_occupancy_phase_over_pi']:+.12f} pi"
    )
print(json_path)
print(csv_path)
print(png_path)
