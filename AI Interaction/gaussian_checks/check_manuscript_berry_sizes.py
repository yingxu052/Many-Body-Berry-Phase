"""Converged Berry phases for the independent Eqs. (A5)-(A6) transcription."""

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server_ready"))
sys.path.insert(0, str(ROOT / "gaussian_checks"))

import gutzwiller_mc_sweep as gm
from appendix_b_overlap import appendix_b_overlap
from manuscript_twist_hamiltonian import bdg_manuscript


def calculate(lx, ly, steps_per_leg):
    model = gm.TwoOrbitalModel(
        lx, ly,
        gm.ModelParameters(t=1.0, t1=1.0, m=-2.0, delta_p=1.0, delta_s=0.5),
    )
    points = gm.twist_loop(steps_per_leg)
    blocks = [
        gm.positive_blocks(bdg_manuscript(model, *point, seam="first"))
        for point in points
    ]
    overlaps = [
        appendix_b_overlap(
            blocks[i][0], blocks[i][1],
            blocks[(i + 1) % len(blocks)][0], blocks[(i + 1) % len(blocks)][1],
        )
        for i in range(len(blocks))
    ]
    raw_phase = math.fsum(float(np.angle(overlap)) for overlap in overlaps)
    return {
        "lx": lx,
        "ly": ly,
        "steps_per_leg": steps_per_leg,
        "raw_phase_radians": raw_phase,
        "raw_phase_over_pi": raw_phase / np.pi,
        "phase_mod_2pi_radians": float(np.angle(np.exp(1j * raw_phase))),
        "phase_mod_pi_radians": float(raw_phase % np.pi),
        "phase_mod_pi_over_pi": float((raw_phase % np.pi) / np.pi),
        "minimum_link_magnitude": float(np.min(np.abs(overlaps))),
        "minimum_bdg_gap": float(min(block[2] for block in blocks)),
    }


def main():
    results = []
    for size in (4, 5, 6, 7):
        for steps in (16, 32):
            result = calculate(size, size, steps)
            results.append(result)
            print(json.dumps(result), flush=True)
    output = ROOT / "gaussian_checks" / "manuscript_cosine_twist_berry_sizes.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
