"""Twist-loop Berry phases for complex and cosine pairing conventions."""

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from model import ModelParameters, TwoOrbitalModel


PARAMETERS = ModelParameters(t=1.0, t1=1.0, m=-2.0, delta_p=1.0, delta_s=0.5)


def twist_loop(steps_per_leg):
    vertices = np.array([
        [0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi],
        [2.0 * np.pi, np.pi], [2.0 * np.pi, 2.0 * np.pi],
    ])
    return np.array([
        (1.0 - step / steps_per_leg) * vertices[leg]
        + (step / steps_per_leg) * vertices[leg + 1]
        for leg in range(4)
        for step in range(steps_per_leg)
    ])


def positive_blocks(h_bdg):
    n = h_bdg.shape[0] // 2
    energies, vectors = np.linalg.eigh(h_bdg)
    positive = energies > 1.0e-8
    if np.count_nonzero(positive) != n:
        raise ValueError("The BdG spectrum is gapless")
    vectors = vectors[:, positive]
    gap = float(min(np.min(energies[positive]), -np.max(energies[~positive])))
    return vectors[:n], vectors[n:], gap


def appendix_b_overlap(u1, v1, u2, v2):
    overlap_matrix = u1.conj().T @ u2 + v1.conj().T @ v2
    return complex(np.sqrt(np.linalg.det(overlap_matrix.conj().T)))


def compute(lx, ly, steps_per_leg, pairing_twist):
    model = TwoOrbitalModel(lx, ly, PARAMETERS)
    points = twist_loop(steps_per_leg)
    blocks = [
        positive_blocks(model.bdg_hamiltonian(*point, pairing_twist=pairing_twist))
        for point in points
    ]
    overlaps = [
        appendix_b_overlap(*blocks[index][:2], *blocks[(index + 1) % len(blocks)][:2])
        for index in range(len(blocks))
    ]
    unwrapped = math.fsum(float(np.angle(overlap)) for overlap in overlaps)
    phase = float(np.angle(np.exp(1j * unwrapped)))
    phase_mod_pi = float(unwrapped % np.pi)
    closure = np.max(np.abs(
        model.bdg_hamiltonian(2.0 * np.pi, 2.0 * np.pi, pairing_twist=pairing_twist)
        - model.bdg_hamiltonian(0.0, 0.0, pairing_twist=pairing_twist)
    ))
    return {
        "lx": lx,
        "ly": ly,
        "steps_per_leg": steps_per_leg,
        "pairing_twist": pairing_twist,
        "unwrapped_phase_radians": unwrapped,
        "berry_phase_radians": phase,
        "berry_phase_over_pi": phase / np.pi,
        "phase_mod_pi_radians": phase_mod_pi,
        "phase_mod_pi_over_pi": phase_mod_pi / np.pi,
        "distance_from_pi_over_2_mod_pi": phase_mod_pi - np.pi / 2.0,
        "minimum_link_magnitude": float(np.min(np.abs(overlaps))),
        "maximum_link_phase_magnitude": float(np.max(np.abs(np.angle(overlaps)))),
        "minimum_bdg_gap": float(min(block[2] for block in blocks)),
        "hamiltonian_closure_error": float(closure),
    }


def main():
    mesh_results = []
    for pairing in ("complex", "cosine"):
        for steps in (4, 8, 16, 32, 64, 128, 256):
            result = compute(4, 4, steps, pairing)
            mesh_results.append(result)
            print(
                f"{pairing:7s} 4x4 M={steps:3d}: "
                f"theta={result['berry_phase_radians']:+.12f}, "
                f"theta/pi={result['berry_phase_over_pi']:+.12f}, "
                f"err(mod pi)={result['distance_from_pi_over_2_mod_pi']:+.6e}",
                flush=True,
            )

    size_results = []
    for pairing in ("complex", "cosine"):
        for size in range(4, 10):
            result = compute(size, size, 32, pairing)
            size_results.append(result)
            print(
                f"{pairing:7s} {size}x{size} M=32: "
                f"theta={result['berry_phase_radians']:+.12f}, "
                f"theta/pi={result['berry_phase_over_pi']:+.12f}",
                flush=True,
            )

    payload = {
        "parameters": vars(PARAMETERS),
        "method": "Appendix-B Gaussian overlaps of positive-energy BdG eigenvectors",
        "mesh_convergence": mesh_results,
        "size_scan": size_results,
    }
    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
