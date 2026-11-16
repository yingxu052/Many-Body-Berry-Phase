"""Test whether weak quenched disorder quantizes Gaussian Berry phases.

Two tests are performed:
1. A generic one-mode (two-level) Gaussian ground state with an analytically
   known, continuously variable Berry phase.
2. The two-orbital BdG twist loop used elsewhere in this project, with a fixed
   random on-site potential added to every Hamiltonian along the loop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gaussian_checks"))
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import (
    PARAMETERS,
    appendix_b_overlap,
    positive_blocks,
    twist_loop,
)
from model import TwoOrbitalModel


SEED = 20261106


def wrap_period(values, period):
    """Wrap values to [-period/2, period/2)."""
    return (np.asarray(values) + period / 2) % period - period / 2


def circular_mean_and_rms(phases, period):
    phases = np.asarray(phases, dtype=float)
    z = np.mean(np.exp(2j * np.pi * phases / period))
    mean = float((np.angle(z) * period / (2 * np.pi)) % period)
    deviations = wrap_period(phases - mean, period)
    return mean, float(np.sqrt(np.mean(deviations**2))), float(abs(z))


def generic_gaussian_phase(theta, onsite):
    """Ground-state Berry phase, in [0, 2*pi), for the loop in the docstring below.

    h(lambda) = sin(theta)(cos(lambda)sigma_x + sin(lambda)sigma_y)
                + (cos(theta) + onsite)sigma_z.
    In this project's Im-log-product-overlaps convention, the lower
    eigenstate has gamma = -Omega/2 modulo 2*pi.
    """
    d_z = np.cos(theta) + np.asarray(onsite)
    cos_alpha = d_z / np.sqrt(np.sin(theta) ** 2 + d_z**2)
    return np.pi * (1.0 + cos_alpha)


def check_generic_phase_numerically(theta=0.85, onsite=0.0):
    """Independent link-overlap check of the analytic Bloch-solid-angle formula."""
    steps = 4096
    lambdas = 2 * np.pi * np.arange(steps) / steps
    vectors = []
    for lam in lambdas:
        h = np.array([
            [np.cos(theta) + onsite, np.sin(theta) * np.exp(-1j * lam)],
            [np.sin(theta) * np.exp(1j * lam), -(np.cos(theta) + onsite)],
        ], dtype=complex)
        _, eigenvectors = np.linalg.eigh(h)
        vectors.append(eigenvectors[:, 0])
    overlaps = [
        np.vdot(vectors[i], vectors[(i + 1) % steps]) for i in range(steps)
    ]
    numerical = math.fsum(float(np.angle(overlap)) for overlap in overlaps) % (2 * np.pi)
    analytic = float(generic_gaussian_phase(theta, onsite))
    error = abs(wrap_period(numerical - analytic, 2 * np.pi))
    if error > 1.0e-6:
        raise RuntimeError(
            "Generic-phase formula failed its numerical check: "
            f"analytic={analytic}, link_overlap={numerical}, error={error}"
        )
    return {"analytic": analytic, "link_overlap": numerical, "error": float(error)}


def generic_disorder_scan():
    theta = 0.85
    strengths = np.linspace(0.0, 0.2, 41)
    nodes, weights = np.polynomial.legendre.leggauss(512)
    rows = []
    for strength in strengths:
        if strength == 0.0:
            mean = float(generic_gaussian_phase(theta, 0.0))
            second = mean**2
        else:
            onsite = strength * nodes
            phases = generic_gaussian_phase(theta, onsite)
            mean = float(np.sum(weights * phases) / 2.0)
            second = float(np.sum(weights * phases**2) / 2.0)
        rows.append({
            "W": float(strength),
            "mean_phase": mean,
            "mean_phase_over_pi": mean / np.pi,
            "rms_about_mean": math.sqrt(max(0.0, second - mean**2)),
        })
    return {
        "model": "one occupied mode of h=d.sigma; d=(sin(theta)cos(lambda), sin(theta)sin(lambda), cos(theta)+V)",
        "theta": theta,
        "distribution": "V uniform on [-W,W], fixed along lambda",
        "clean_phase_over_pi": float(generic_gaussian_phase(theta, 0.0) / np.pi),
        "formula_check": check_generic_phase_numerically(theta),
        "rows": rows,
    }


def disordered_bdg_hamiltonian(model, point, onsite):
    h_bdg = model.bdg_hamiltonian(*point, pairing_twist="cosine")
    n = model.n_modes
    indices = np.arange(n)
    # H_BdG = [[h, Delta], [-Delta*, -h^T]].  Thus an on-site h_ii=V_i
    # appears as +V_i in the particle block and -V_i in the hole block.
    h_bdg[indices, indices] += onsite
    h_bdg[n + indices, n + indices] -= onsite
    return h_bdg


def bdg_loop_phase(model, points, onsite):
    blocks = [
        positive_blocks(disordered_bdg_hamiltonian(model, point, onsite))
        for point in points
    ]
    overlaps = [
        appendix_b_overlap(*blocks[index][:2], *blocks[(index + 1) % len(blocks)][:2])
        for index in range(len(blocks))
    ]
    total = math.fsum(float(np.angle(overlap)) for overlap in overlaps)
    return {
        "phase_mod_pi": float(total % np.pi),
        "minimum_link_magnitude": float(np.min(np.abs(overlaps))),
        "minimum_bdg_gap": float(min(block[2] for block in blocks)),
    }


def bdg_disorder_scan(lx=4, ly=4, steps=32, samples=64):
    model = TwoOrbitalModel(lx, ly, PARAMETERS)
    points = twist_loop(steps)
    zero = np.zeros(model.n_modes)
    clean = bdg_loop_phase(model, points, zero)
    rng = np.random.default_rng(SEED)
    base_disorder = rng.uniform(-1.0, 1.0, size=(samples, model.n_modes))
    rows = []
    for strength in (0.05, 0.1, 0.2):
        results = [
            bdg_loop_phase(model, points, strength * disorder)
            for disorder in base_disorder
        ]
        phases = np.array([result["phase_mod_pi"] for result in results])
        mean, rms, coherence = circular_mean_and_rms(phases, np.pi)
        lifted = clean["phase_mod_pi"] + wrap_period(phases - clean["phase_mod_pi"], np.pi)
        row = {
            "W": strength,
            "samples": samples,
            "mean_phase_mod_pi": mean,
            "mean_phase_over_pi": mean / np.pi,
            "circular_rms": rms,
            "circular_coherence": coherence,
            "lifted_arithmetic_mean_over_pi": float(np.mean(lifted) / np.pi),
            "minimum_bdg_gap": float(min(result["minimum_bdg_gap"] for result in results)),
            "minimum_link_magnitude": float(min(result["minimum_link_magnitude"] for result in results)),
            "phases_mod_pi": phases.tolist(),
            "sample_results": results,
        }
        rows.append(row)
        print(
            f"BdG W={strength:.2f}: mean/pi={mean / np.pi:.9f}, "
            f"RMS={rms:.6f}, min gap={row['minimum_bdg_gap']:.6f}, "
            f"min |link|={row['minimum_link_magnitude']:.6f}",
            flush=True,
        )
    return {
        "lattice": [lx, ly],
        "steps_per_leg": steps,
        "samples_per_strength": samples,
        "seed": SEED,
        "parameters": vars(PARAMETERS),
        "pairing_twist": "cosine",
        "distribution": "independent on-site V_i uniform on [-W,W] for every orbital; fixed along the loop",
        "averaging": "circular mean modulo pi; RMS uses wrapped deviations modulo pi",
        "clean": clean,
        "rows": rows,
    }


def make_figure(generic, bdg, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    strengths = np.array([row["W"] for row in generic["rows"]])
    means = np.array([row["mean_phase_over_pi"] for row in generic["rows"]])
    rms = np.array([row["rms_about_mean"] for row in generic["rows"]]) / np.pi
    ax.fill_between(strengths, means - rms, means + rms, alpha=0.25, label="mean ± RMS")
    ax.plot(strengths, means, lw=2, label="disorder-averaged phase")
    ax.set_xlabel(r"disorder strength $W/t$")
    ax.set_ylabel(r"$\gamma/\pi$")
    ax.set_title("Generic Gaussian loop: no quantization")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    rng = np.random.default_rng(SEED + 1)
    for index, row in enumerate(bdg["rows"]):
        x = row["W"] + rng.uniform(-0.0025, 0.0025, size=row["samples"])
        phases = np.asarray(row["phases_mod_pi"]) / np.pi
        ax.scatter(x, phases, s=12, alpha=0.35)
        ax.errorbar(
            row["W"], row["mean_phase_over_pi"], yerr=row["circular_rms"] / np.pi,
            fmt="ko", ms=5, capsize=3, lw=1.2,
        )
    ax.axhline(bdg["clean"]["phase_mod_pi"] / np.pi, color="black", ls="--", lw=1, label=r"clean $\pi/2$")
    ax.set_xlabel(r"disorder strength $W/t$")
    ax.set_ylabel(r"$\gamma\;\mathrm{mod}\;\pi$, units of $\pi$")
    ax.set_title("Project BdG twist loop: ensemble averaging")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    generic = generic_disorder_scan()
    print(
        f"Generic Gaussian clean phase/pi={generic['clean_phase_over_pi']:.9f}; "
        f"at W=0.2, mean/pi={generic['rows'][-1]['mean_phase_over_pi']:.9f}",
        flush=True,
    )
    bdg = bdg_disorder_scan()
    payload = {"generic_gaussian": generic, "project_bdg_twist_loop": bdg}
    json_output = Path(__file__).resolve().with_suffix(".json")
    json_output.write_text(json.dumps(payload, indent=2) + "\n")
    figure_output = Path(__file__).resolve().with_suffix(".png")
    make_figure(generic, bdg, figure_output)
    print(f"Saved {json_output}")
    print(f"Saved {figure_output}")


if __name__ == "__main__":
    main()
