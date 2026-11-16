"""Plot the dense BdG spectrum along the four-leg twist path."""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exact_gutzwiller"))
import model as gm


def main():
    lx = ly = 4
    steps_per_leg = 200
    params = gm.ModelParameters(t=1.0, t1=1.0, m=-2.0, delta_p=1.0, delta_s=0.5)
    model = gm.TwoOrbitalModel(lx, ly, params)
    vertices = np.array([
        [0.0, 0.0], [np.pi, 0.0], [np.pi, np.pi],
        [2 * np.pi, np.pi], [2 * np.pi, 2 * np.pi],
    ])

    points = []
    leg_numbers = []
    leg_steps = []
    for leg in range(4):
        for step in range(steps_per_leg + 1):
            fraction = step / steps_per_leg
            points.append((1 - fraction) * vertices[leg] + fraction * vertices[leg + 1])
            leg_numbers.append(leg)
            leg_steps.append(step)
    points = np.asarray(points)
    energies = np.array([
        np.linalg.eigvalsh(model.bdg_hamiltonian(*point, pairing_twist="complex"))
        for point in points
    ])

    absolute_energies = np.abs(energies)
    flat_index = int(np.argmin(absolute_energies))
    path_index, level_index = np.unravel_index(flat_index, energies.shape)
    minimum_gap = float(absolute_energies[path_index, level_index])
    minimum_point = points[path_index]
    summary = {
        "lattice": [lx, ly],
        "parameters": {"t": 1.0, "t1": 1.0, "m": -2.0, "delta_p": 1.0, "delta_s": 0.5},
        "pairing_convention": "original complex antisymmetric reverse-bond pairing: exp(i*phi)",
        "steps_per_leg": steps_per_leg,
        "number_of_path_points_including_leg_endpoints": len(points),
        "number_of_bdg_levels": energies.shape[1],
        "minimum_absolute_bdg_energy": minimum_gap,
        "minimum_gap_twist": minimum_point.tolist(),
        "minimum_gap_path_index": int(path_index),
        "minimum_gap_leg": int(leg_numbers[path_index]),
        "level_crossing_detected": bool(minimum_gap < 1e-8),
    }

    output = Path(__file__).resolve().parent
    np.savez_compressed(
        output / "twist_spectrum_4x4.npz",
        points=points, energies=energies,
        leg_numbers=np.asarray(leg_numbers), leg_steps=np.asarray(leg_steps),
    )
    (output / "twist_spectrum_4x4.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    titles = [
        r"Leg 1: $(\phi_x,0)$, $\phi_y=0$",
        r"Leg 2: $(\pi,\phi_y)$, $\phi_x=\pi$",
        r"Leg 3: $(\phi_x,\pi)$, $\phi_y=\pi$",
        r"Leg 4: $(2\pi,\phi_y)$, $\phi_x=2\pi$",
    ]
    xlabels = [r"$\phi_x/\pi$", r"$\phi_y/\pi$", r"$\phi_x/\pi$", r"$\phi_y/\pi$"]
    starts = [0.0, 0.0, 1.0, 1.0]
    for leg, ax in enumerate(axes.flat):
        mask = np.asarray(leg_numbers) == leg
        x = points[mask, 0 if leg in (0, 2) else 1] / np.pi
        ax.plot(x, energies[mask], color="#2563a5", lw=0.55, alpha=0.55)
        ax.axhline(0.0, color="#c0392b", lw=1.0, ls="--")
        ax.set_title(titles[leg])
        ax.set_xlabel(xlabels[leg])
        ax.set_xlim(starts[leg], starts[leg] + 1.0)
        ax.grid(alpha=0.18)
        ax.set_ylabel("BdG energy" if leg in (0, 2) else "")
    fig.suptitle(
        rf"BdG spectrum on the twist path, $4\times4$, $m=-2$; "
        rf"minimum $ |E|={minimum_gap:.6f}$",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output / "twist_spectrum_4x4.png", dpi=220)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
