"""Remove legwise real-gauge phases and isolate the four vertex Berry-phase contributions."""

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exact_gutzwiller"))
from complex_twist_berry_convergence import PARAMETERS, twist_loop
from gaussian_overlap import normalized_gaussian_overlap
from model import TwoOrbitalModel, gaussian_bcs_state
from single_occupancy_overlap import single_occupancy_overlap
from validate_gaussian_overlap import basis_change_sign


REFERENCE_MASK = 0xAAC8DA33


def pairing_from_blocks(u, v, reference_mask):
    u = u.copy()
    v = v.copy()
    for mode in range(u.shape[0]):
        if reference_mask & (1 << mode):
            u[mode, :], v[mode, :] = v[mode, :], u[mode, :]
    z = -np.linalg.solve(u.conj().T, v.conj().T)
    return 0.5 * (z - z.T)


def pfaffian(matrix):
    """LTL Pfaffian, matching the exact-enumeration convention."""
    n = matrix.shape[0]
    if n == 0:
        return 1.0 + 0.0j
    if n % 2:
        return 0.0 + 0.0j
    a = np.array(matrix, dtype=complex, copy=True)
    result = 1.0 + 0.0j
    for k in range(0, n - 1, 2):
        pivot = k + 1 + np.argmax(np.abs(a[k, k + 1:]))
        if abs(a[k, pivot]) < 1.0e-30:
            return 0.0 + 0.0j
        if pivot != k + 1:
            a[[k + 1, pivot], :] = a[[pivot, k + 1], :]
            a[:, [k + 1, pivot]] = a[:, [pivot, k + 1]]
            result = -result
        pivot_value = a[k, k + 1]
        result *= pivot_value
        x = a[k, k + 2:].copy()
        y = a[k + 1, k + 2:].copy()
        for i in range(k + 2, n):
            for j in range(i + 1, n):
                a[i, j] += (y[i - k - 2] * x[j - k - 2]
                            - x[i - k - 2] * y[j - k - 2]) / pivot_value
                a[j, i] = -a[i, j]
    return result


def reference_action(model, direction):
    """Return R^-1|reference> and the phase <reference|U R^-1|reference>."""
    reference_modes = [mode for mode in range(model.n_modes) if REFERENCE_MASK & (1 << mode)]
    source_modes = []
    for mode in reference_modes:
        cell, orbital = divmod(mode, 2)
        x, y = cell % model.lx, cell // model.lx
        if direction == "x":
            source_modes.append(model.mode((1 - x) % model.lx, y, orbital))
        else:
            source_modes.append(model.mode(x, (1 - y) % model.ly, orbital))
    source_modes.sort()
    source_mask = sum(1 << mode for mode in source_modes)
    mapped_modes = []
    unitary_phase = 1.0 + 0.0j
    for mode in source_modes:
        cell, orbital = divmod(mode, 2)
        x, y = cell % model.lx, cell // model.lx
        if direction == "x":
            mapped_modes.append(model.mode((1 - x) % model.lx, y, orbital))
        else:
            mapped_modes.append(model.mode(x, (1 - y) % model.ly, orbital))
            unitary_phase *= 1.0j if orbital == 0 else -1.0j
    inversions = sum(
        mapped_modes[i] > mapped_modes[j]
        for i in range(len(mapped_modes)) for j in range(i + 1, len(mapped_modes))
    )
    if inversions % 2:
        unitary_phase = -unitary_phase
    if set(mapped_modes) != set(reference_modes):
        raise RuntimeError("Reflection did not return the reference configuration")
    return source_mask, unitary_phase


def state_and_antiunitary_phase(hamiltonian, action):
    n = hamiltonian.shape[0] // 2
    z = gaussian_bcs_state(hamiltonian, reference_mask=REFERENCE_MASK).z
    source_mask, unitary_phase = action
    transformed_mask = source_mask ^ REFERENCE_MASK
    occupied = [mode for mode in range(n) if transformed_mask & (1 << mode)]
    amplitude = complex(basis_change_sign(source_mask, REFERENCE_MASK, n))
    if occupied:
        amplitude *= pfaffian(z[np.ix_(occupied, occupied)])
    # <reference|Theta psi> = unitary_phase * amplitude(source_mask)^*.
    alpha = unitary_phase * amplitude.conjugate()
    return z, alpha


def reflection_symmetries(model):
    return reference_action(model, "x"), reference_action(model, "y")


def compute(steps_per_leg, mixing=0.5, projected=False):
    model = TwoOrbitalModel(4, 4, PARAMETERS)
    symmetry_x, symmetry_y = reflection_symmetries(model)
    points = twist_loop(steps_per_leg)
    states = []
    alphas = []
    for index, point in enumerate(points):
        linear = model.bdg_hamiltonian(*point, pairing_twist="linear")
        cosine = model.bdg_hamiltonian(*point, pairing_twist="cosine")
        hamiltonian = (1.0 - mixing) * linear + mixing * cosine
        leg = index // steps_per_leg
        symmetry = symmetry_x if leg in (0, 2) else symmetry_y
        z, alpha = state_and_antiunitary_phase(hamiltonian, symmetry)
        states.append(z)
        alphas.append(alpha)

    count = len(points)
    if projected:
        overlaps = [
            single_occupancy_overlap(
                states[index], states[(index + 1) % count], model.n_cells,
                reference_mask1=REFERENCE_MASK, reference_mask2=REFERENCE_MASK,
                workers=8,
            )["normalized_overlap"]
            for index in range(count)
        ]
    else:
        overlaps = [
            normalized_gaussian_overlap(states[index], states[(index + 1) % count])
            for index in range(count)
        ]
    original_phases = np.angle(overlaps)
    base_gauge_phases = -0.5 * np.angle(alphas)
    gauge_phases = base_gauge_phases.copy()

    # Within each leg, choose the real-gauge sign continuously. Do not adjust
    # the target of the last link: that link crosses into the next real structure.
    for leg in range(4):
        start = leg * steps_per_leg
        for step in range(1, steps_per_leg):
            current = start + step
            previous = current - 1
            desired = gauge_phases[previous] + original_phases[previous]
            shift = np.round((desired - gauge_phases[current]) / np.pi) * np.pi
            gauge_phases[current] += shift

    gauge_differences = np.roll(gauge_phases, -1) - gauge_phases
    stripped_overlaps = np.asarray(overlaps) * np.exp(-1j * gauge_differences)
    stripped_phases = np.angle(stripped_overlaps)
    vertex_indices = [(leg + 1) * steps_per_leg - 1 for leg in range(4)]
    interior_mask = np.ones(count, dtype=bool)
    interior_mask[vertex_indices] = False
    vertex_phases = stripped_phases[vertex_indices]
    original_total = math.fsum(original_phases)
    stripped_total = math.fsum(stripped_phases)
    vertex_total = math.fsum(vertex_phases)
    return {
        "steps_per_leg": steps_per_leg,
        "mixing": mixing,
        "projection": "one_parton_per_cell" if projected else "none",
        "alpha_phases": np.angle(alphas).tolist(),
        "alpha_magnitudes": np.abs(alphas).tolist(),
        "original_link_phases": original_phases.tolist(),
        "stripped_link_phases": stripped_phases.tolist(),
        "original_total_phase": original_total,
        "original_berry_phase": float(np.angle(np.exp(1j * original_total))),
        "maximum_antiunitary_overlap_unit_error": float(max(abs(abs(alpha) - 1.0) for alpha in alphas)),
        "maximum_interior_stripped_phase": float(np.max(np.abs(stripped_phases[interior_mask]))),
        "maximum_interior_stripped_imaginary_overlap": float(
            np.max(np.abs(stripped_overlaps[interior_mask].imag))
        ),
        "vertex_link_indices": vertex_indices,
        "vertex_phases": vertex_phases.tolist(),
        "vertex_total_phase": vertex_total,
        "vertex_berry_phase": float(np.angle(np.exp(1j * vertex_total))),
        "stripped_total_phase": stripped_total,
        "total_phase_closure_error_mod_2pi": abs(
            np.angle(np.exp(1j * (original_total - stripped_total)))
        ),
        "leg_original_phases": [
            math.fsum(original_phases[leg * steps_per_leg:(leg + 1) * steps_per_leg])
            for leg in range(4)
        ],
        "leg_gauge_phase_changes": [
            gauge_phases[(leg + 1) * steps_per_leg % count]
            - gauge_phases[leg * steps_per_leg]
            for leg in range(4)
        ],
    }


def main():
    results = []
    jobs = [
        (mixing, steps, False)
        for mixing in (0.0, 0.5, 1.0)
        for steps in (4, 8, 16, 32)
    ]
    jobs.append((0.5, 4, True))
    for mixing, steps, projected in jobs:
        result = compute(steps, mixing=mixing, projected=projected)
        results.append(result)
        print(
            f"mixing={mixing:.1f}, M={steps:2d}, projected={projected}: "
            f"original={result['original_berry_phase']:+.12f}, "
            f"vertex={result['vertex_berry_phase']:+.12f}, "
            f"max interior phase={result['maximum_interior_stripped_phase']:.3e}, "
            f"max |alpha|-1={result['maximum_antiunitary_overlap_unit_error']:.3e}",
            flush=True,
        )
        print(f"  vertex phases: {result['vertex_phases']}")
    output = Path(__file__).resolve().with_suffix(".json")
    output.write_text(json.dumps({
        "real_gauge": "|psi>=exp(i*vartheta)|R>, Theta_leg|R>=|R>",
        "reference_mask": REFERENCE_MASK,
        "results": results,
    }, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
