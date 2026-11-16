"""Minimal server-side example; requires only NumPy."""

import numpy as np

from berry_phase import compute_berry_phases


# Projection strengths: 0 = unprojected, 1 = fully Gutzwiller projected.
g_values = np.linspace(0.0, 1.0, 11)

berry_phases = compute_berry_phases(
    g_values,
    m=-2.0,
    t=1.0,
    t1=1.0,
    delta_p=1.0,
    delta_s=0.5,
    steps_per_leg=8,
    lx=4,
    ly=4,
)

# NumPy array with the same shape as g_values, containing phases in radians.
print(berry_phases)
