"""
Observer Stability Minima Test
==============================

This script demonstrates a key idea of the Global Spectral TOE:

    In a single global spectral world, not all observer perspectives
    are equivalent. Some projections of the same world are objectively
    more stable than others.

Nothing in this code enforces stability, locality, or preferred observers.
The result emerges purely from spectral structure.

What this script shows:
- One global operator (one world)
- Many observers defined as projections of that world
- Each observer reconstructs an effective physics
- We measure how sensitive that physics is to a tiny change of perspective
- Some observers sit at clear minima of instability

If you see clear stability minima, this is the "wow" moment.
"""

import numpy as np

# ============================================================
# GLOBAL PARAMETERS
# ============================================================

np.random.seed(0)

DIM = 120               # dimension of the global world
VISIBLE_FRACTION = 0.3 # fraction of the world visible to an observer
N_OBSERVERS = 40        # number of observer perspectives to test

EPS = 1e-12
K_EIG = 6               # number of spectral modes used for effective physics

# ============================================================
# GLOBAL OPERATOR (ONE WORLD)
# ============================================================
#
# This is the only "world" in the script.
# All observers see projections of the SAME operator.
#

A = np.random.randn(DIM, DIM) * 0.1
for i in range(DIM):
    A[i, (i + 1) % DIM] += 1.0
    A[i, (i - 1) % DIM] += 1.0

B = np.random.randn(DIM, DIM) * 0.02
T = A + B

# ============================================================
# EFFECTIVE PHYSICS (FROM SPECTRUM)
# ============================================================
#
# These definitions follow the logic of the TOE:
#
# - time scale  ~ separation of leading eigenvalues
# - masses      ~ logarithmic gaps relative to the leading mode
# - arrow       ~ spectral asymmetry
#

def effective_physics(eigvals):
    lams = np.sort(np.abs(eigvals))[::-1]

    time_eff = np.log(lams[0] / (lams[1] + EPS) + EPS)
    arrow = np.sum(np.diff(np.log(lams + EPS)))

    masses = [
        -np.log(lams[i] / (lams[0] + EPS) + EPS)
        for i in range(1, 1 + K_EIG)
    ]

    return np.array([time_eff, arrow] + masses)

# ============================================================
# OBSERVER PROJECTION
# ============================================================
#
# Each observer is defined by a different projection of the same world.
# No observer is privileged by construction.
#

def observer_projection(observer_id):
    size = int(DIM * VISIBLE_FRACTION)
    shift = (observer_id * size) // 3
    start = (DIM // 2 - size // 2 + shift) % DIM
    return np.arange(start, start + size) % DIM

# ============================================================
# STABILITY FUNCTIONAL
# ============================================================
#
# Stability is defined as:
#
#   How much does effective physics change
#   under a tiny change of observer perspective?
#
# Lower value = more stable observer.
#

def stability_measure(observer_id):
    idx = observer_projection(observer_id)

    T_proj = T[np.ix_(idx, idx)]
    phys_0 = effective_physics(np.linalg.eigvals(T_proj))

    # minimal perturbation of the observer
    idx_shift = (idx + 1) % DIM
    T_proj_shift = T[np.ix_(idx_shift, idx_shift)]
    phys_1 = effective_physics(np.linalg.eigvals(T_proj_shift))

    return np.linalg.norm(phys_1 - phys_0), phys_0

# ============================================================
# RUN TEST
# ============================================================

print("==============================================")
print("OBSERVER STABILITY MINIMA TEST")
print("==============================================")
print("GLOBAL DIM =", DIM)
print("OBSERVERS  =", N_OBSERVERS)
print()

records = []

for obs in range(N_OBSERVERS):
    stability, phys = stability_measure(obs)
    records.append((obs, stability, phys))

    print(
        f"Observer {obs:02d} | "
        f"stability = {stability:9.3e} | "
        f"time = {phys[0]:8.4e} | "
        f"arrow = {phys[1]:7.3f}"
    )

# ============================================================
# ANALYSIS: STABILITY LANDSCAPE
# ============================================================

stabilities = np.array([r[1] for r in records])

print("\n==============================================")
print("STABILITY LANDSCAPE SUMMARY")
print("==============================================")
print("Min stability  =", stabilities.min())
print("Max stability  =", stabilities.max())
print("Mean stability =", stabilities.mean())
print("Std stability  =", stabilities.std())

# most stable observers
stable_indices = np.argsort(stabilities)[:5]

print("\nMost stable observers (natural perspectives):")
for i in stable_indices:
    obs, stab, phys = records[i]
    print(
        f"Observer {obs:02d} | "
        f"stability = {stab:.3e} | "
        f"time = {phys[0]:.4e}"
    )

print("==============================================")

"""
Expected qualitative outcome
----------------------------

You should observe:

- Clear minima of observer instability
- Repetition of the same stable observers
- No tuning, learning, or selection mechanisms

Interpretation:

    The global world is unique.
    Observer perspectives are not.

Some perspectives reconstruct effective physics
that is significantly more stable than others.

This suggests that what we perceive as "physical laws"
may be projections selected by stability,
not fundamental absolutes.
"""
