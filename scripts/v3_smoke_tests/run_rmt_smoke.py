#!/usr/bin/env python3
"""
Version 3 RMT smoke test — tooling / statistics sanity check.
================================================================

This is a SMALL, FAST, SELF-CONTAINED sanity test. It is NOT a reproduction
of the full published RMT result and it does NOT require a GPU.

What it does:
  1. Generates small synthetic Poisson and GOE spectra (pure NumPy).
  2. Computes the adjacent-gap-ratio statistic ⟨r⟩ the same way the repo's
     `rmt/ensemble_references.py` does.
  3. Checks that the two references land near their textbook values and are
     well separated (Poisson ≈ 0.386 < GOE ≈ 0.531). This validates that the
     r-statistic machinery itself is behaving.
  4. Builds a tiny CPU copy of the TOE excitation operator (ring Laplacian +
     Arnold-cat diagonal phase, ported from `rmt/excitation_operator.py`) and
     reports its ⟨r⟩, which should sit in the intermediate / pseudo-integrable
     window between Poisson and GOE — consistent with the documented RMT
     finding, but at toy size and WITHOUT claiming the full result.

What it does NOT do:
  - It does not run the GPU scaling sweep (`rmt/scaling_tests.py`, CuPy).
  - It does not do unfolding, finite-size scaling, or Brody/semi-Poisson fits.
  - It does not, by itself, establish the published intermediate-statistics
    claim. See docs/RMT_NUMERICAL_AUDIT_PLAN.md for the full audit scope.

Exit code 0 = sanity checks passed (references behave). Exit code 1 = a
reference statistic fell outside a generous sanity band (tooling problem).

Author: Grzegorz Olbryk · accompanies Zenodo Version 3 (DOI 10.5281/zenodo.20374769)
"""

import argparse
import sys

import numpy as np


# --- r-statistic, matching rmt/ensemble_references.py exactly (NumPy port) ---

def r_statistic(eigs):
    """Mean adjacent-gap ratio ⟨min(s_i,s_{i+1})/max(s_i,s_{i+1})⟩."""
    eigs = np.sort(np.asarray(eigs, dtype=np.float64))
    s = eigs[1:] - eigs[:-1]
    s = s / np.mean(s)
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    r = r[np.isfinite(r)]
    return float(np.mean(r))


def goe_eigenvalues(N, rng):
    A = rng.normal(0.0, 1.0, (N, N))
    H = (A + A.T) / np.sqrt(2.0 * N)
    return np.sort(np.linalg.eigvalsh(H))


def poisson_eigenvalues(N, rng):
    return np.sort(np.cumsum(rng.exponential(1.0, N)))


# --- tiny CPU copy of the TOE excitation operator (cf. excitation_operator.py) ---

def arnold_cat_phase(N, x0=0.1, y0=0.3):
    xs = np.empty(N, dtype=np.float64)
    x, y = float(x0), float(y0)
    for n in range(N):
        xs[n] = x
        x, y = (2.0 * x + y) % 1.0, (x + y) % 1.0
    return np.cos(2.0 * np.pi * xs)


def build_excitation_operator(N, alpha):
    L = np.zeros((N, N), dtype=np.float64)
    idx = np.arange(N)
    L[idx, idx] = 2.0
    L[idx, (idx + 1) % N] = -1.0
    L[idx, (idx - 1) % N] = -1.0
    L[idx, idx] += alpha * arnold_cat_phase(N)
    return L


# Reference values and generous sanity bands (toy-size, so bands are wide).
POISSON_REF = 0.386
GOE_REF = 0.531
POISSON_BAND = (0.34, 0.44)
GOE_BAND = (0.47, 0.58)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=600, help="matrix size (default 600)")
    ap.add_argument("--alpha", type=float, default=0.30,
                    help="excitation-operator coupling (default 0.30)")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed (default 123)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    N = args.N

    print("=" * 64)
    print("Global Spectral TOE — Version 3 RMT smoke test (sanity only)")
    print("=" * 64)
    print(f"NumPy {np.__version__} | N={N} | alpha={args.alpha} | seed={args.seed}")
    print("(GPU not required; this is a toy-size tooling check, not the full result)")
    print()

    r_poi = r_statistic(poisson_eigenvalues(N, rng))
    r_goe = r_statistic(goe_eigenvalues(N, rng))
    r_toe = r_statistic(np.linalg.eigvalsh(build_excitation_operator(N, args.alpha)))

    print(f"{'ensemble':<22}{'<r>':>8}   {'reference':>12}")
    print("-" * 46)
    print(f"{'Poisson (synthetic)':<22}{r_poi:>8.4f}   {POISSON_REF:>12.4f}")
    print(f"{'GOE (synthetic)':<22}{r_goe:>8.4f}   {GOE_REF:>12.4f}")
    print(f"{'TOE excitation op':<22}{r_toe:>8.4f}   {'intermediate':>12}")
    print()

    ok = True
    if not (POISSON_BAND[0] <= r_poi <= POISSON_BAND[1]):
        print(f"FAIL: Poisson <r>={r_poi:.4f} outside sanity band {POISSON_BAND}")
        ok = False
    if not (GOE_BAND[0] <= r_goe <= GOE_BAND[1]):
        print(f"FAIL: GOE <r>={r_goe:.4f} outside sanity band {GOE_BAND}")
        ok = False
    if not (r_poi < r_goe):
        print("FAIL: Poisson <r> is not below GOE <r> (ordering broken)")
        ok = False

    print("-" * 46)
    if ok:
        print("SANITY OK: r-statistic tooling behaves; Poisson < GOE as expected.")
        if r_poi < r_toe < r_goe:
            print("NOTE: TOE excitation operator lands BETWEEN Poisson and GOE")
            print("      (intermediate / pseudo-integrable), consistent with the")
            print("      documented RMT finding — at toy size, NOT the full result.")
        else:
            print("NOTE: at this toy N the TOE <r> did not cleanly separate; this")
            print("      smoke test only validates tooling, not the published claim.")
        print()
        print("This is a SMOKE TEST. It does not reproduce the full public RMT")
        print("result. See docs/V3_REPRODUCIBILITY.md and")
        print("docs/RMT_NUMERICAL_AUDIT_PLAN.md.")
        return 0
    else:
        print("SMOKE TEST FAILED: a reference statistic misbehaved (tooling issue).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
