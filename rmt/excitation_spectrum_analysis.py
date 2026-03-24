"""
Excitation Spectrum of the Spectral TOE
========================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 9 — Construct effective theory of excitations

Given the Hessian structure from Task 8:
    H_{(c,k),(d,j)} = 4W_c δ_{cd} δ_{kj} - 4w_{cd}

The excitation spectrum decomposes into:
1. Internal modes (within sectors): λ = 4W_c, degeneracy N_c - 1
2. Uniform modes (inter-sector): n_sectors eigenvalues from reduced matrix
3. One zero mode (global shift)

This script computes the full spectrum and analyzes:
- Mass hierarchy from sector structure
- Degeneracy patterns
- Whether mass ratios could match physical observations
"""

import numpy as np
from scipy import linalg as la


def build_hessian(w, N_modes):
    """Build the full Hessian at the trivial vacuum (w symmetric, w_{aa} = 0)."""
    n_sectors = len(N_modes)
    N_total = sum(N_modes)
    H = np.zeros((N_total, N_total))

    W = np.array([sum(w[c][b] * N_modes[b] for b in range(n_sectors))
                  for c in range(n_sectors)])

    offsets = [0]
    for n in N_modes:
        offsets.append(offsets[-1] + n)

    for c in range(n_sectors):
        for d in range(n_sectors):
            for k in range(N_modes[c]):
                for j in range(N_modes[d]):
                    i = offsets[c] + k
                    jj = offsets[d] + j
                    if c == d and k == j:
                        H[i, jj] = 4 * W[c]
                    H[i, jj] -= 4 * w[c][d]

    return H


def analytic_spectrum(w, N_modes):
    """Compute the excitation spectrum analytically.

    Returns (eigenvalues, descriptions) sorted by eigenvalue.
    """
    n_sectors = len(N_modes)
    W = np.array([sum(w[c][b] * N_modes[b] for b in range(n_sectors))
                  for c in range(n_sectors)])

    spectrum = []

    # 1. Internal modes: λ = 4W_c, degeneracy N_c - 1
    for c in range(n_sectors):
        if N_modes[c] > 1:
            spectrum.append({
                'eigenvalue': 4 * W[c],
                'degeneracy': N_modes[c] - 1,
                'type': f'internal (sector {c})',
                'description': f'Internal mode of sector {c}: fluctuations '
                               f'within sector, orthogonal to sector mean'
            })

    # 2. Uniform modes: reduce to n_sectors × n_sectors problem
    # The reduced matrix is: M'_{cd} = 4W_c δ_{cd} - 4w_{cd} N_d
    M_red = np.zeros((n_sectors, n_sectors))
    for c in range(n_sectors):
        for d in range(n_sectors):
            if c == d:
                M_red[c, d] = 4 * W[c] - 4 * w[c][c] * N_modes[c]
            else:
                M_red[c, d] = -4 * w[c][d] * N_modes[d]

    # Note: M_red is generally NOT symmetric (w_{cd}N_d ≠ w_{dc}N_c unless N_c = N_d)
    eigvals_red = np.sort(np.real(la.eigvals(M_red)))

    for i, ev in enumerate(eigvals_red):
        if abs(ev) < 1e-10:
            spectrum.append({
                'eigenvalue': 0.0,
                'degeneracy': 1,
                'type': 'zero mode (Goldstone)',
                'description': 'Global shift mode: ℓ → ℓ + const'
            })
        else:
            spectrum.append({
                'eigenvalue': ev,
                'degeneracy': 1,
                'type': f'uniform mode {i}',
                'description': f'Inter-sector collective mode: all modes in each '
                               f'sector move together'
            })

    spectrum.sort(key=lambda x: x['eigenvalue'])
    return spectrum


def mass_from_eigenvalue(lam):
    """Mass ∝ √λ (harmonic approximation)."""
    if lam <= 0:
        return 0.0
    return np.sqrt(lam)


def main():
    print()
    print("=" * 90)
    print("  Excitation Spectrum of the Spectral TOE")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Task 9")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # Model 1: 3 sectors (motivated by generations?)
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Model 1: Three sectors (generic weights)")
    print("-" * 70)

    N_modes_1 = [10, 15, 8]
    w_1 = np.array([
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.7],
        [0.5, 0.7, 0.0],
    ])

    # Full Hessian (numerical)
    H_1 = build_hessian(w_1, N_modes_1)
    eigvals_full = np.sort(la.eigvalsh(H_1))

    # Analytic spectrum
    spectrum_1 = analytic_spectrum(w_1, N_modes_1)

    print(f"\n  N_modes = {N_modes_1}, total = {sum(N_modes_1)}")
    print(f"  Weights: w_01 = {w_1[0,1]}, w_02 = {w_1[0,2]}, w_12 = {w_1[1,2]}")

    print(f"\n  Analytic spectrum:")
    print(f"  {'λ':>10} {'degen':>6} {'mass':>10} {'type'}")
    print(f"  " + "-" * 60)
    for s in spectrum_1:
        m = mass_from_eigenvalue(s['eigenvalue'])
        print(f"  {s['eigenvalue']:10.4f} {s['degeneracy']:6d} {m:10.4f} {s['type']}")

    print(f"\n  Full Hessian eigenvalues (first 10): {eigvals_full[:10].round(4)}")
    print(f"  Full Hessian eigenvalues (last 5):  {eigvals_full[-5:].round(4)}")

    # Verify analytic vs numerical
    # Count eigenvalues at each analytic value
    print(f"\n  Verification (analytic vs numerical):")
    for s in spectrum_1:
        lam = s['eigenvalue']
        count = np.sum(np.abs(eigvals_full - lam) < 0.01)
        expected = s['degeneracy']
        ok = "✓" if count == expected else "✗"
        print(f"    λ = {lam:8.4f}: analytic deg = {expected}, "
              f"numerical count = {count} {ok}")

    # -----------------------------------------------------------------------
    # Mass hierarchy analysis
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Mass Hierarchy Analysis")
    print("-" * 70)

    # Masses from non-zero eigenvalues
    masses = sorted(set(mass_from_eigenvalue(s['eigenvalue'])
                        for s in spectrum_1 if s['eigenvalue'] > 0.01))

    print(f"\n  Distinct non-zero masses: {len(masses)}")
    for i, m in enumerate(masses):
        print(f"    m_{i} = {m:.4f}")

    if len(masses) >= 2:
        print(f"\n  Mass ratios:")
        for i in range(len(masses)):
            for j in range(i + 1, len(masses)):
                print(f"    m_{j}/m_{i} = {masses[j]/masses[i]:.4f}")

    # -----------------------------------------------------------------------
    # Model 2: Vary weights to explore mass ratios
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Model 2: Scanning weight space for mass ratios")
    print("-" * 70)

    N_modes_2 = [5, 5, 5]  # equal sizes for simplicity
    print(f"\n  N_modes = {N_modes_2}")
    print(f"  {'w_01':>6} {'w_02':>6} {'w_12':>6} | {'m_1':>8} {'m_2':>8} | {'ratio':>6}")
    print(f"  " + "-" * 55)

    for w01 in [0.5, 1.0, 2.0]:
        for w02 in [0.3, 0.7, 1.5]:
            for w12 in [0.4, 1.0]:
                w = np.array([[0, w01, w02], [w01, 0, w12], [w02, w12, 0]])
                spec = analytic_spectrum(w, N_modes_2)
                ms = sorted(set(mass_from_eigenvalue(s['eigenvalue'])
                                for s in spec if s['eigenvalue'] > 0.01))
                if len(ms) >= 2:
                    ratio = ms[1] / ms[0]
                    print(f"  {w01:6.1f} {w02:6.1f} {w12:6.1f} | "
                          f"{ms[0]:8.4f} {ms[1]:8.4f} | {ratio:6.3f}")

    # -----------------------------------------------------------------------
    # Connection to observer stability
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  Connection to Observer Stability")
    print("-" * 70)
    print()
    print("  The observer_stability_minima.py test uses a DIFFERENT operator:")
    print("  a random matrix T with cyclic structure, NOT the spectral functional L.")
    print()
    print("  The 'effective physics' is defined via eigenvalues of projected T,")
    print("  not via the Hessian of L. These are two independent constructions.")
    print()
    print("  Connection status: NO direct connection. Both define excitation spectra")
    print("  but from different operators with different motivations.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  SUMMARY: EXCITATION SPECTRUM OF L")
    print("=" * 90)
    print()
    print("  1. DEFINITION: Excitations are perturbations δℓ^(a)_k around vacuum.")
    print("     Their dynamics (at quadratic level) are governed by the Hessian of L.")
    print()
    print("  2. SPECTRUM STRUCTURE:")
    print("     - 1 zero mode (global shift, always present)")
    print("     - (n_sectors - 1) inter-sector modes (distinct eigenvalues)")
    print("     - Σ_c (N_c - 1) internal modes (degenerate within each sector)")
    print()
    print("  3. MASS HIERARCHY: possible from inter-sector mode eigenvalues.")
    print("     Depends on relational weights w_{ab} and sector sizes N_a.")
    print("     Mass ratios can be tuned but are NOT predicted by the axioms.")
    print()
    print("  4. FUNDAMENTAL LIMITATION: The spectrum has massive degeneracies")
    print("     (N_c - 1 fold per sector) because L has no intra-sector structure.")
    print("     Real particles have non-degenerate spectra with specific quantum")
    print("     numbers — this requires structure beyond L.")
    print()
    print("  5. NO CONNECTION to observer stability test (different operator).")
    print()
    print("  CONCLUSION: L produces a minimal excitation spectrum with tunable")
    print("  mass hierarchy but massive degeneracies. It cannot reproduce the")
    print("  Standard Model spectrum without additional structure.")
    print()
    print("=" * 90)


if __name__ == '__main__':
    main()
