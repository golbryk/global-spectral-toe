"""
Emergent Gravity Analysis for the Spectral TOE
================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 8 — Derive emergent gravity from spectral functional

The spectral functional is:
    L = Σ_{a,b} w_{ab} Σ_{m,n} (ℓ^(a)_m - ℓ^(b)_n)²

This script:
1. Computes the Euler-Lagrange equations (vacuum conditions)
2. Computes the Hessian (second variation) of L around vacuum
3. Tests whether the Hessian has structure that could generate a Poisson equation
4. Documents the NEGATIVE result: L is diagonal in mode index → no interactions

The mathematical derivation is:

    ∂L/∂ℓ^(c)_k = 4 Σ_b w_{cb} N_b [ℓ^(c)_k - c_b]

where c_b = (1/N_b) Σ_n ℓ^(b)_n is the mean of sector b (assuming w symmetric).

Vacuum condition: ℓ^(c)_k = [Σ_b w_{cb} N_b c_b] / [Σ_b w_{cb} N_b]
    → all modes in sector c have the SAME value at vacuum.

Second variation (Hessian) around vacuum:
    ∂²L / (∂ℓ^(c)_k ∂ℓ^(d)_j)
        = 4 Σ_b w_{cb} N_b · δ_{cd} δ_{kj}     [diagonal in mode index]
        - 4 w_{cd}                                [cross-sector coupling]

The Hessian is BLOCK-DIAGONAL: modes within a sector are decoupled.
There is no "kinetic" term (no gradient structure), so no Laplacian emerges.
"""

import numpy as np
from itertools import product


def build_spectral_functional(ell, w):
    """Compute L = Σ_{a,b} w_{ab} Σ_{m,n} (ℓ^(a)_m - ℓ^(b)_n)².

    Parameters
    ----------
    ell : list of arrays, ell[a] = array of ℓ^(a)_n values
    w   : 2D array, w[a][b] = relational weight

    Returns
    -------
    L : float
    """
    n_sectors = len(ell)
    L = 0.0
    for a in range(n_sectors):
        for b in range(n_sectors):
            for m in range(len(ell[a])):
                for n in range(len(ell[b])):
                    L += w[a][b] * (ell[a][m] - ell[b][n]) ** 2
    return L


def euler_lagrange(ell, w):
    """Compute ∂L/∂ℓ^(c)_k for all (c, k).

    Returns list of arrays, grad[c][k] = ∂L/∂ℓ^(c)_k.
    """
    n_sectors = len(ell)
    grad = [np.zeros_like(ell[c]) for c in range(n_sectors)]

    for c in range(n_sectors):
        for k in range(len(ell[c])):
            val = 0.0
            for b in range(n_sectors):
                for n in range(len(ell[b])):
                    val += 2 * w[c][b] * (ell[c][k] - ell[b][n])
                    val += 2 * w[b][c] * (ell[c][k] - ell[b][n])
                    # Note: we added both (c,b) and (b,c) contributions
                    # For symmetric w, this gives factor 4
            # Correct: avoid double-counting
            # ∂L/∂ℓ^(c)_k = 2 Σ_b w_{cb} Σ_n (ℓ^(c)_k - ℓ^(b)_n)
            #              + 2 Σ_a w_{ac} Σ_m (-(ℓ^(a)_m - ℓ^(c)_k))
            # For symmetric w: = 4 Σ_b w_{cb} Σ_n (ℓ^(c)_k - ℓ^(b)_n)
            pass

    # Cleaner implementation:
    grad = [np.zeros_like(ell[c]) for c in range(n_sectors)]
    for c in range(n_sectors):
        for k in range(len(ell[c])):
            val = 0.0
            for b in range(n_sectors):
                N_b = len(ell[b])
                c_b = np.mean(ell[b])
                # For symmetric w:
                val += 4 * w[c][b] * N_b * (ell[c][k] - c_b)
            grad[c][k] = val

    return grad


def vacuum_solution(w, N_modes):
    """Find vacuum: c_a* = [Σ_b w_{ab} N_b c_b*] / [Σ_b w_{ab} N_b].

    The vacuum equation is M c = 0 where M = D - WN (generally non-symmetric).
    The trivial solution c = α·(1,...,1) is always in the null space.

    For generic w and N, this is the ONLY solution (rank(M) = n_sectors - 1).
    Non-trivial vacua (different c_a*) only exist if M has rank < n_sectors - 1.
    """
    n_sectors = len(N_modes)
    N = np.array(N_modes, dtype=float)

    WN = np.array(w) * N[np.newaxis, :]
    D = np.diag(WN.sum(axis=1))
    M = D - WN

    # Verify: M · 1 = 0 always
    ones = np.ones(n_sectors)
    residual = M @ ones
    assert np.max(np.abs(residual)) < 1e-10, "M·1 ≠ 0!"

    # Check rank: for generic w, rank = n_sectors - 1
    rank = np.linalg.matrix_rank(M, tol=1e-10)

    # The trivial vacuum: c = (1, 1, ..., 1) (all sectors equal)
    c_star = ones.copy()

    # SVD to find null space dimension
    U, S, Vt = np.linalg.svd(M)
    null_dim = np.sum(S < 1e-10)

    return c_star, S, rank, null_dim


def build_hessian(w, N_modes):
    """Build the full Hessian ∂²L/(∂ℓ^(c)_k ∂ℓ^(d)_j) at vacuum.

    The total number of variables is Σ_a N_a.
    Variable index: (c, k) → flat_index = Σ_{a<c} N_a + k.

    For symmetric w:
    ∂²L/(∂ℓ^(c)_k ∂ℓ^(d)_j) =
        4 δ_{cd} δ_{kj} Σ_b w_{cb} N_b    [diagonal: self-coupling]
        - 4 w_{cd}                           [off-diagonal: cross-sector coupling]
    """
    n_sectors = len(N_modes)
    N_total = sum(N_modes)
    H = np.zeros((N_total, N_total))

    # Build index map
    offsets = [0]
    for n in N_modes:
        offsets.append(offsets[-1] + n)

    for c in range(n_sectors):
        for d in range(n_sectors):
            for k in range(N_modes[c]):
                for j in range(N_modes[d]):
                    i_ck = offsets[c] + k
                    i_dj = offsets[d] + j
                    if c == d and k == j:
                        # Diagonal: 4 Σ_b w_{cb} N_b
                        val = 4 * sum(w[c][b] * N_modes[b] for b in range(n_sectors))
                        H[i_ck, i_dj] = val
                    # Cross-sector coupling (all pairs)
                    H[i_ck, i_dj] -= 4 * w[c][d]

    return H


def analyze_hessian_structure(H, N_modes):
    """Analyze the structure of the Hessian.

    Key question: does it have a Laplacian-like structure?
    A Laplacian would have off-diagonal couplings between NEIGHBORING modes
    within a sector. The spectral functional's Hessian only has
    UNIFORM cross-sector coupling.
    """
    N_total = H.shape[0]
    eigvals = np.sort(np.linalg.eigvalsh(H))

    # Check block structure
    offsets = [0]
    for n in N_modes:
        offsets.append(offsets[-1] + n)

    n_sectors = len(N_modes)

    print(f"  Hessian size: {N_total} × {N_total}")
    print(f"  Eigenvalues: min={eigvals[0]:.6f}, max={eigvals[-1]:.6f}")
    print(f"  Null space dim: {np.sum(np.abs(eigvals) < 1e-10)}")
    print()

    # Check: within-sector blocks
    print("  Within-sector block structure:")
    for c in range(n_sectors):
        block = H[offsets[c]:offsets[c + 1], offsets[c]:offsets[c + 1]]
        diag_vals = np.unique(np.diag(block).round(8))
        off_diag = block - np.diag(np.diag(block))
        off_diag_vals = np.unique(off_diag[off_diag != 0].round(8))
        print(f"    Sector {c} ({N_modes[c]} modes):")
        print(f"      Diagonal: {diag_vals}")
        print(f"      Off-diagonal: {off_diag_vals if len(off_diag_vals) > 0 else '[all zero]'}")

    print()
    print("  Cross-sector block structure:")
    for c in range(n_sectors):
        for d in range(c + 1, n_sectors):
            block = H[offsets[c]:offsets[c + 1], offsets[d]:offsets[d + 1]]
            vals = np.unique(block.round(8))
            print(f"    Sectors {c}↔{d}: unique values = {vals}")


def main():
    print()
    print("=" * 90)
    print("  Emergent Gravity Analysis for the Spectral TOE")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Task 8")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # Test case: 3 sectors with different numbers of modes
    # -----------------------------------------------------------------------
    n_sectors = 3
    N_modes = [4, 5, 3]  # modes per sector
    w = np.array([
        [0.0, 1.0, 0.5],
        [1.0, 0.0, 0.7],
        [0.5, 0.7, 0.0],
    ])

    print(f"\n  Configuration: {n_sectors} sectors, N_modes = {N_modes}")
    print(f"  Relational weights w:")
    for row in w:
        print(f"    {row}")

    # -----------------------------------------------------------------------
    # 8a: Vacuum solution
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  8a. Vacuum Configuration")
    print("-" * 70)

    c_star, sv, rank, null_dim = vacuum_solution(w, N_modes)
    print(f"  Vacuum fixed points c_a*: {c_star}")
    print(f"  Vacuum matrix singular values: {sv.round(6)}")
    print(f"  Rank of M: {rank} (n_sectors - 1 = {n_sectors - 1})")
    print(f"  Null space dimension: {null_dim}")

    if rank == n_sectors - 1:
        print("  → ONLY the trivial vacuum exists (all c_a equal).")
    else:
        print(f"  → Non-trivial vacuum MAY exist (null space dim = {null_dim}).")

    # Verify: trivial vacuum c = (1,1,1) is a stationary point
    ell_vac = [np.full(N_modes[a], c_star[a]) for a in range(n_sectors)]
    grad_vac = euler_lagrange(ell_vac, w)
    max_grad = max(np.max(np.abs(g)) for g in grad_vac)
    print(f"  |∇L| at trivial vacuum: {max_grad:.2e}")

    if max_grad < 1e-8:
        print("  → Confirmed: trivial vacuum is a stationary point.")
    else:
        print("  → ERROR in computation.")

    # L at trivial vacuum
    L_vac = build_spectral_functional(ell_vac, w)
    print(f"  L at trivial vacuum: {L_vac:.6f}")
    print(f"  → L = 0 at trivial vacuum (global minimum).")

    # -----------------------------------------------------------------------
    # 8b: Hessian structure
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  8b. Hessian (Second Variation) at Vacuum")
    print("-" * 70)

    H = build_hessian(w, N_modes)
    analyze_hessian_structure(H, N_modes)

    # -----------------------------------------------------------------------
    # Key structural test: is the Hessian a Laplacian?
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  KEY STRUCTURAL TEST: Does the Hessian have Laplacian structure?")
    print("-" * 70)

    # A Laplacian on a 1D chain of N modes would have:
    #   H_{kk} = 2, H_{k,k±1} = -1, H_{kj} = 0 otherwise
    # This creates nearest-neighbor coupling → gradient terms → waves → 1/r potential

    # The spectral functional's Hessian has:
    #   Within sector c: H_{ck,ck} = 4 Σ_b w_{cb} N_b (same for all k)
    #                    H_{ck,cj} = -4 w_{cc} (uniform, NOT nearest-neighbor)
    #   Cross sectors:   H_{ck,dj} = -4 w_{cd} (uniform)

    print()
    print("  Within-sector coupling pattern:")
    print("  - Diagonal: 4 Σ_b w_{cb} N_b  (constant across all modes k)")
    print("  - Off-diag: -4 w_{cc}          (UNIFORM, not nearest-neighbor)")
    print()
    print("  This is a COMPLETE GRAPH coupling, not a CHAIN/LATTICE coupling.")
    print("  Every mode couples equally to every other mode in the same sector.")
    print()
    print("  For a Poisson equation ∇²φ = ρ to emerge, we need:")
    print("    - A notion of DISTANCE between modes (spatial structure)")
    print("    - NEAREST-NEIGHBOR coupling (gradient → Laplacian)")
    print("    - Coupling that DECAYS with distance")
    print()
    print("  The spectral functional has NONE of these.")
    print("  Its Hessian is uniform (all-to-all) → no spatial structure.")

    # -----------------------------------------------------------------------
    # 8c: Can the Kepler potential V(r) = a/r emerge?
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  8c. Can Kepler Potential Emerge?")
    print("-" * 70)

    print()
    print("  The v1 code (qcf_core.py) obtains V(r) ≈ a/r by:")
    print("    1. Placing a mass density ρ(x) on a 3D grid")
    print("    2. Solving ∇²G = ρ via FFT (solve_poisson_3d)")
    print("    3. Fitting G(r) = a/r + b")
    print()
    print("  Steps 1-2 are PUT IN BY HAND — the Poisson equation")
    print("  is not derived from the spectral axioms A1-A4.")
    print()
    print("  The spectral functional L has no spatial coordinates.")
    print("  The modes ℓ^(a)_n are abstract numbers, not fields on a manifold.")
    print("  There is no Laplacian in the spectral functional.")
    print()
    print("  CONCLUSION: V(r) = a/r in the v1 code is an INPUT, not an OUTPUT.")
    print("  The spectral axioms cannot produce it without additional structure.")

    # -----------------------------------------------------------------------
    # What WOULD be needed
    # -----------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  WHAT WOULD BE NEEDED FOR EMERGENT GRAVITY")
    print("-" * 70)

    print()
    print("  To derive a Poisson-like equation from a spectral functional,")
    print("  one would need to modify L to include:")
    print()
    print("  Option 1: Spatial embedding")
    print("    L = Σ_{a,b} w_{ab}(x,y) Σ_{m,n} (ℓ^(a)_m(x) - ℓ^(b)_n(y))²")
    print("    where w_{ab}(x,y) decays with |x-y|.")
    print("    → This introduces space BY HAND (defeats the purpose).")
    print()
    print("  Option 2: Spectral gradient terms")
    print("    L = Σ_a Σ_n (ℓ^(a)_{n+1} - ℓ^(a)_n)² + [inter-sector terms]")
    print("    → Mode index n becomes a 'position' coordinate.")
    print("    → This gives a 1D Laplacian on spectral space, not physical space.")
    print()
    print("  Option 3: Emergent geometry from spectral data")
    print("    Define distance d(m,n) = f(ℓ^(a)_m - ℓ^(a)_n)")
    print("    → Requires additional axioms beyond A1-A4.")
    print("    → Connes' noncommutative geometry uses d(x,y) = sup{|f(x)-f(y)| : ||[D,f]|| ≤ 1}")
    print("      but this requires a Dirac operator D, which is not part of the spectral TOE.")
    print()
    print("  NONE of these options are available within axioms A1-A4 as stated.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  SUMMARY: TASK 8 — NEGATIVE RESULT")
    print("=" * 90)
    print()
    print("  8a. The Euler-Lagrange equations of L yield vacuum conditions")
    print("      ℓ^(c)_k = const for all k in sector c (uniform spectrum).")
    print("      NO Poisson equation emerges.")
    print()
    print("  8b. The Hessian of L at vacuum is UNIFORM (all-to-all coupling)")
    print("      within and across sectors. There is no nearest-neighbor")
    print("      structure, no gradient term, no Laplacian.")
    print("      Perturbations around vacuum do NOT interact through L.")
    print()
    print("  8c. The Kepler potential V(r) = a/r in the v1 code comes from")
    print("      solve_poisson_3d, which is an EXTERNAL INPUT, not derived")
    print("      from the spectral functional.")
    print()
    print("  FUNDAMENTAL GAP: The spectral functional L operates on abstract")
    print("  spectra with no spatial structure. To derive gravity, one needs")
    print("  emergent spatial geometry — this requires additional axioms or")
    print("  mechanisms beyond A1-A4.")
    print()
    print("=" * 90)


if __name__ == '__main__':
    main()
