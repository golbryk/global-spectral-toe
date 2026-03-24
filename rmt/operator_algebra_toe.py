#!/usr/bin/env python3
"""
Operator Algebra TOE Explorer
==============================
Question: What is the MOST GENERAL algebraic system with reflection positivity (RP)?

Approach:
  1. Matrix algebras M_n(C) with completely positive (CP) maps as dynamics
  2. Test RP via spectral properties of the CP map's transfer matrix
  3. Ask: does RP + spectral gap CONSTRAIN the algebra?

Key insight from mass gap proof: RP is the single non-trivial input from which
all quantum mechanics follows (OS reconstruction). So we study what RP demands.

GPU-accelerated via CuPy.

Author: Grzegorz Olbryk
Date: 2026-03-24
"""

import numpy as np
import time
import sys

try:
    import cupy as cp
    GPU = True
    dev_name = cp.cuda.runtime.getDeviceProperties(0)['name']
    print(f"GPU: {dev_name}")
except ImportError:
    print("CuPy not available, using NumPy (CPU fallback)")
    import numpy as cp
    GPU = False

# ============================================================================
# PART 1: Completely Positive Maps on M_n(C)
# ============================================================================

def random_cp_map_kraus(n, num_kraus=None, rng=None):
    """
    Generate a random completely positive trace-preserving (CPTP) map on M_n(C)
    via Kraus operators: T(rho) = sum_k V_k rho V_k^dagger
    with sum_k V_k^dagger V_k = I (trace preservation).

    Returns Kraus operators and the superoperator matrix (n^2 x n^2).
    """
    if rng is None:
        rng = np.random
    if num_kraus is None:
        num_kraus = n  # typical: n Kraus operators

    # Generate random Kraus operators via partial isometry of a random unitary
    # on C^n tensor C^(num_kraus)
    dim = n * num_kraus
    # Random Ginibre matrix -> QR for Haar-ish unitary
    G = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(G)

    # Extract Kraus operators as n x n blocks from first n columns
    kraus = []
    for k in range(num_kraus):
        V_k = Q[k*n:(k+1)*n, :n]
        kraus.append(V_k)

    # Build superoperator: T_super |rho>> = |T(rho)>>
    # where |rho>> is the vectorization of rho (column-stacking)
    # T_super = sum_k conj(V_k) tensor V_k
    T_super = np.zeros((n*n, n*n), dtype=complex)
    for V_k in kraus:
        T_super += np.kron(np.conj(V_k), V_k)

    return kraus, T_super


def random_cp_map_gpu(n, num_kraus=None, rng=None):
    """Returns superoperator (on CPU — matrices are small, GPU overhead dominates)."""
    kraus, T_super = random_cp_map_kraus(n, num_kraus, rng)
    return T_super


def spectral_gap(T_super_gpu):
    """
    Compute spectral gap of a CPTP superoperator.

    For CPTP maps, the largest eigenvalue is always 1 (corresponding to the
    fixed point / steady state). The spectral gap is:
        gap = 1 - |lambda_2|
    where lambda_2 is the second-largest eigenvalue by modulus.

    RP connection: A positive spectral gap means exponential mixing,
    which in the OS/transfer-matrix framework corresponds to a mass gap.
    """
    evals = np.linalg.eigvals(T_super_gpu)
    mods = np.abs(evals)
    mods_sorted = np.sort(mods)[::-1]

    # Largest should be ~1 (CPTP)
    lambda_1 = float(mods_sorted[0])
    lambda_2 = float(mods_sorted[1])

    gap = lambda_1 - lambda_2
    return gap, lambda_1, lambda_2, evals


def check_rp(T_super_gpu, n):
    """
    Check reflection positivity of the transfer matrix T.

    RP condition: For the transfer matrix interpretation, RP requires
    that T is a POSITIVE operator (not just completely positive map).
    Specifically, the superoperator T must satisfy:
        <f|T|f> >= 0 for all f in the "reflected" subspace.

    In practice for matrix algebras, this means T_super (as a matrix)
    should have non-negative spectrum when restricted to self-adjoint elements.

    We check: T restricted to the real subspace of Hermitian matrices
    has all real non-negative eigenvalues.
    """
    # Build projector onto Hermitian subspace
    # Hermitian n x n matrices form a real n^2-dim subspace of C^{n x n}
    # Basis: E_ii (diagonal), (E_ij + E_ji)/sqrt(2), i(E_ij - E_ji)/sqrt(2)

    basis = []
    # Diagonal elements
    for i in range(n):
        e = np.zeros((n, n), dtype=complex)
        e[i, i] = 1.0
        basis.append(e.flatten())
    # Off-diagonal (real part)
    for i in range(n):
        for j in range(i+1, n):
            e = np.zeros((n, n), dtype=complex)
            e[i, j] = 1.0 / np.sqrt(2)
            e[j, i] = 1.0 / np.sqrt(2)
            basis.append(e.flatten())
    # Off-diagonal (imaginary part)
    for i in range(n):
        for j in range(i+1, n):
            e = np.zeros((n, n), dtype=complex)
            e[i, j] = 1j / np.sqrt(2)
            e[j, i] = -1j / np.sqrt(2)
            basis.append(e.flatten())

    P = np.array(basis)  # n^2 x n^2 (rows are basis vectors)
    # T restricted to Hermitian subspace
    T_herm = P @ T_super_gpu @ P.conj().T

    evals_herm = np.linalg.eigvals(T_herm)

    # RP: all eigenvalues should be real and non-negative
    max_imag = np.max(np.abs(evals_herm.imag))
    min_real = np.min(evals_herm.real)

    rp_holds = (max_imag < 1e-8) and (min_real > -1e-8)

    return rp_holds, min_real, max_imag, evals_herm


# ============================================================================
# PART 2: Systematic Scan
# ============================================================================

def scan_algebra(n, num_samples=10000, num_kraus=None, seed=42):
    """
    Scan random CPTP maps on M_n(C).
    Returns statistics on spectral gap and RP.
    """
    rng = np.random.RandomState(seed)

    gaps = []
    rp_count = 0
    rp_gaps = []
    non_rp_gaps = []
    eigenvalue_spectra = []

    for i in range(num_samples):
        T = random_cp_map_gpu(n, num_kraus=num_kraus, rng=rng)
        gap, lam1, lam2, evals = spectral_gap(T)
        gaps.append(gap)

        rp, min_real, max_imag, evals_herm = check_rp(T, n)

        if rp:
            rp_count += 1
            rp_gaps.append(gap)
        else:
            non_rp_gaps.append(gap)

        # Store first 50 spectra for analysis
        if i < 50:
            eigenvalue_spectra.append(evals)

    gaps = np.array(gaps)
    rp_gaps = np.array(rp_gaps) if rp_gaps else np.array([])
    non_rp_gaps = np.array(non_rp_gaps) if non_rp_gaps else np.array([])

    return {
        'n': n,
        'num_samples': num_samples,
        'gaps': gaps,
        'rp_fraction': rp_count / num_samples,
        'rp_count': rp_count,
        'rp_gaps': rp_gaps,
        'non_rp_gaps': non_rp_gaps,
        'spectra': eigenvalue_spectra
    }


# ============================================================================
# PART 3: Tensor Product Algebras (SM-like structures)
# ============================================================================

def tensor_cp_map(n1, n2, num_kraus=None, rng=None):
    """
    CPTP map on M_{n1}(C) tensor M_{n2}(C) = M_{n1*n2}(C)
    with PRODUCT structure: T = T1 tensor T2.

    This preserves the tensor factorization — like a gauge theory
    where SU(n1) and SU(n2) sectors evolve independently.
    """
    if rng is None:
        rng = np.random

    _, T1 = random_cp_map_kraus(n1, num_kraus=num_kraus, rng=rng)
    _, T2 = random_cp_map_kraus(n2, num_kraus=num_kraus, rng=rng)

    # Tensor product of superoperators
    T_tensor = np.kron(T1, T2)
    return T_tensor


def entangled_cp_map(n1, n2, num_kraus=None, rng=None):
    """
    CPTP map on M_{n1*n2}(C) that does NOT preserve tensor structure.
    This represents interaction / entanglement between sectors.
    """
    n = n1 * n2
    return random_cp_map_gpu(n, num_kraus=num_kraus, rng=rng)


# ============================================================================
# PART 4: Emergence Tests
# ============================================================================

def analyze_eigenvalue_structure(evals):
    """
    Analyze eigenvalue structure for gauge-theory-like patterns.

    Look for:
    1. Degeneracies (representation multiplicities)
    2. Casimir-like ordering (eigenvalues grouped by "angular momentum")
    3. Gap structure (hierarchical gaps between groups)
    """
    mods = np.abs(evals)
    mods_sorted = np.sort(mods)[::-1]

    # Find clusters of eigenvalues (representation-like grouping)
    threshold = 0.01  # eigenvalues within this are "degenerate"
    clusters = []
    current_cluster = [mods_sorted[0]]

    for i in range(1, len(mods_sorted)):
        if abs(mods_sorted[i] - current_cluster[-1]) < threshold:
            current_cluster.append(mods_sorted[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [mods_sorted[i]]
    clusters.append(current_cluster)

    # Degeneracy pattern
    degeneracies = [len(c) for c in clusters]

    # Gaps between clusters
    cluster_centers = [np.mean(c) for c in clusters]
    inter_gaps = [cluster_centers[i] - cluster_centers[i+1]
                  for i in range(len(cluster_centers)-1)]

    # Casimir test: are the cluster centers related by c_2 = j(j+1) pattern?
    # Normalize by first gap
    if len(inter_gaps) > 1 and inter_gaps[0] > 1e-10:
        gap_ratios = [g / inter_gaps[0] for g in inter_gaps]
    else:
        gap_ratios = []

    return {
        'num_clusters': len(clusters),
        'degeneracies': degeneracies,
        'cluster_centers': cluster_centers,
        'inter_gaps': inter_gaps,
        'gap_ratios': gap_ratios
    }


def symmetry_score(evals):
    """
    Measure how "symmetric" the eigenvalue distribution is.

    For a gauge group G acting on the algebra, eigenvalues come in
    representation multiplets. High symmetry = more degeneracies.

    Score = sum of (degeneracy - 1) / total_count
    """
    mods = np.sort(np.abs(evals))[::-1]
    threshold = 0.01
    deg_count = 0
    i = 0
    while i < len(mods):
        j = i + 1
        while j < len(mods) and abs(mods[j] - mods[i]) < threshold:
            deg_count += 1
            j += 1
        i = j
    return deg_count / len(mods)


def tensor_stability_test(n, num_trials=100, rng=None):
    """
    Test: Does A tensor A have the "same structure" as A?

    For each CPTP map T on M_n, form T tensor T on M_n tensor M_n = M_{n^2}.
    Compare gap structure.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    single_gaps = []
    tensor_gaps = []

    for _ in range(num_trials):
        _, T = random_cp_map_kraus(n, rng=rng)
        gap_single, _, _, _ = spectral_gap(T)

        # T tensor T
        T_tensor = np.kron(T, T)
        gap_tensor, _, _, _ = spectral_gap(T_tensor)

        single_gaps.append(gap_single)
        tensor_gaps.append(gap_tensor)

    return np.array(single_gaps), np.array(tensor_gaps)


# ============================================================================
# PART 5: The Deep Question — Does RP select SM?
# ============================================================================

def deep_scan():
    """
    For each algebra dimension n = 2,...,8:
    1. Generate many random CPTP maps
    2. Check RP fraction and gap distribution
    3. Look for special structure at n = 2*3 = 6 (SU(2) x SU(3))
    """
    results = {}

    for n in range(2, 9):
        print(f"\n{'='*60}")
        print(f"  M_{n}(C)  [dim = {n}^2 = {n*n}]")
        print(f"{'='*60}")

        # Adjust samples: larger n is slower (n^4 superoperator)
        if n <= 4:
            num_samples = 10000
        elif n <= 6:
            num_samples = 3000
        else:
            num_samples = 1000

        t0 = time.time()
        res = scan_algebra(n, num_samples=num_samples, seed=42+n)
        dt = time.time() - t0

        print(f"  Samples: {num_samples}  ({dt:.1f}s)")
        print(f"  RP fraction: {res['rp_fraction']:.4f} ({res['rp_count']}/{num_samples})")
        print(f"  Gap: mean={np.mean(res['gaps']):.4f}, "
              f"std={np.std(res['gaps']):.4f}, "
              f"min={np.min(res['gaps']):.6f}, "
              f"max={np.max(res['gaps']):.4f}")

        if len(res['rp_gaps']) > 0:
            print(f"  RP gap:    mean={np.mean(res['rp_gaps']):.4f}, "
                  f"std={np.std(res['rp_gaps']):.4f}")
        if len(res['non_rp_gaps']) > 0:
            print(f"  Non-RP gap: mean={np.mean(res['non_rp_gaps']):.4f}, "
                  f"std={np.std(res['non_rp_gaps']):.4f}")

        # Eigenvalue structure analysis on first 50 spectra
        sym_scores = []
        deg_patterns = []
        for spec in res['spectra']:
            ss = symmetry_score(spec)
            sym_scores.append(ss)
            analysis = analyze_eigenvalue_structure(spec)
            deg_patterns.append(analysis['degeneracies'])

        print(f"  Symmetry score: mean={np.mean(sym_scores):.4f}, "
              f"std={np.std(sym_scores):.4f}")

        # Most common degeneracy pattern
        from collections import Counter
        deg_strs = [str(d[:5]) for d in deg_patterns]  # first 5 clusters
        common = Counter(deg_strs).most_common(3)
        print(f"  Top degeneracy patterns (first 5 clusters):")
        for pattern, count in common:
            print(f"    {pattern}: {count}/{len(deg_patterns)}")

        results[n] = res

    return results


# ============================================================================
# PART 6: Product vs Entangled Structure
# ============================================================================

def product_vs_entangled_scan():
    """
    Compare PRODUCT maps (T1 x T2) vs ENTANGLED maps on M_{n1*n2}.

    Key question: Does RP prefer product structure?
    If so, this would explain why gauge groups factorize as SU(3) x SU(2) x U(1).
    """
    print(f"\n{'='*60}")
    print(f"  PRODUCT vs ENTANGLED CP Maps")
    print(f"{'='*60}")

    configs = [
        (2, 2, "M_2 x M_2  (SU(2) x SU(2))"),
        (2, 3, "M_2 x M_3  (electroweak-like)"),
        (3, 3, "M_3 x M_3  (color x color)"),
        (2, 4, "M_2 x M_4  (SU(2) x SU(4))"),
    ]

    rng = np.random.RandomState(123)
    num_samples = 2000

    for n1, n2, label in configs:
        n = n1 * n2
        print(f"\n  {label}  [total dim = {n}]")
        print(f"  {'-'*50}")

        # Product maps
        prod_gaps = []
        prod_rp = 0
        prod_rp_gaps = []
        for _ in range(num_samples):
            T = tensor_cp_map(n1, n2, rng=rng)
            gap, _, _, _ = spectral_gap(T)
            prod_gaps.append(gap)
            rp, _, _, _ = check_rp(T, n)
            if rp:
                prod_rp += 1
                prod_rp_gaps.append(gap)

        # Entangled maps
        ent_gaps = []
        ent_rp = 0
        ent_rp_gaps = []
        for _ in range(num_samples):
            T = entangled_cp_map(n1, n2, rng=rng)
            gap, _, _, _ = spectral_gap(T)
            ent_gaps.append(gap)
            rp, _, _, _ = check_rp(T, n)
            if rp:
                ent_rp += 1
                ent_rp_gaps.append(gap)

        print(f"  PRODUCT:   RP = {prod_rp/num_samples:.4f}, "
              f"gap = {np.mean(prod_gaps):.4f} +/- {np.std(prod_gaps):.4f}")
        print(f"  ENTANGLED: RP = {ent_rp/num_samples:.4f}, "
              f"gap = {np.mean(ent_gaps):.4f} +/- {np.std(ent_gaps):.4f}")

        if prod_rp_gaps:
            print(f"  PRODUCT RP gap:   {np.mean(prod_rp_gaps):.4f}")
        if ent_rp_gaps:
            print(f"  ENTANGLED RP gap: {np.mean(ent_rp_gaps):.4f}")


# ============================================================================
# PART 7: Tensor Stability
# ============================================================================

def tensor_stability_scan():
    """
    Test: Is there an algebra where the gap is STABLE under tensoring?
    gap(T x T) ~ gap(T)?

    Physical interpretation: the mass gap shouldn't depend on system size.
    """
    print(f"\n{'='*60}")
    print(f"  TENSOR STABILITY: gap(T tensor T) vs gap(T)")
    print(f"{'='*60}")

    for n in [2, 3, 4, 5]:
        if n >= 5:
            # n^2 = 25, tensor = 625 x 625 superoperator — still feasible
            num_trials = 50
        else:
            num_trials = 200

        single, tensor = tensor_stability_test(n, num_trials=num_trials)

        ratio = tensor / (single + 1e-15)
        print(f"\n  M_{n}(C): gap_tensor/gap_single = "
              f"{np.mean(ratio):.4f} +/- {np.std(ratio):.4f}")
        print(f"    gap_single: {np.mean(single):.4f}, gap_tensor: {np.mean(tensor):.4f}")

        # Correlation
        corr = np.corrcoef(single, tensor)[0, 1]
        print(f"    Correlation: {corr:.4f}")


# ============================================================================
# PART 8: Gap Distribution — Is There a Critical Dimension?
# ============================================================================

def gap_distribution_analysis(results):
    """
    Analyze: Is there a critical algebra dimension where the gap distribution changes?
    """
    print(f"\n{'='*60}")
    print(f"  GAP DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")

    print(f"\n  {'n':>3} | {'mean_gap':>10} | {'std_gap':>10} | {'min_gap':>10} | "
          f"{'gap>0.1':>8} | {'gap>0.5':>8} | {'RP_frac':>8}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    for n in sorted(results.keys()):
        r = results[n]
        g = r['gaps']
        frac_01 = np.mean(g > 0.1)
        frac_05 = np.mean(g > 0.5)
        print(f"  {n:3d} | {np.mean(g):10.4f} | {np.std(g):10.4f} | {np.min(g):10.6f} | "
              f"{frac_01:8.4f} | {frac_05:8.4f} | {r['rp_fraction']:8.4f}")

    # KEY QUESTION: Does RP_fraction have a non-monotonic dependence on n?
    ns = sorted(results.keys())
    rp_fracs = [results[n]['rp_fraction'] for n in ns]
    print(f"\n  RP fraction trend: {['%.4f' % f for f in rp_fracs]}")

    # Is there a PEAK?
    if len(rp_fracs) > 2:
        peak_n = ns[np.argmax(rp_fracs)]
        print(f"  Peak RP fraction at n = {peak_n}")


# ============================================================================
# PART 9: Casimir Emergence Test
# ============================================================================

def casimir_emergence_test(n, num_samples=500):
    """
    For CPTP maps with RP and gap, check if eigenvalue clusters
    follow a Casimir-like pattern: lambda_j ~ exp(-alpha * j(j+1))
    for some alpha.

    This would indicate that the algebra "knows about" angular momentum
    / representation theory even without being told about a gauge group.
    """
    print(f"\n{'='*60}")
    print(f"  CASIMIR EMERGENCE TEST: M_{n}(C)")
    print(f"{'='*60}")

    rng = np.random.RandomState(99 + n)

    casimir_scores = []

    for i in range(num_samples):
        T = random_cp_map_gpu(n, rng=rng)
        gap, _, _, evals = spectral_gap(T)
        rp, _, _, _ = check_rp(T, n)

        if not rp or gap < 0.05:
            continue

        # Get eigenvalue moduli, sorted descending
        mods = np.sort(np.abs(evals))[::-1]

        # Fit to Casimir pattern: log(|lambda_j|) ~ -alpha * j(j+1)
        # Use first min(6, len) eigenvalues
        k = min(6, len(mods))
        js = np.arange(k)
        cas = js * (js + 1)
        log_mods = np.log(mods[:k] + 1e-15)

        # Linear regression: log_mods = a + b * cas
        if np.std(cas) > 0:
            A = np.column_stack([np.ones(k), cas])
            result = np.linalg.lstsq(A, log_mods, rcond=None)
            coeffs = result[0]
            residuals = log_mods - A @ coeffs
            r_squared = 1 - np.var(residuals) / (np.var(log_mods) + 1e-15)
            casimir_scores.append(r_squared)

    if casimir_scores:
        scores = np.array(casimir_scores)
        print(f"  RP+gap samples: {len(scores)}")
        print(f"  Casimir fit R^2: mean={np.mean(scores):.4f}, "
              f"median={np.median(scores):.4f}, "
              f"fraction R^2>0.9: {np.mean(scores > 0.9):.4f}")
    else:
        print(f"  No RP+gap samples found!")


# ============================================================================
# PART 10: SM-specific Test
# ============================================================================

def sm_algebra_test():
    """
    The SM algebra is M_1(C) + M_2(C) + M_3(C) (direct sum).
    (This is the finite algebra in Connes' NCG approach.)

    Test: Does this algebra have SPECIAL RP/gap properties compared to
    M_6(C) (which has the same total dimension)?

    Direct sum: block-diagonal CPTP maps vs full M_6 maps.
    """
    print(f"\n{'='*60}")
    print(f"  SM ALGEBRA TEST: M_1+M_2+M_3 vs M_6")
    print(f"{'='*60}")

    rng = np.random.RandomState(777)
    num_samples = 3000
    n = 6  # total matrix size (1+2+3 = 6 if we embed as block-diagonal)

    # Actually: direct sum M_1 + M_2 + M_3 means block-diagonal
    # CP maps that preserve each block
    block_gaps = []
    block_rp = 0
    block_sym = []

    full_gaps = []
    full_rp = 0
    full_sym = []

    for _ in range(num_samples):
        # Block-diagonal: independent CP maps on each block
        _, T1 = random_cp_map_kraus(1, rng=rng)  # 1x1
        _, T2 = random_cp_map_kraus(2, rng=rng)  # 4x4
        _, T3 = random_cp_map_kraus(3, rng=rng)  # 9x9

        # Block diagonal superoperator (1 + 4 + 9 = 14... but M_6 has 36x36 super)
        # Actually we need to embed in M_6 properly
        # M_1 + M_2 + M_3 as block-diagonal in M_6:
        # rho = diag_block(rho_1, rho_2, rho_3)
        # The superoperator acts on vec(rho) which is 36-dim
        # But the block-diagonal subspace is 1+4+9 = 14 dim

        # Simpler: just compare the MINIMUM gap across blocks
        # M_1 is trivial (1x1), gap is meaningless — skip
        gap2, _, _, ev2 = spectral_gap(T2)
        gap3, _, _, ev3 = spectral_gap(T3)

        # The gap of the direct sum is the MINIMUM of individual gaps
        block_gap = min(gap2, gap3)
        block_gaps.append(block_gap)

        rp2, _, _, _ = check_rp(T2, 2)
        rp3, _, _, _ = check_rp(T3, 3)
        if rp2 and rp3:
            block_rp += 1

        # Symmetry score of combined spectrum
        combined = np.concatenate([ev2, ev3])
        block_sym.append(symmetry_score(combined))

        # Full M_6 map
        T_full = random_cp_map_gpu(n, rng=rng)
        gap_full, _, _, evals_full = spectral_gap(T_full)
        full_gaps.append(gap_full)

        rp_full, _, _, _ = check_rp(T_full, n)
        if rp_full:
            full_rp += 1
        full_sym.append(symmetry_score(evals_full))

    print(f"\n  M_1+M_2+M_3 (SM-like block-diagonal):")
    print(f"    RP fraction: {block_rp/num_samples:.4f}")
    print(f"    Gap: mean={np.mean(block_gaps):.4f}, std={np.std(block_gaps):.4f}")
    print(f"    Symmetry: mean={np.mean(block_sym):.4f}")

    print(f"\n  M_6 (full, no block structure):")
    print(f"    RP fraction: {full_rp/num_samples:.4f}")
    print(f"    Gap: mean={np.mean(full_gaps):.4f}, std={np.std(full_gaps):.4f}")
    print(f"    Symmetry: mean={np.mean(full_sym):.4f}")

    # KEY COMPARISON
    print(f"\n  RATIO (block/full):")
    print(f"    RP:  {block_rp/num_samples:.4f} / {full_rp/num_samples:.4f} "
          f"= {(block_rp+1)/(full_rp+1):.2f}x")
    print(f"    Gap: {np.mean(block_gaps):.4f} / {np.mean(full_gaps):.4f} "
          f"= {np.mean(block_gaps)/np.mean(full_gaps):.2f}x")


# ============================================================================
# PART 11: SELF-ADJOINT CP Maps (physically motivated RP)
# ============================================================================

def random_selfadjoint_cp_map(n, num_kraus=None, rng=None):
    """
    Generate a self-adjoint CPTP map: T = T^dagger as a superoperator.

    Physical motivation: In lattice gauge theory, the transfer matrix is
    self-adjoint (T = e^{-aH}). Generic CPTP maps are NOT self-adjoint.
    RP is a property of self-adjoint positive transfer matrices.

    Construction: T(rho) = sum_k V_k rho V_k^dagger where V_k = V_k^dagger.
    (Hermitian Kraus operators -> self-adjoint channel.)
    Then normalize to make trace-preserving.
    """
    if rng is None:
        rng = np.random
    if num_kraus is None:
        num_kraus = n

    # Random Hermitian Kraus operators
    kraus = []
    for _ in range(num_kraus):
        G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        H = (G + G.conj().T) / 2  # Hermitian
        kraus.append(H)

    # Build superoperator
    T_super = np.zeros((n*n, n*n), dtype=complex)
    for V_k in kraus:
        T_super += np.kron(np.conj(V_k), V_k)

    # For Hermitian V_k, conj(V_k) = V_k^T, so kron(V_k^T, V_k)
    # The superoperator IS self-adjoint (as a matrix on C^{n^2}).

    # Normalize: make trace-preserving by rescaling
    # sum V_k^2 should = I for TP. Instead, compute M = sum V_k^2
    # and rescale each V_k by M^{-1/2}
    M = sum(V_k @ V_k for V_k in kraus)
    # M is positive definite (generically)
    try:
        M_inv_sqrt = np.linalg.inv(np.linalg.cholesky(M)).T
    except np.linalg.LinAlgError:
        # M not positive definite, add small identity
        M += 0.01 * np.eye(n)
        M_inv_sqrt = np.linalg.inv(np.linalg.cholesky(M)).T

    kraus_tp = [M_inv_sqrt @ V_k for V_k in kraus]

    T_super_tp = np.zeros((n*n, n*n), dtype=complex)
    for V_k in kraus_tp:
        T_super_tp += np.kron(np.conj(V_k), V_k)

    return T_super_tp


def random_thermal_cp_map(n, beta=1.0, rng=None):
    """
    Generate a thermal / Gibbs-like CPTP map.

    T(rho) = e^{-beta H/2} rho e^{-beta H/2} / tr(e^{-beta H} rho)
    (unnormalized version, then project to TP)

    This is the closest analog to the lattice transfer matrix:
    T = e^{-aH} where H is a random Hamiltonian.
    """
    if rng is None:
        rng = np.random

    # Random Hamiltonian (Hermitian matrix from GUE)
    G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = (G + G.conj().T) / (2 * np.sqrt(2 * n))

    # e^{-beta H/2}
    exp_H = np.linalg.matrix_power(
        np.diag(np.exp(-beta * np.linalg.eigvalsh(H) / 2)), 1)
    # Actually compute properly via eigendecomposition
    evals, evecs = np.linalg.eigh(H)
    exp_half = evecs @ np.diag(np.exp(-beta * evals / 2)) @ evecs.conj().T

    # Superoperator: T|rho>> = |exp_half rho exp_half>> / Z
    # = (conj(exp_half) kron exp_half) |rho>>
    T_super = np.kron(np.conj(exp_half), exp_half)

    # Normalize to be trace-preserving
    # The trace of T(rho) = tr(exp_half^2 rho) = tr(e^{-beta H} rho)
    # We need to divide by Z = tr(e^{-beta H}) to get a proper state,
    # but for the superoperator to be TP we need a different normalization.
    # Actually: T_super as defined maps identity to e^{-beta H},
    # which has trace Z. So rescale: T_super /= Z^{1/n^2}... no.
    # Better: just return as-is and note the largest eigenvalue is Z.
    return T_super, H, evals


def selfadjoint_rp_scan():
    """
    Scan self-adjoint and thermal CP maps for RP.
    These are the physically motivated ensembles.
    """
    print(f"\n{'='*70}")
    print(f"  PHASE 2: SELF-ADJOINT & THERMAL CP MAPS")
    print(f"{'='*70}")

    for n in range(2, 9):
        num_samples = 5000 if n <= 5 else 2000
        rng = np.random.RandomState(200 + n)

        # --- Self-adjoint CP maps ---
        sa_rp_count = 0
        sa_gaps = []
        sa_rp_gaps = []

        for _ in range(num_samples):
            T = random_selfadjoint_cp_map(n, rng=rng)
            gap, lam1, lam2, evals = spectral_gap(T)
            sa_gaps.append(gap)

            rp, min_real, _, _ = check_rp(T, n)
            if rp:
                sa_rp_count += 1
                sa_rp_gaps.append(gap)

        # --- Thermal CP maps ---
        th_rp_count = 0
        th_gaps = []
        th_rp_gaps = []
        th_casimir_scores = []

        for _ in range(num_samples):
            T, H, H_evals = random_thermal_cp_map(n, beta=1.0, rng=rng)
            gap, lam1, lam2, evals = spectral_gap(T)
            th_gaps.append(gap)

            rp, min_real, _, evals_herm = check_rp(T, n)
            if rp:
                th_rp_count += 1
                th_rp_gaps.append(gap)

            # For thermal maps: check if eigenvalue ordering matches H eigenvalues
            # (Casimir-like structure means T knows about the Hamiltonian)
            T_evals_sorted = np.sort(np.abs(evals))[::-1]
            # Expected from e^{-beta H}: eigenvalues of T should be
            # e^{-beta (E_i + E_j)} for pairs (i,j)
            expected = np.sort(np.exp(-1.0 * (H_evals[:, None] + H_evals[None, :])).flatten())[::-1]
            if len(expected) == len(T_evals_sorted):
                corr = np.corrcoef(T_evals_sorted, expected[:len(T_evals_sorted)])[0, 1]
                if not np.isnan(corr):
                    th_casimir_scores.append(corr)

        print(f"\n  M_{n}(C)  [dim = {n}^2 = {n*n}]")
        print(f"  {'─'*55}")
        print(f"  Self-adjoint CP:  RP = {sa_rp_count/num_samples:.4f} "
              f"({sa_rp_count}/{num_samples}), "
              f"gap = {np.mean(sa_gaps):.4f} +/- {np.std(sa_gaps):.4f}")
        if sa_rp_gaps:
            print(f"    RP gap: {np.mean(sa_rp_gaps):.4f} +/- {np.std(sa_rp_gaps):.4f}")

        print(f"  Thermal CP:       RP = {th_rp_count/num_samples:.4f} "
              f"({th_rp_count}/{num_samples}), "
              f"gap = {np.mean(th_gaps):.4f} +/- {np.std(th_gaps):.4f}")
        if th_rp_gaps:
            print(f"    RP gap: {np.mean(th_rp_gaps):.4f} +/- {np.std(th_rp_gaps):.4f}")
        if th_casimir_scores:
            print(f"    Casimir correlation: {np.mean(th_casimir_scores):.4f}")


# ============================================================================
# PART 12: WEAKER RP — Positivity of the TRACE functional
# ============================================================================

def weak_rp_test(n, num_samples=5000, rng=None):
    """
    Weaker RP: instead of requiring ALL eigenvalues of T|_Herm to be >= 0,
    require only that tr(T^k) >= 0 for all k (positivity of moments).

    This is closer to how RP is actually used in lattice QFT:
    <f, T f> >= 0 for observables f, not for arbitrary vectors.

    Even weaker: just check that T has a UNIQUE maximal eigenvalue
    (Perron-Frobenius-like condition), which is sufficient for
    OS reconstruction.
    """
    if rng is None:
        rng = np.random.RandomState(300 + n)

    strong_rp = 0
    weak_rp_moments = 0
    pf_condition = 0  # Perron-Frobenius: unique dominant eigenvalue

    for _ in range(num_samples):
        _, T = random_cp_map_kraus(n, rng=rng)
        evals = np.linalg.eigvals(T)
        mods = np.abs(evals)
        idx = np.argsort(mods)[::-1]
        evals_sorted = evals[idx]

        # Strong RP
        rp, _, _, _ = check_rp(T, n)
        if rp:
            strong_rp += 1

        # Weak RP: tr(T^k) >= 0 for k = 1,...,10
        moments_ok = True
        for k in range(1, 11):
            tr_Tk = np.real(np.sum(evals**k))
            if tr_Tk < -1e-8:
                moments_ok = False
                break
        if moments_ok:
            weak_rp_moments += 1

        # Perron-Frobenius: dominant eigenvalue is real, positive, unique
        if (abs(evals_sorted[0].imag) < 1e-8 and
            evals_sorted[0].real > 0 and
            mods[idx[0]] - mods[idx[1]] > 1e-8):
            pf_condition += 1

    return {
        'n': n,
        'strong_rp': strong_rp / num_samples,
        'weak_rp': weak_rp_moments / num_samples,
        'perron_frobenius': pf_condition / num_samples
    }


def weak_rp_scan():
    """Compare strong RP, weak RP (moments), and Perron-Frobenius across n."""
    print(f"\n{'='*70}")
    print(f"  WEAK RP HIERARCHY")
    print(f"{'='*70}")

    print(f"\n  {'n':>3} | {'Strong RP':>10} | {'Weak RP':>10} | {'Perron-Frob':>12}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

    for n in range(2, 9):
        ns = 5000 if n <= 5 else 2000
        res = weak_rp_test(n, num_samples=ns)
        print(f"  {n:3d} | {res['strong_rp']:10.4f} | {res['weak_rp']:10.4f} | "
              f"{res['perron_frobenius']:12.4f}")


# ============================================================================
# PART 13: GAUGE GROUP FROM COMMUTANT
# ============================================================================

def commutant_analysis(n, num_samples=500):
    """
    For CPTP maps with gap, compute the commutant of T:
    C(T) = {A in M_n : T(A) = lambda_1 * A}
    i.e., the fixed-point algebra (eigenspace of dominant eigenvalue).

    The gauge group emerges as the automorphism group of this fixed-point algebra.
    If C(T) = M_k(C) for some k, the gauge group is U(k).
    """
    print(f"\n  Commutant analysis for M_{n}(C):")
    rng = np.random.RandomState(400 + n)

    fp_dims = []

    for _ in range(num_samples):
        _, T = random_cp_map_kraus(n, rng=rng)
        evals = np.linalg.eigvals(T)
        mods = np.abs(evals)

        # Find eigenvalues close to the maximum
        max_mod = np.max(mods)
        threshold = 0.99 * max_mod

        # Count dimension of "near-fixed-point" space
        fp_dim = np.sum(mods > threshold)
        fp_dims.append(fp_dim)

    fp_dims = np.array(fp_dims)
    from collections import Counter
    counts = Counter(fp_dims)

    print(f"    Fixed-point dimension distribution:")
    for dim, count in sorted(counts.items()):
        pct = count / num_samples * 100
        if pct > 0.5:
            print(f"      dim={dim}: {pct:.1f}%")
        # Check if dim is a perfect square (-> M_k subalgebra)
        k = int(np.sqrt(dim))
        if k*k == dim and dim > 1:
            print(f"        -> M_{k}(C) subalgebra! Gauge group U({k})")


def gauge_emergence_scan():
    """Scan commutant structure across algebra dimensions."""
    print(f"\n{'='*70}")
    print(f"  GAUGE GROUP FROM COMMUTANT (fixed-point algebra)")
    print(f"{'='*70}")

    for n in range(2, 8):
        commutant_analysis(n, num_samples=2000)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("  OPERATOR ALGEBRA TOE EXPLORER")
    print("  Question: Does RP + spectral gap select the SM algebra?")
    print("="*70)

    t_start = time.time()

    # --- Part A: Deep scan across algebra dimensions ---
    results = deep_scan()

    # --- Part B: Gap distribution analysis ---
    gap_distribution_analysis(results)

    # --- Part C: Product vs Entangled ---
    product_vs_entangled_scan()

    # --- Part D: Tensor stability ---
    tensor_stability_scan()

    # --- Part E: Casimir emergence ---
    for n in [2, 3, 4, 5]:
        casimir_emergence_test(n, num_samples=500)

    # --- Part F: SM algebra test ---
    sm_algebra_test()

    # --- Part G: Self-adjoint & Thermal CP maps (Phase 2) ---
    selfadjoint_rp_scan()

    # --- Part H: Weak RP hierarchy ---
    weak_rp_scan()

    # --- Part I: Gauge emergence from commutant ---
    gauge_emergence_scan()

    t_total = time.time() - t_start

    print(f"\n{'='*70}")
    print(f"  TOTAL TIME: {t_total:.1f}s")
    print(f"{'='*70}")

    # --- SUMMARY ---
    print(f"\n{'='*70}")
    print(f"  SUMMARY OF FINDINGS")
    print(f"{'='*70}")

    print(f"""
  1. RP FRACTION vs ALGEBRA DIMENSION:
     Does RP become rarer for larger algebras? (Would suggest RP is selective.)
     Answer: See table above.

  2. SPECTRAL GAP DISTRIBUTION:
     Is the gap distribution universal or n-dependent?
     Answer: See gap statistics above.

  3. PRODUCT vs ENTANGLED:
     Does RP prefer factorized dynamics? (Would explain SM gauge group.)
     Answer: See product vs entangled comparison above.

  4. TENSOR STABILITY:
     Is there a special n where gap(T x T) ~ gap(T)?
     Answer: See tensor stability scan above.

  5. CASIMIR EMERGENCE:
     Do RP maps spontaneously exhibit Casimir-like eigenvalue patterns?
     Answer: See Casimir test above.

  6. SM ALGEBRA:
     Does M_1+M_2+M_3 have special RP/gap properties vs M_6?
     Answer: See SM test above.

  7. SELF-ADJOINT & THERMAL CP MAPS:
     Do physically motivated maps (self-adjoint, thermal) have much higher RP rates?
     Answer: See Phase 2 results above.

  8. WEAK RP HIERARCHY:
     Strong RP vs Weak RP (positive moments) vs Perron-Frobenius.
     Which level is generic? Which is selective?
     Answer: See weak RP table above.

  9. GAUGE EMERGENCE FROM COMMUTANT:
     Does the fixed-point algebra of T reveal a gauge group U(k)?
     Answer: See commutant analysis above.
""")


if __name__ == '__main__':
    main()
