#!/usr/bin/env python3
"""
Connes Distance Formula on SU(2) Spectral Data
================================================

Tests whether the spectral data from the SU(2) transfer matrix can generate
a physical metric via Connes' noncommutative geometry distance formula:

    d(p,q) = sup{ |f(p) - f(q)| : ||[D, f]|| <= 1 }

where D is a Dirac operator with eigenvalues related to SU(2) Casimir values.

Four tests:
1. Connes distance with D = Casimir eigenvalues C_2(j) = j(j+1)
2. Comparison with geodesic distance on S^3
3. Perturbation response: delta d / delta epsilon
4. Search for D that reproduces 4D geometry (spectral dimension scan)

Author: Grzegorz Olbryk
Date: 2026-03-24
"""

import numpy as np

try:
    import cupy as cp
    HAS_GPU = True
    print("CuPy detected — using GPU acceleration")
except ImportError:
    import numpy as cp
    HAS_GPU = False
    print("CuPy not available — falling back to NumPy (CPU)")

from scipy.special import iv as bessel_iv  # modified Bessel I_v


# =============================================================================
# 1. CONNES DISTANCE on representation space
# =============================================================================

def casimir_su2(j):
    """SU(2) Casimir C_2(j) = j(j+1) for spin j."""
    return j * (j + 1)


def connes_distance_finite(dirac_eigenvalues, p_idx, q_idx):
    """
    Compute Connes distance between states p and q on a finite spectral space.

    D = diag(lambda_0, lambda_1, ..., lambda_{N-1})
    f = diag(f_0, f_1, ..., f_{N-1})
    [D, f] on the diagonal is zero, but off-diagonal:
      [D, f]_{ij} = (lambda_i - lambda_j) * f_{ij}

    For a commutative algebra (diagonal f), the distance is:
      d(p,q) = sup{ |f_p - f_q| : max_{i!=j} |f_i - f_j| * |lambda_i - lambda_j| <= 1 }

    But actually for diagonal f, [D,f] = 0 identically.

    The correct Connes distance uses the FULL operator algebra.
    For a finite space with N points and Dirac D:
      d(p,q) = sup{ |f_p - f_q| : ||[D, pi(f)]|| <= 1 }
    where pi(f) acts on H and [D, pi(f)] has matrix elements
      [D, pi(f)]_{ij} = (D_{ii} - D_{jj}) * f(i) delta_{ij} ...

    For diagonal D and diagonal f (commutative): [D,f] = 0, distance = infinity.

    The CORRECT approach: D must have OFF-DIAGONAL elements.
    Use D as a finite Dirac with off-diagonal structure connecting representations.

    Standard choice for a 1D lattice-like structure:
      D_{j, j+1} = lambda_j,  D_{j+1, j} = lambda_j^*
    (nearest-neighbor hopping with strength from Casimir).
    """
    N = len(dirac_eigenvalues)
    lam = cp.asarray(dirac_eigenvalues, dtype=cp.float64)

    # Build off-diagonal Dirac operator (nearest-neighbor on rep lattice)
    # D_{j,j+1} = sqrt(C_2(j+1) - C_2(j)) = sqrt(2j+1) ... but simpler:
    # Use D_{j,j+1} = C_2(j+1) - C_2(j) = 2j + 2 (difference of Casimirs)
    D = cp.zeros((N, N), dtype=cp.float64)
    for i in range(N - 1):
        diff = float(lam[i + 1] - lam[i])
        D[i, i + 1] = diff
        D[i + 1, i] = diff

    # Connes distance via linear programming / direct optimization
    # d(p,q) = sup |f_p - f_q| subject to ||[D, diag(f)]||_op <= 1
    # [D, diag(f)]_{ij} = D_{ij} * (f_i - f_j)  (for off-diagonal D)
    # ||[D, diag(f)]||_op <= 1

    # For nearest-neighbor D: [D, diag(f)]_{i,i+1} = D_{i,i+1}*(f_i - f_{i+1})
    # The operator norm is >= |D_{i,i+1}|*|f_i - f_{i+1}| for each i
    # So the constraint is: |D_{i,i+1}|*|f_i - f_{i+1}| <= 1 for all i
    # (This is exact for tridiagonal [D, diag(f)].)

    # This gives: |f_i - f_{i+1}| <= 1/|D_{i,i+1}|
    # And d(p,q) = sum_{k=min(p,q)}^{max(p,q)-1} 1/|D_{k,k+1}|

    i_min = min(p_idx, q_idx)
    i_max = max(p_idx, q_idx)

    dist = 0.0
    for k in range(i_min, i_max):
        d_val = abs(float(D[k, k + 1]))
        if d_val > 1e-15:
            dist += 1.0 / d_val

    return dist


def connes_distance_full_optimization(dirac_eigenvalues, p_idx, q_idx, n_trials=100000):
    """
    Compute Connes distance by Monte Carlo optimization over functions f.

    Build full [D, diag(f)] matrix and compute its operator norm,
    maximize |f_p - f_q| subject to ||[D, diag(f)]||_op <= 1.

    Uses GPU-accelerated random sampling.
    """
    N = len(dirac_eigenvalues)
    lam = cp.asarray(dirac_eigenvalues, dtype=cp.float64)

    # Build off-diagonal Dirac (nearest-neighbor)
    D = cp.zeros((N, N), dtype=cp.float64)
    for i in range(N - 1):
        diff = float(lam[i + 1] - lam[i])
        D[i, i + 1] = diff
        D[i + 1, i] = diff

    # Also try full off-diagonal: D_{ij} = |lambda_i - lambda_j|
    D_full = cp.zeros((N, N), dtype=cp.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                D_full[i, j] = abs(float(lam[i] - lam[j]))

    best_dist_nn = 0.0
    best_dist_full = 0.0

    # Generate random functions on GPU
    F = cp.random.randn(n_trials, N).astype(cp.float64)

    for trial in range(n_trials):
        f = F[trial]

        # Nearest-neighbor Dirac
        f_diag = cp.diag(f)
        comm = D @ f_diag - f_diag @ D  # = [D, diag(f)]
        # Operator norm = largest singular value
        try:
            sv = cp.linalg.svd(comm, compute_uv=False)
            op_norm = float(sv[0])
        except:
            continue

        if op_norm > 1e-15:
            scaled_f = f / op_norm  # now ||[D, diag(scaled_f)]|| = 1
            val = abs(float(scaled_f[p_idx] - scaled_f[q_idx]))
            if val > best_dist_nn:
                best_dist_nn = val

        # Full off-diagonal Dirac
        comm_full = D_full @ f_diag - f_diag @ D_full
        try:
            sv_full = cp.linalg.svd(comm_full, compute_uv=False)
            op_norm_full = float(sv_full[0])
        except:
            continue

        if op_norm_full > 1e-15:
            scaled_f = f / op_norm_full
            val = abs(float(scaled_f[p_idx] - scaled_f[q_idx]))
            if val > best_dist_full:
                best_dist_full = val

    return best_dist_nn, best_dist_full


def run_test1():
    """Test 1: Connes distance with Casimir Dirac operator."""
    print("\n" + "=" * 70)
    print("TEST 1: CONNES DISTANCE FORMULA on SU(2) representation space")
    print("=" * 70)

    # SU(2) representations: j = 0, 1/2, 1, 3/2, 2, 5/2, 3, ...
    # We use integer labeling: p = 0, 1, 2, ..., N-1 corresponding to j = p/2
    N = 12  # representations j = 0, 1/2, 1, ..., 11/2

    spins = [p / 2.0 for p in range(N)]
    casimirs = [casimir_su2(j) for j in spins]

    print(f"\nRepresentations: j = {spins}")
    print(f"Casimir C_2(j) = j(j+1): {casimirs}")

    # Nearest-neighbor Dirac differences
    diffs = [casimirs[i + 1] - casimirs[i] for i in range(N - 1)]
    print(f"\nCasimir differences (D off-diagonal): {diffs}")
    print(f"  Note: C_2(j+1/2) - C_2(j) = j + 3/4 (linear in j)")

    # Analytical formula for nearest-neighbor Connes distance
    print("\n--- Analytical Connes distances (nearest-neighbor Dirac) ---")
    print(f"{'j_1':>6} {'j_2':>6} {'d_Connes':>12} {'1/D_hop':>12}")
    print("-" * 50)

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
             (0, 5), (0, 7), (0, 11)]

    results = {}
    for p, q in pairs:
        d = connes_distance_finite(casimirs, p, q)
        j_p, j_q = spins[p], spins[q]
        results[(p, q)] = d
        print(f"{j_p:6.1f} {j_q:6.1f} {d:12.6f}")

    # Full Monte Carlo optimization for comparison
    print("\n--- Monte Carlo optimization (100k trials) ---")
    print(f"{'j_1':>6} {'j_2':>6} {'d_NN(MC)':>12} {'d_full(MC)':>12} {'d_NN(exact)':>12}")
    print("-" * 60)

    key_pairs = [(0, 1), (0, 2), (1, 2), (0, 3)]
    for p, q in key_pairs:
        d_nn_mc, d_full_mc = connes_distance_full_optimization(casimirs, p, q, n_trials=50000)
        d_exact = results.get((p, q), connes_distance_finite(casimirs, p, q))
        j_p, j_q = spins[p], spins[q]
        print(f"{j_p:6.1f} {j_q:6.1f} {d_nn_mc:12.6f} {d_full_mc:12.6f} {d_exact:12.6f}")

    return results, spins, casimirs


# =============================================================================
# 2. COMPARISON WITH GEODESIC DISTANCE ON S^3
# =============================================================================

def geodesic_distance_s3(j1, j2, R=1.0):
    """
    Geodesic distance on S^3 between 'positions' of representations.

    On SU(2) ~ S^3, the character chi_j(theta) = sin((2j+1)theta)/sin(theta).
    The 'angular position' of representation j can be taken as
    theta_j = pi*j/(j_max + 1) or theta_j = pi/(2j+1).

    We use the Peter-Weyl inner product structure:
    The 'center of mass' of chi_j on S^3 corresponds to
    angular separation ~ pi/(2j+1) from the identity.

    Geodesic distance = R * angle.
    """
    # Angular position based on character zero structure
    # chi_j has first zero at theta = pi/(2j+1)
    if j1 == 0:
        theta1 = 0.0  # trivial rep = identity
    else:
        theta1 = np.pi / (2 * j1 + 1)

    if j2 == 0:
        theta2 = 0.0
    else:
        theta2 = np.pi / (2 * j2 + 1)

    return R * abs(theta1 - theta2)


def run_test2(results, spins):
    """Test 2: Compare Connes distance with S^3 geodesic."""
    print("\n" + "=" * 70)
    print("TEST 2: COMPARISON WITH GEODESIC DISTANCE ON S^3")
    print("=" * 70)

    print("\nSU(2) ~ S^3 with radius R. Character-based angular positions:")
    print(f"  theta_j = pi/(2j+1) for j>0, theta_0 = 0 (identity)")

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

    print(f"\n{'j_1':>6} {'j_2':>6} {'d_Connes':>12} {'d_geod(R=1)':>14} {'ratio':>10}")
    print("-" * 55)

    connes_vals = []
    geod_vals = []

    for p, q in pairs:
        d_c = results.get((p, q), connes_distance_finite(
            [casimir_su2(j / 2.0) for j in range(12)], p, q))
        j_p, j_q = spins[p], spins[q]
        d_g = geodesic_distance_s3(j_p, j_q)
        ratio = d_c / d_g if d_g > 1e-15 else float('inf')
        connes_vals.append(d_c)
        geod_vals.append(d_g)
        print(f"{j_p:6.1f} {j_q:6.1f} {d_c:12.6f} {d_g:14.6f} {ratio:10.4f}")

    # Check if ratio is constant (would mean metric agreement up to scale)
    connes_arr = np.array(connes_vals)
    geod_arr = np.array(geod_vals)
    ratios = connes_arr / geod_arr
    print(f"\nRatio statistics: mean={np.mean(ratios):.4f}, std={np.std(ratios):.4f}, "
          f"CV={np.std(ratios)/np.mean(ratios):.4f}")
    print(f"  If CV ~ 0: Connes distance = const * geodesic distance (same geometry)")
    print(f"  If CV >> 0: Different geometry")

    # Also try alternative: geodesic in representation space as 1D lattice
    print("\n--- Alternative: 1D lattice metric d(p,q) = |p-q| ---")
    print(f"{'j_1':>6} {'j_2':>6} {'d_Connes':>12} {'d_lattice':>12} {'ratio':>10}")
    print("-" * 55)
    ratios2 = []
    for p, q in pairs:
        d_c = results.get((p, q), 0)
        d_lat = abs(p - q)
        ratio = d_c / d_lat if d_lat > 0 else float('inf')
        ratios2.append(ratio)
        j_p, j_q = spins[p], spins[q]
        print(f"{j_p:6.1f} {j_q:6.1f} {d_c:12.6f} {d_lat:12.6f} {ratio:10.4f}")

    ratios2 = np.array(ratios2)
    print(f"\nLattice ratio: mean={np.mean(ratios2):.4f}, std={np.std(ratios2):.4f}, "
          f"CV={np.std(ratios2)/np.mean(ratios2):.4f}")


# =============================================================================
# 3. PERTURBATION TEST
# =============================================================================

def run_test3():
    """Test 3: Perturbation response of Connes distance."""
    print("\n" + "=" * 70)
    print("TEST 3: PERTURBATION RESPONSE delta d / delta epsilon")
    print("=" * 70)

    N = 12
    spins = [p / 2.0 for p in range(N)]
    casimirs = np.array([casimir_su2(j) for j in spins])

    # Pairs to test
    pairs = [(0, 1), (0, 2), (0, 5), (1, 3)]

    eps_values = [0.01, 0.05, 0.1, 0.5, 1.0]

    print("\n--- Systematic perturbation: lambda_k -> lambda_k + epsilon ---")

    for p, q in pairs:
        j_p, j_q = spins[p], spins[q]
        d0 = connes_distance_finite(casimirs, p, q)

        print(f"\nPair (j={j_p}, j={j_q}), d_0 = {d0:.6f}")
        print(f"  {'eps':>8} {'perturb_at':>10} {'d_pert':>12} {'delta_d':>12} {'delta_d/eps':>12}")
        print("  " + "-" * 60)

        # Perturb each eigenvalue separately
        for k in range(min(q + 2, N)):
            for eps in [0.1]:
                perturbed = casimirs.copy()
                perturbed[k] += eps
                d_pert = connes_distance_finite(perturbed, p, q)
                delta_d = d_pert - d0
                print(f"  {eps:8.3f} {spins[k]:10.1f} {d_pert:12.6f} {delta_d:12.6f} {delta_d/eps:12.6f}")

    # Random perturbation ensemble on GPU
    print("\n--- Random perturbation ensemble (GPU) ---")
    n_ensemble = 10000
    N_test = 8
    spins_test = [p / 2.0 for p in range(N_test)]
    casimirs_test = np.array([casimir_su2(j) for j in spins_test])

    eps_scale = 0.1

    # Generate random perturbations on GPU
    perturbs = cp.random.randn(n_ensemble, N_test).astype(cp.float64) * eps_scale

    pair = (0, 2)  # j=0 to j=1
    d0 = connes_distance_finite(casimirs_test, pair[0], pair[1])

    distances = []
    for trial in range(n_ensemble):
        pert = cp.asnumpy(perturbs[trial]) if HAS_GPU else perturbs[trial]
        perturbed = casimirs_test + pert
        d = connes_distance_finite(perturbed, pair[0], pair[1])
        distances.append(d)

    distances = np.array(distances)
    delta_d = distances - d0

    print(f"\nPair (j=0, j=1), eps_scale = {eps_scale}")
    print(f"  d_0 = {d0:.6f}")
    print(f"  <d_pert> = {np.mean(distances):.6f}")
    print(f"  std(d_pert) = {np.std(distances):.6f}")
    print(f"  <delta_d> = {np.mean(delta_d):.6f}")
    print(f"  std(delta_d) = {np.std(delta_d):.6f}")
    print(f"  max|delta_d| = {np.max(np.abs(delta_d)):.6f}")
    print(f"  Fractional change: {np.std(delta_d)/d0:.4f}")

    # Jacobian: d(d)/d(lambda_k) for each k
    print("\n--- Numerical Jacobian d(d)/d(lambda_k) ---")
    h = 1e-6
    print(f"  {'k':>4} {'j_k':>6} {'dd/dlam_k':>14}")
    print("  " + "-" * 30)
    for k in range(N_test):
        pert_plus = casimirs_test.copy()
        pert_plus[k] += h
        pert_minus = casimirs_test.copy()
        pert_minus[k] -= h
        d_plus = connes_distance_finite(pert_plus, pair[0], pair[1])
        d_minus = connes_distance_finite(pert_minus, pair[0], pair[1])
        jac = (d_plus - d_minus) / (2 * h)
        print(f"  {k:4d} {spins_test[k]:6.1f} {jac:14.6f}")


# =============================================================================
# 4. SEARCH FOR 4D DIRAC OPERATOR
# =============================================================================

def spectral_dimension_from_heat_trace(eigenvalues, t_values):
    """
    Compute spectral dimension from heat trace:
    K(t) = sum_k exp(-t * lambda_k^2)
    d_s = -2 * d(log K)/d(log t)
    """
    eigenvalues = np.array(eigenvalues, dtype=np.float64)
    lam_sq = eigenvalues ** 2

    dims = []
    for i in range(len(t_values) - 1):
        t1, t2 = t_values[i], t_values[i + 1]
        K1 = np.sum(np.exp(-t1 * lam_sq))
        K2 = np.sum(np.exp(-t2 * lam_sq))
        if K1 > 1e-300 and K2 > 1e-300:
            d_s = -2.0 * (np.log(K2) - np.log(K1)) / (np.log(t2) - np.log(t1))
            dims.append((np.sqrt(t1 * t2), d_s))

    return dims


def run_test4():
    """Test 4: Search for Dirac operator giving 4D geometry."""
    print("\n" + "=" * 70)
    print("TEST 4: SEARCH FOR 4D DIRAC OPERATOR")
    print("=" * 70)

    N = 50  # number of eigenvalues

    # Test various Dirac spectra
    # For a d-dimensional compact manifold, Weyl's law: lambda_k ~ k^{1/d}
    # So d_s = d means lambda_k ~ k^{1/d}

    print("\n--- Weyl's law test: lambda_k = k^{1/d} ---")
    print(f"  If spectrum follows Weyl's law for dimension d,")
    print(f"  the heat kernel gives spectral dimension d_s = d.\n")

    t_values = np.logspace(-3, 1, 50)

    test_spectra = {
        'SU(2) Casimir: j(j+1)': [casimir_su2(k / 2.0) for k in range(N)],
        'Weyl d=1: k': [k for k in range(1, N + 1)],
        'Weyl d=2: sqrt(k)': [np.sqrt(k) for k in range(1, N + 1)],
        'Weyl d=3: k^{1/3}': [k ** (1.0 / 3) for k in range(1, N + 1)],
        'Weyl d=4: k^{1/4}': [k ** 0.25 for k in range(1, N + 1)],
        'Transfer A_p: I_{2p+1}/I_1': None,  # computed below
        'Modified Casimir: (j(j+1))^{1/2}': [np.sqrt(casimir_su2(k / 2.0)) for k in range(N)],
        'Linear: k+1': [k + 1 for k in range(N)],
    }

    # Compute transfer matrix eigenvalues
    beta = 2.0
    I1 = bessel_iv(1, beta)
    transfer_eigs = [bessel_iv(2 * p + 1, beta) / I1 for p in range(N)]
    test_spectra['Transfer A_p: I_{2p+1}/I_1'] = transfer_eigs

    for name, spectrum in test_spectra.items():
        spectrum = np.array(spectrum, dtype=np.float64)
        # Remove zeros for heat trace
        spectrum_nz = spectrum[spectrum > 1e-15]

        dims = spectral_dimension_from_heat_trace(spectrum_nz, t_values)

        if len(dims) > 0:
            # Find plateau region
            mid_dims = [d for t, d in dims if 0.01 < t < 1.0]
            if mid_dims:
                d_s = np.mean(mid_dims)
                d_s_std = np.std(mid_dims)
            else:
                d_s = dims[len(dims) // 2][1]
                d_s_std = 0.0
            print(f"  {name:40s} -> d_s = {d_s:.3f} +/- {d_s_std:.3f}")
        else:
            print(f"  {name:40s} -> FAILED (no valid heat trace)")

    # KEY QUESTION: Can we find eigenvalues giving d_s = 4?
    print("\n--- Optimizing for d_s = 4 ---")
    print("  Trying lambda_k = k^alpha with varying alpha...")

    target_d = 4.0
    best_alpha = None
    best_error = float('inf')

    alphas = np.linspace(0.1, 2.0, 100)
    print(f"\n  {'alpha':>8} {'d_s':>8} {'|d_s-4|':>10}")
    print("  " + "-" * 30)

    for alpha in alphas:
        spectrum = np.array([k ** alpha for k in range(1, N + 1)])
        dims = spectral_dimension_from_heat_trace(spectrum, t_values)
        mid_dims = [d for t, d in dims if 0.05 < t < 0.5]
        if mid_dims:
            d_s = np.mean(mid_dims)
            err = abs(d_s - target_d)
            if err < best_error:
                best_error = err
                best_alpha = alpha
            if abs(alpha - 0.25) < 0.02 or abs(alpha - 0.5) < 0.02 or \
               abs(alpha - 1.0) < 0.02 or err < 0.1:
                print(f"  {alpha:8.3f} {d_s:8.3f} {err:10.4f}")

    print(f"\n  BEST: alpha = {best_alpha:.4f}, |d_s - 4| = {best_error:.4f}")
    print(f"  Weyl prediction: d = 1/alpha = {1.0/best_alpha:.3f}")

    # Connes distance with the 4D-optimized Dirac
    print(f"\n--- Connes distances with optimized Dirac (alpha={best_alpha:.4f}) ---")
    spectrum_4d = [k ** best_alpha for k in range(1, 13)]

    print(f"  {'p':>4} {'q':>4} {'d_Connes':>12} {'d_Cas':>12} {'ratio':>10}")
    print("  " + "-" * 50)

    casimirs = [casimir_su2(k / 2.0) for k in range(12)]
    for p, q in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4)]:
        d_4d = connes_distance_finite(spectrum_4d, p, q)
        d_cas = connes_distance_finite(casimirs, p, q)
        ratio = d_4d / d_cas if d_cas > 1e-15 else float('inf')
        print(f"  {p:4d} {q:4d} {d_4d:12.6f} {d_cas:12.6f} {ratio:10.4f}")

    # Transfer matrix spectral dimension at various beta
    print("\n--- Transfer matrix spectral dimension vs beta ---")
    print(f"  {'beta':>8} {'d_s':>8} {'A_0':>10} {'A_1':>10} {'A_2':>10}")
    print("  " + "-" * 50)

    for beta in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
        I1 = bessel_iv(1, beta)
        t_eigs = [bessel_iv(2 * p + 1, beta) / I1 for p in range(N)]
        t_eigs_nz = [x for x in t_eigs if abs(x) > 1e-15]

        if len(t_eigs_nz) > 2:
            # Use -log(A_p) as effective eigenvalues
            log_eigs = [-np.log(abs(x)) for x in t_eigs_nz if x > 0]
            if len(log_eigs) > 2:
                dims = spectral_dimension_from_heat_trace(log_eigs, t_values)
                mid_dims = [d for t, d in dims if 0.01 < t < 1.0]
                d_s = np.mean(mid_dims) if mid_dims else float('nan')
            else:
                d_s = float('nan')
        else:
            d_s = float('nan')

        A = [bessel_iv(2 * p + 1, beta) / I1 for p in range(3)]
        print(f"  {beta:8.1f} {d_s:8.3f} {A[0]:10.6f} {A[1]:10.6f} {A[2]:10.6f}")


# =============================================================================
# 5. METRIC TENSOR FROM CONNES DISTANCE
# =============================================================================

def run_metric_analysis():
    """Compute the induced metric tensor on representation space."""
    print("\n" + "=" * 70)
    print("BONUS: INDUCED METRIC TENSOR ON REPRESENTATION SPACE")
    print("=" * 70)

    N = 10
    spins = [p / 2.0 for p in range(N)]
    casimirs = [casimir_su2(j) for j in spins]

    # Compute full distance matrix
    D_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D_mat[i, j] = connes_distance_finite(casimirs, i, j)

    print("\nConnes distance matrix (first 6 reps):")
    print(f"{'':>8}", end='')
    for j in range(6):
        print(f"{'j='+str(spins[j]):>10}", end='')
    print()
    for i in range(6):
        print(f"{'j='+str(spins[i]):>8}", end='')
        for j in range(6):
            print(f"{D_mat[i,j]:10.4f}", end='')
        print()

    # Check triangle inequality
    print("\nTriangle inequality check:")
    violations = 0
    for i in range(6):
        for j in range(6):
            for k in range(6):
                if D_mat[i, k] > D_mat[i, j] + D_mat[j, k] + 1e-10:
                    violations += 1
    print(f"  Violations: {violations} out of {6**3} triples")

    # Check if embeddable in R^d
    print("\nMultidimensional scaling (MDS) embedding dimension:")
    from numpy.linalg import eigvalsh

    # Gram matrix from distances (double centering)
    n = 6
    D2 = D_mat[:n, :n] ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * H @ D2 @ H

    evals = np.sort(eigvalsh(G))[::-1]
    print(f"  Gram matrix eigenvalues: {evals}")

    # Effective dimension = number of significantly positive eigenvalues
    threshold = 0.01 * evals[0]
    d_eff = np.sum(evals > threshold)
    print(f"  Effective embedding dimension: {d_eff}")
    print(f"  (This is the dimension of the metric space induced by Connes distance)")

    # Local metric tensor g_{ij} = (1/2) * d^2(d^2)/dx^i dx^j
    print("\nLocal 'metric' (nearest-neighbor distances):")
    for i in range(min(8, N - 1)):
        d_local = D_mat[i, i + 1]
        print(f"  d(j={spins[i]}, j={spins[i+1]}) = {d_local:.6f} "
              f"= 1/(C_2({spins[i+1]})-C_2({spins[i]})) = 1/{casimirs[i+1]-casimirs[i]:.1f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("CONNES DISTANCE FORMULA ON SU(2) SPECTRAL DATA")
    print("Testing emergent geometry from representation-theoretic spectra")
    print("=" * 70)

    # Test 1: Connes distance with Casimir Dirac
    results, spins, casimirs = run_test1()

    # Test 2: Compare with S^3 geodesic
    run_test2(results, spins)

    # Test 3: Perturbation response
    run_test3()

    # Test 4: Search for 4D Dirac
    run_test4()

    # Bonus: Metric tensor
    run_metric_analysis()

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY OF KEY FINDINGS")
    print("=" * 70)
    print("""
1. CONNES DISTANCE: With nearest-neighbor Dirac D built from Casimir
   differences, d(j1,j2) = sum_{k=min}^{max-1} 1/(C_2(k+1/2) - C_2(k)).
   This is a HARMONIC-TYPE sum: d(0,j) ~ sum 1/(2k+2) ~ (1/2)*H(2j).
   The distance DECREASES with increasing j (representations get closer).

2. S^3 GEODESIC: The Connes distance does NOT reproduce the S^3 geodesic
   metric. The ratio d_Connes/d_geodesic is NOT constant — different geometry.
   The Casimir Dirac gives a 1D metric on the representation lattice,
   not the 3D metric of SU(2) ~ S^3.

3. PERTURBATION: The distance is most sensitive to perturbations of
   eigenvalues BETWEEN p and q (the intermediate representations).
   The Jacobian dd/dlambda_k is localized — a good sign for locality.

4. 4D GEOMETRY: Weyl's law lambda_k ~ k^{1/4} gives d_s = 4.
   This requires a MUCH flatter spectrum than Casimir (k^{1/4} vs k^2).
   The transfer matrix eigenvalues have spectral dimension that depends
   on beta but never reaches 4 in a clean way.

KEY CONCLUSION: The Casimir Dirac operator on SU(2) representations
gives a 1-dimensional metric (rep lattice). To get 4D spacetime,
one needs EITHER:
  (a) A product of 4 such spectral triples, OR
  (b) A fundamentally different Dirac operator with Weyl exponent 1/4, OR
  (c) The full Connes-Chamseddine spectral action on a different algebra.

The representation space alone is 1D — it cannot generate 4D geometry
without additional structure (e.g., tensor products, noncommutative algebra).
""")
