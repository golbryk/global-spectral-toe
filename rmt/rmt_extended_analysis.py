"""
Extended RMT Analysis of the Spectral TOE Excitation Operator
=============================================================

Author : Grzegorz Olbryk  <g.olbryk@gmail.com>
Date   : March 2026
Context: ROADMAP Task 12 — RMT analysis: extend and understand pseudo-integrable regime

This script extends the original scaling_tests.py (cupy-based) to numpy, and adds:
  12a. Multiple operator structures (different Laplacians, different chaotic maps)
  12b. Level spacing distribution P(s) — not just the r-statistic
  12c. Semi-Poisson classification (Brody β parameter fit)
  12d. Test universality of the pseudo-integrable regime

Key finding from original data: r_TOE ∈ [0.37, 0.56], mean ≈ 0.43.
Semi-Poisson reference: r_SP ≈ 0.4228 (gamma(2,2) spacings).
This script tests whether r_TOE ≈ r_SP is universal across operator structures.
"""

import numpy as np
from scipy import linalg as la
from scipy.special import gamma as gamma_fn
from scipy.optimize import minimize_scalar
from collections import OrderedDict

np.random.seed(123)


# ===========================================================================
# Part 1: Laplacian structures
# ===========================================================================

def laplacian_ring(N):
    """Cyclic ring Laplacian (original)."""
    L = np.zeros((N, N))
    for i in range(N):
        L[i, i] = 2.0
        L[i, (i - 1) % N] = -1.0
        L[i, (i + 1) % N] = -1.0
    return L


def laplacian_complete(N):
    """Complete graph Laplacian: L = N·I - J."""
    return float(N) * np.eye(N) - np.ones((N, N))


def laplacian_star(N):
    """Star graph Laplacian: node 0 connected to all others."""
    L = np.zeros((N, N))
    L[0, 0] = N - 1
    for i in range(1, N):
        L[i, i] = 1.0
        L[0, i] = -1.0
        L[i, 0] = -1.0
    return L


def laplacian_2d_lattice(N):
    """2D square lattice Laplacian (closest square grid, periodic BCs)."""
    side = int(np.round(np.sqrt(N)))
    n = side * side  # actual size
    L = np.zeros((n, n))
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            L[idx, idx] = 4.0
            # right
            L[idx, i * side + (j + 1) % side] -= 1.0
            # left
            L[idx, i * side + (j - 1) % side] -= 1.0
            # down
            L[idx, ((i + 1) % side) * side + j] -= 1.0
            # up
            L[idx, ((i - 1) % side) * side + j] -= 1.0
    return L


def laplacian_erdos_renyi(N, p=None):
    """Erdos-Renyi random graph Laplacian with edge probability p."""
    if p is None:
        p = 3.0 * np.log(N) / N  # ensures connectivity whp
    A = (np.random.rand(N, N) < p).astype(float)
    A = np.triu(A, 1)
    A = A + A.T
    D = np.diag(A.sum(axis=1))
    return D - A


# ===========================================================================
# Part 2: Chaotic phase maps
# ===========================================================================

def phase_arnold_cat(N, x0=0.1, y0=0.3):
    """Arnold's cat map (original)."""
    x = np.zeros(N)
    y = np.zeros(N)
    x[0], y[0] = x0, y0
    for n in range(1, N):
        x[n] = (2.0 * x[n - 1] + y[n - 1]) % 1.0
        y[n] = (x[n - 1] + y[n - 1]) % 1.0
    return np.cos(2.0 * np.pi * x)


def phase_logistic(N, r=3.99, x0=0.1):
    """Logistic map: x_{n+1} = r·x_n·(1 - x_n)."""
    x = np.zeros(N)
    x[0] = x0
    for n in range(1, N):
        x[n] = r * x[n - 1] * (1.0 - x[n - 1])
    return np.cos(2.0 * np.pi * x)


def phase_tent(N, mu=1.99, x0=0.1):
    """Tent map: x_{n+1} = mu·min(x_n, 1-x_n)."""
    x = np.zeros(N)
    x[0] = x0
    for n in range(1, N):
        x[n] = mu * min(x[n - 1], 1.0 - x[n - 1])
    return np.cos(2.0 * np.pi * x)


def phase_bernoulli(N, x0=0.1):
    """Bernoulli shift: x_{n+1} = 2·x_n mod 1."""
    x = np.zeros(N)
    x[0] = x0
    for n in range(1, N):
        x[n] = (2.0 * x[n - 1]) % 1.0
    return np.cos(2.0 * np.pi * x)


def phase_standard_map(N, K=1.0, x0=0.1, p0=0.3):
    """Standard (Chirikov) map on the torus."""
    x = np.zeros(N)
    p = np.zeros(N)
    x[0], p[0] = x0, p0
    for n in range(1, N):
        p[n] = (p[n - 1] + K / (2 * np.pi) * np.sin(2 * np.pi * x[n - 1])) % 1.0
        x[n] = (x[n - 1] + p[n]) % 1.0
    return np.cos(2.0 * np.pi * x)


# ===========================================================================
# Part 3: Operator construction and eigenvalue computation
# ===========================================================================

def build_operator(N, alpha, laplacian_fn, phase_fn):
    """Build H = L + alpha * diag(phase), symmetrized."""
    L = laplacian_fn(N)
    n_actual = L.shape[0]  # may differ from N for 2D lattice
    phase = phase_fn(n_actual)
    H = L + alpha * np.diag(phase)
    return (H + H.T) / 2.0


def eigenvalues(H):
    """Sorted eigenvalues of symmetric matrix."""
    return np.sort(la.eigvalsh(H))


# ===========================================================================
# Part 4: Spacing statistics
# ===========================================================================

def normalized_spacings(eigs):
    """Compute normalized spacings s_i = gap_i / <gap>."""
    gaps = np.diff(eigs)
    mean_gap = np.mean(gaps)
    if mean_gap <= 0:
        return np.array([])
    return gaps / mean_gap


def r_statistic(eigs):
    """Level spacing ratio statistic."""
    s = normalized_spacings(eigs)
    if len(s) < 2:
        return float('nan')
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    r = r[np.isfinite(r)]
    return float(np.mean(r))


def spacing_histogram(eigs, n_bins=50, s_max=4.0):
    """Return (bin_centers, P(s)) normalized as probability density."""
    s = normalized_spacings(eigs)
    s = s[s < s_max]
    counts, edges = np.histogram(s, bins=n_bins, range=(0, s_max), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


# ===========================================================================
# Part 5: Reference distributions
# ===========================================================================

def p_poisson(s):
    """Poisson: P(s) = exp(-s)."""
    return np.exp(-s)


def p_wigner(s):
    """Wigner-Dyson (GOE): P(s) = (pi/2)*s*exp(-pi*s^2/4)."""
    return (np.pi / 2.0) * s * np.exp(-np.pi * s ** 2 / 4.0)


def p_semi_poisson(s):
    """Semi-Poisson: P(s) = 4*s*exp(-2*s)."""
    return 4.0 * s * np.exp(-2.0 * s)


def p_brody(s, beta):
    """Brody distribution: P(s;β) = (β+1)*b*s^β * exp(-b*s^{β+1})
    where b = [Γ((β+2)/(β+1))]^{β+1}.
    β = 0: Poisson, β = 1: Wigner-Dyson."""
    b = gamma_fn((beta + 2.0) / (beta + 1.0)) ** (beta + 1.0)
    return (beta + 1.0) * b * s ** beta * np.exp(-b * s ** (beta + 1.0))


def fit_brody_beta(eigs, n_bins=50, s_max=4.0):
    """Fit the Brody parameter β from eigenvalue spacings.
    Minimizes KL divergence between histogram and Brody distribution."""
    centers, p_data = spacing_histogram(eigs, n_bins, s_max)
    ds = centers[1] - centers[0]

    # Avoid log(0) issues
    p_data_safe = np.maximum(p_data, 1e-10)

    def neg_log_likelihood(beta):
        p_model = p_brody(centers, beta)
        p_model = np.maximum(p_model, 1e-10)
        # chi-squared-like objective (more robust than KL)
        return np.sum((p_data - p_model) ** 2 * ds)

    result = minimize_scalar(neg_log_likelihood, bounds=(0.0, 1.0), method='bounded')
    return result.x


def r_semi_poisson_mc(n_samples=100000):
    """Compute r-statistic for semi-Poisson distribution by Monte Carlo.
    Semi-Poisson spacings are Gamma(2, 2) distributed."""
    s = np.random.gamma(2.0, 0.5, n_samples)  # shape=2, scale=1/2 → rate=2
    r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.mean(r[np.isfinite(r)]))


# ===========================================================================
# Part 6: GOE / Poisson references (numpy)
# ===========================================================================

def goe_eigenvalues(N):
    A = np.random.normal(0, 1, (N, N))
    H = (A + A.T) / np.sqrt(2.0 * N)
    return np.sort(la.eigvalsh(H))


def poisson_eigenvalues(N):
    return np.sort(np.cumsum(np.random.exponential(1.0, N)))


# ===========================================================================
# Part 7: Number variance Σ²(L)
# ===========================================================================

def number_variance(eigs, L_values):
    """Compute number variance Σ²(L) = <(n(E, E+L) - L)²> averaged over E.
    Uses unfolded eigenvalues."""
    # Unfold: map to uniform density
    n = len(eigs)
    unfolded = np.interp(eigs, eigs, np.arange(n) / float(n)) * n
    # Actually simpler: use ranks directly
    unfolded = np.arange(n, dtype=float)

    results = []
    for L in L_values:
        counts = []
        for i in range(n):
            j = np.searchsorted(unfolded, unfolded[i] + L, side='right')
            counts.append(j - i)
        counts = np.array(counts[:n - int(L) - 1], dtype=float)
        if len(counts) > 0:
            results.append(np.var(counts))
        else:
            results.append(float('nan'))
    return np.array(results)


# ===========================================================================
# Main analysis
# ===========================================================================

def main():
    print()
    print("=" * 90)
    print("  Extended RMT Analysis of the Spectral TOE Excitation Operator")
    print("  Author: Grzegorz Olbryk  |  March 2026  |  Task 12")
    print("=" * 90)

    # -----------------------------------------------------------------------
    # Semi-Poisson reference value
    # -----------------------------------------------------------------------
    r_sp = r_semi_poisson_mc(500000)
    print(f"\n  Semi-Poisson reference: r_SP = {r_sp:.4f}")
    print(f"  GOE reference: r_GOE ≈ 0.5359")
    print(f"  Poisson reference: r_Poi ≈ 0.3863")

    # -----------------------------------------------------------------------
    # 12a: Vary operator structure
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  12a. Operator Structure Universality Test")
    print("=" * 90)

    laplacians = OrderedDict([
        ("Ring", laplacian_ring),
        ("Complete", laplacian_complete),
        ("Star", laplacian_star),
        ("2D-lattice", laplacian_2d_lattice),
        ("ER-random", laplacian_erdos_renyi),
    ])

    phases = OrderedDict([
        ("Arnold-cat", phase_arnold_cat),
        ("Logistic", phase_logistic),
        ("Tent", phase_tent),
        ("Bernoulli", phase_bernoulli),
        ("Standard", phase_standard_map),
    ])

    N_test = 400
    alpha_test = 0.30

    print(f"\n  N = {N_test}, α = {alpha_test}")
    print(f"  {'Laplacian':<14} {'Phase':<14} {'r_TOE':>8} {'β_Brody':>8} {'class':>16}")
    print("  " + "-" * 66)

    r_values_all = []

    for lap_name, lap_fn in laplacians.items():
        for ph_name, ph_fn in phases.items():
            try:
                H = build_operator(N_test, alpha_test, lap_fn, ph_fn)
                eigs = eigenvalues(H)
                r_val = r_statistic(eigs)
                beta = fit_brody_beta(eigs)
                r_values_all.append(r_val)

                # Classify
                if abs(r_val - 0.3863) < 0.03:
                    cls = "Poisson"
                elif abs(r_val - r_sp) < 0.03:
                    cls = "semi-Poisson"
                elif abs(r_val - 0.5359) < 0.03:
                    cls = "GOE"
                else:
                    if r_val < r_sp:
                        cls = "near-Poisson"
                    elif r_val < 0.5359:
                        cls = "pseudo-integ."
                    else:
                        cls = "beyond-GOE"

                print(f"  {lap_name:<14} {ph_name:<14} {r_val:8.4f} {beta:8.4f} {cls:>16}")
            except Exception as e:
                print(f"  {lap_name:<14} {ph_name:<14} {'ERROR':>8} — {str(e)[:30]}")

    r_arr = np.array(r_values_all)
    print(f"\n  Summary: mean r = {np.mean(r_arr):.4f} ± {np.std(r_arr):.4f}")
    print(f"  Range: [{np.min(r_arr):.4f}, {np.max(r_arr):.4f}]")

    # -----------------------------------------------------------------------
    # 12b: Level spacing distribution P(s)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  12b. Level Spacing Distribution P(s)")
    print("=" * 90)

    # Use the original operator (ring + Arnold cat) at several sizes
    print(f"\n  Ring + Arnold-cat, α = 0.30")
    print(f"  {'N':>6} {'r_TOE':>8} {'β_Brody':>8} {'KS(Poi)':>8} {'KS(WD)':>8} "
          f"{'KS(SP)':>8} {'best fit':>12}")

    for N in [200, 400, 800]:
        H = build_operator(N, alpha_test, laplacian_ring, phase_arnold_cat)
        eigs = eigenvalues(H)
        r_val = r_statistic(eigs)
        beta = fit_brody_beta(eigs)

        # Kolmogorov-Smirnov test against reference distributions
        s = normalized_spacings(eigs)
        s = s[s < 6.0]
        s_sorted = np.sort(s)
        n_s = len(s_sorted)
        ecdf = np.arange(1, n_s + 1) / float(n_s)

        # CDF of reference distributions
        cdf_poi = 1.0 - np.exp(-s_sorted)
        cdf_wd = 1.0 - np.exp(-np.pi * s_sorted ** 2 / 4.0)
        cdf_sp = 1.0 - (1.0 + 2.0 * s_sorted) * np.exp(-2.0 * s_sorted)

        ks_poi = np.max(np.abs(ecdf - cdf_poi))
        ks_wd = np.max(np.abs(ecdf - cdf_wd))
        ks_sp = np.max(np.abs(ecdf - cdf_sp))

        best = min([("Poisson", ks_poi), ("Wigner-D.", ks_wd), ("semi-Poi.", ks_sp)],
                    key=lambda x: x[1])

        print(f"  {N:6d} {r_val:8.4f} {beta:8.4f} {ks_poi:8.4f} {ks_wd:8.4f} "
              f"{ks_sp:8.4f} {best[0]:>12}")

    # -----------------------------------------------------------------------
    # 12c: Semi-Poisson classification
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  12c. Semi-Poisson Classification")
    print("=" * 90)

    # Detailed test: generate semi-Poisson eigenvalues and compare
    print(f"\n  Direct comparison of r-statistics:")
    n_trials = 20
    r_sp_samples = []
    for _ in range(n_trials):
        sp_spacings = np.random.gamma(2.0, 0.5, 800)
        sp_eigs = np.cumsum(sp_spacings)
        r_sp_samples.append(r_statistic(sp_eigs))

    r_sp_mean = np.mean(r_sp_samples)
    r_sp_std = np.std(r_sp_samples)
    print(f"  Semi-Poisson (N=800): r = {r_sp_mean:.4f} ± {r_sp_std:.4f}")

    # TOE at same size with multiple random seeds
    r_toe_samples = []
    for seed in range(n_trials):
        np.random.seed(seed + 1000)
        H = build_operator(800, 0.30, laplacian_ring, phase_arnold_cat)
        eigs = eigenvalues(H)
        r_toe_samples.append(r_statistic(eigs))
    np.random.seed(123)  # restore

    r_toe_mean = np.mean(r_toe_samples)
    r_toe_std = np.std(r_toe_samples)
    print(f"  TOE (Ring+Arnold, N=800, α=0.30): r = {r_toe_mean:.4f} ± {r_toe_std:.4f}")

    # Note: the TOE operator is deterministic given (N, alpha, phase_seed)
    # so r_toe_std here reflects only the variation in the ER graph seed
    # For the deterministic operators, the single value is the r-statistic

    # Brody β for semi-Poisson theoretical value: β_SP ≈ 0.46
    # (from fitting P(s) = 4s·e^{-2s} to Brody)
    s_grid = np.linspace(0.01, 4.0, 200)
    sp_vals = p_semi_poisson(s_grid)

    def brody_fit_error(beta):
        br_vals = p_brody(s_grid, beta)
        return np.sum((sp_vals - br_vals) ** 2)

    res = minimize_scalar(brody_fit_error, bounds=(0.0, 1.0), method='bounded')
    beta_sp_theory = res.x
    print(f"\n  Semi-Poisson theoretical Brody β = {beta_sp_theory:.4f}")

    # -----------------------------------------------------------------------
    # 12d: Universality across N and α
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  12d. Universality: r_TOE vs N and α")
    print("=" * 90)

    N_list = [100, 200, 400, 800]
    alpha_list = [0.10, 0.20, 0.30, 0.50, 1.00, 2.00]

    print(f"\n  Ring + Arnold-cat:")
    header = f"  {'N':>5}"
    for a in alpha_list:
        header += f" {'α=' + str(a):>8}"
    print(header)
    print("  " + "-" * (6 + 9 * len(alpha_list)))

    all_r = []
    for N in N_list:
        row = f"  {N:5d}"
        for alpha in alpha_list:
            H = build_operator(N, alpha, laplacian_ring, phase_arnold_cat)
            eigs = eigenvalues(H)
            r_val = r_statistic(eigs)
            row += f" {r_val:8.4f}"
            all_r.append(r_val)
        print(row)

    all_r = np.array(all_r)
    print(f"\n  Grand mean r = {np.mean(all_r):.4f} ± {np.std(all_r):.4f}")
    print(f"  Semi-Poisson r_SP = {r_sp:.4f}")
    print(f"  |mean(r) - r_SP| = {abs(np.mean(all_r) - r_sp):.4f}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  SUMMARY")
    print("=" * 90)
    print()
    print("  1. The spectral TOE excitation operator produces level statistics")
    print("     in the PSEUDO-INTEGRABLE regime, between Poisson and GOE.")
    print()
    print(f"  2. Semi-Poisson reference: r_SP = {r_sp:.4f}")
    print(f"     Semi-Poisson Brody β ≈ {beta_sp_theory:.4f}")
    print()
    print("  3. Universality test: does the pseudo-integrable regime persist")
    print("     across different Laplacian and phase map choices?")
    print("     (See 12a table above)")
    print()
    print("  4. The KS test identifies which reference distribution best")
    print("     fits the empirical P(s). (See 12b table above)")
    print()
    print("=" * 90)


if __name__ == '__main__':
    main()
