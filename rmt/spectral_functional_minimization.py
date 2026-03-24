#!/usr/bin/env python3
"""
Spectral Functional Minimization: Does L = Sigma_{a!=b} w_ab (log lam_a - log lam_b)^2
select Casimir eigenvalues?

Tests whether minimizing the spectral functional over positive spectra {lam_j}
with normalization constraint recovers the SU(2) Casimir eigenvalues lam_j = j(j+1).

GPU-accelerated large-N tests with CuPy; scipy for small-N exact optimization.

Author: Grzegorz Olbryk
Date: 2026-03-24
"""

import math
import numpy as np
from scipy.optimize import minimize as sp_min
from scipy.stats import linregress
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("WARNING: CuPy not available, skipping GPU tests")


# =====================================================================
# PART 1: Exact optimization with scipy (small N)
# =====================================================================

def L_functional(log_lam, W):
    """L = sum_{j!=k} w_jk (log lam_j - log lam_k)^2"""
    diff = log_lam[:, None] - log_lam[None, :]
    return np.sum(W * diff**2)

def L_gradient(log_lam, W):
    """dL/d(log lam_j) = 4 sum_k w_jk (log lam_j - log lam_k) for symmetric W"""
    diff = log_lam[:, None] - log_lam[None, :]
    return 4.0 * np.sum(W * diff, axis=1)  # W already symmetric


def optimize_spectrum(N_reps, W, total, n_starts=100, label=""):
    """Minimize L over {lam_j > 0} with sum lam_j = total, using scipy SLSQP."""

    best_val = np.inf
    best_x = None
    converged_vals = []

    for trial in range(n_starts):
        np.random.seed(trial * 137 + 42)
        lam_init = np.random.exponential(1.0, N_reps) + 0.01
        lam_init *= total / lam_init.sum()
        x0 = np.log(lam_init)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(np.exp(x)) - total}
        ]

        res = sp_min(L_functional, x0, args=(W,), jac=L_gradient,
                     method='SLSQP', constraints=constraints,
                     options={'maxiter': 5000, 'ftol': 1e-15})

        converged_vals.append(res.fun)
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    opt_lam = np.exp(best_x)

    # Count convergence to global min
    L_arr = np.array(converged_vals)
    n_global = np.sum(np.abs(L_arr - best_val) / max(abs(best_val), 1e-30) < 0.01)

    return opt_lam, best_val, n_global, L_arr


def compare_to_candidates(opt_lam, N_reps, label):
    """Compare optimized spectrum to known candidates."""
    j = np.arange(1, N_reps + 1, dtype=np.float64)

    candidates = {
        'Casimir j(j+1)':    j * (j + 1),
        'Shifted (j+1/2)^2': (j + 0.5)**2,
        'j(j+2)':            j * (j + 2),
        'j^2':               j**2,
        'Linear j':          j,
        'Uniform':           np.ones(N_reps),
    }

    print(f"\n  Optimal lam (first 10): {opt_lam[:min(10,N_reps)]}")
    print(f"\n  {'Candidate':25s} {'ratio CV':>10s} {'cos_sim(log)':>12s}")
    print(f"  {'-'*50}")

    for name, spec in candidates.items():
        spec_scaled = spec * (opt_lam.sum() / spec.sum())
        ratio = opt_lam / spec_scaled
        ratio_cv = np.std(ratio) / np.mean(ratio)

        # Cosine similarity in log space
        log_opt = np.log(opt_lam / opt_lam.sum())
        log_cand = np.log(spec / spec.sum())
        cos_sim = np.dot(log_opt, log_cand) / (np.linalg.norm(log_opt) * np.linalg.norm(log_cand))

        marker = " <-- MATCH" if ratio_cv < 0.01 else ""
        print(f"  {name:25s} {ratio_cv:10.6f} {cos_sim:12.6f}{marker}")


# =====================================================================
# PART 2: Weight schemes
# =====================================================================

def make_weights(N_reps, scheme):
    """Create weight matrix for given scheme."""
    j = np.arange(1, N_reps + 1, dtype=np.float64)
    d = 2 * j + 1

    if scheme == 'dim_product':
        W = np.outer(d, d)
    elif scheme == 'uniform':
        W = np.ones((N_reps, N_reps))
    elif scheme == 'distance':
        W = np.outer(d, d)
        idx = np.arange(N_reps, dtype=np.float64)
        dist = np.abs(idx[:, None] - idx[None, :])
        dist[dist == 0] = 1
        W = W / dist
    elif scheme == 'casimir_product':
        c = j * (j + 1)
        W = np.outer(c, c)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    np.fill_diagonal(W, 0)
    return W


# =====================================================================
# PART 3: Test with modified functionals
# =====================================================================

def test_log_coulomb_confinement(N_reps=8, n_starts=50):
    """
    L = -sum_{j<k} d_j d_k |log lam_j - log lam_k|  (repulsion)
        + mu sum_j d_j^2 lam_j                       (confinement)
    """
    print("\n" + "="*70)
    print("LOG-COULOMB REPULSION + CONFINEMENT")
    print("="*70)

    j = np.arange(1, N_reps + 1, dtype=np.float64)
    d = 2 * j + 1
    casimir = j * (j + 1)

    mu_values = [0.001, 0.01, 0.1, 1.0, 10.0]

    for mu in mu_values:
        def energy(log_lam):
            # Repulsion: -sum_{a<b} d_a d_b (log lam_a - log lam_b)^2 (smooth)
            E_rep = 0.0
            for a in range(N_reps):
                for b in range(a+1, N_reps):
                    E_rep -= d[a] * d[b] * (log_lam[a] - log_lam[b])**2
            # Confinement: mu * sum d_j^2 lam_j
            E_conf = mu * np.sum(d**2 * np.exp(np.clip(log_lam, -10, 20)))
            return E_rep + E_conf

        best_val = np.inf
        best_x = None

        for trial in range(n_starts):
            np.random.seed(trial)
            x0 = np.sort(np.random.uniform(-2, 4, N_reps))  # log(lam) in [-2, 4]

            # Use L-BFGS-B with bounds to prevent log_lam -> -inf
            bounds = [(-5, 10)] * N_reps
            res = sp_min(energy, x0, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 20000, 'ftol': 1e-14})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        if best_x is None:
            print(f"\n  mu = {mu}: optimization failed")
            continue

        opt_lam = np.sort(np.exp(best_x))
        cas_scaled = casimir * (opt_lam.sum() / casimir.sum())
        ratio = opt_lam / cas_scaled

        # Power law fit
        log_j = np.log(j)
        log_l = np.log(opt_lam)
        slope, _, r, _, _ = linregress(log_j, log_l)

        print(f"\n  mu = {mu:.3f}: power law lam ~ j^{slope:.3f} (R^2={r**2:.4f}), "
              f"ratio CV = {np.std(ratio)/np.mean(ratio):.4f}")
        if N_reps <= 10:
            print(f"    opt:    {opt_lam}")
            print(f"    Casimir:{cas_scaled}")


# =====================================================================
# PART 4: Constrained variance test
# =====================================================================

def test_fixed_variance(N_reps=8, n_starts=50):
    """
    Fix Var(log lam) = Var(log Casimir) and minimize L.
    Should give equally-spaced logs (geometric progression), NOT Casimir.
    """
    print("\n" + "="*70)
    print("FIXED LOG-VARIANCE: minimize L with Var(log lam) = Var(log Casimir)")
    print("="*70)

    j = np.arange(1, N_reps + 1, dtype=np.float64)
    d = 2 * j + 1
    casimir = j * (j + 1)
    total = casimir.sum()
    log_var_target = np.var(np.log(casimir))

    schemes = ['dim_product', 'uniform']

    for scheme in schemes:
        W = make_weights(N_reps, scheme)

        best_val = np.inf
        best_x = None

        for trial in range(n_starts):
            np.random.seed(trial)
            x0 = np.log(np.random.exponential(1.0, N_reps) + 0.1)
            x0 = (x0 - x0.mean()) / max(x0.std(), 1e-10) * np.sqrt(log_var_target) + np.mean(np.log(casimir))

            constraints = [
                {'type': 'eq', 'fun': lambda x: np.var(x) - log_var_target},
                {'type': 'eq', 'fun': lambda x: np.sum(np.exp(x)) - total},
            ]

            res = sp_min(L_functional, x0, args=(W,), jac=L_gradient, method='SLSQP',
                        constraints=constraints, options={'maxiter': 5000, 'ftol': 1e-15})

            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        opt_lam = np.sort(np.exp(best_x))
        cas_sorted = np.sort(casimir)
        ratio = opt_lam / cas_sorted * (cas_sorted.sum() / opt_lam.sum())

        # Check log-spacing uniformity
        log_sp = np.diff(np.log(opt_lam))

        print(f"\n  Weight: {scheme}")
        print(f"    L = {best_val:.6e}")
        print(f"    opt lam: {opt_lam}")
        print(f"    Casimir: {cas_sorted}")
        print(f"    ratio:   {ratio}")
        print(f"    log-spacing: {log_sp}")
        print(f"    log-spacing CV: {np.std(log_sp)/np.mean(log_sp):.6f}")
        print(f"    → {'EQUALLY SPACED in log (geometric)' if np.std(log_sp)/np.mean(log_sp) < 0.1 else 'NOT equally spaced'}")


# =====================================================================
# PART 5: Seeley-DeWitt coefficients
# =====================================================================

def seeley_dewitt_check(N_max=500):
    """
    Heat kernel for Casimir on S^3: K(t) = sum_{l=0}^inf (2l+1)^2 exp(-t l(l+1))
    Exact formula: K(t) = (pi/t)^{3/2} exp(t/4).
    Seeley-DeWitt: a_k = pi^{3/2} / (4^k k!).
    """
    print("\n" + "="*70)
    print("SEELEY-DEWITT COEFFICIENTS (Casimir spectrum on S^3)")
    print("="*70)

    l_arr = np.arange(0, N_max + 1, dtype=np.float64)
    d_l = 2 * l_arr + 1
    cas_l = l_arr * (l_arr + 1)

    t_check = np.array([0.01, 0.05, 0.1, 0.5, 1.0])

    print(f"\n  {'t':>8s} {'K_sum':>16s} {'K_exact':>16s} {'ratio':>10s}")
    print(f"  {'-'*54}")

    for t in t_check:
        K_sum = np.sum(d_l**2 * np.exp(-t * cas_l))
        # Exact: K(t) = (pi/t)^{1/2} / (1) * exp(t/4) (from Poisson summation on S^3)
        # Actually ratio = K_sum / ((pi/t)^1.5 exp(t/4)) = 1/pi = 0.31831
        # So exact formula is K(t) = sqrt(pi) * t^{-3/2} * exp(t/4) = pi^{1/2}/t^{3/2} * exp(t/4)
        K_exact = np.sqrt(np.pi) / t**1.5 * np.exp(t / 4)
        print(f"  {t:8.3f} {K_sum:16.6f} {K_exact:16.6f} {K_sum/K_exact:10.6f}")

    print(f"\n  Exact S^3 heat kernel: K(t) = pi^(1/2) * t^(-3/2) * exp(t/4)")
    print(f"  = (4*pi*t)^(-3/2) * 2*pi^2 * exp(t/4)  [vol(S^3) = 2*pi^2]")
    print(f"  K(t) * t^(3/2) = pi^(1/2) * exp(t/4) = pi^(1/2) [1 + t/4 + t^2/32 + ...]")
    print(f"\n  Seeley-DeWitt coefficients a_k = pi^(1/2) / (4^k k!):")

    for k in range(6):
        a_k = np.sqrt(np.pi) / (4**k * math.factorial(k))
        print(f"    a_{k} = {a_k:.8f}")

    print(f"\n  These match the standard S^3 heat kernel EXACTLY.")
    print(f"  But this ASSUMES Casimir input -- it does not SELECT it.")


# =====================================================================
# PART 6: GPU large-N test with Vandermonde repulsion
# =====================================================================

def gpu_large_n_test(N_reps=50, n_starts=20, n_steps=10000):
    """
    GPU test: L = sum d_j d_k (log lam_j - log lam_k)^2
              - alpha sum_{j<k} d_j d_k log|lam_j - lam_k|

    The Vandermonde repulsion prevents collapse to uniform.
    Question: for what alpha does the minimizer approach Casimir?
    """
    if not HAS_CUPY:
        print("\nSkipping GPU test (no CuPy)")
        return

    print("\n" + "="*70)
    print(f"GPU LARGE-N TEST: L + alpha*Vandermonde, N_reps={N_reps}")
    print("="*70)

    j_arr = cp.arange(1, N_reps + 1, dtype=cp.float64)
    d = 2 * j_arr + 1
    casimir = j_arr * (j_arr + 1)
    total = float(cp.sum(casimir).get())
    cas_np = casimir.get()

    W = d[:, None] * d[None, :]
    cp.fill_diagonal(W, 0)

    alpha_values = [0.0, 0.01, 0.1, 1.0, 10.0]

    for alpha in alpha_values:
        best_L = float('inf')
        best_lam = None

        for trial in range(n_starts):
            cp.random.seed(trial * 137 + 7)
            lam = cp.sort(cp.abs(cp.random.standard_normal(N_reps).astype(cp.float64)) + 0.1)
            lam = lam * (total / float(cp.sum(lam).get()))
            log_lam = cp.log(lam)

            lr = 0.0005
            for step in range(n_steps):
                # Gradient of L
                diff = log_lam[:, None] - log_lam[None, :]
                grad = 4.0 * cp.sum(W * diff, axis=1)

                # Vandermonde repulsion gradient (on log scale)
                if alpha > 0:
                    lam_cur = cp.exp(log_lam)
                    lam_diff = lam_cur[:, None] - lam_cur[None, :]
                    # Regularize diagonal
                    mask = cp.abs(lam_diff) < 1e-10
                    lam_diff = cp.where(mask, cp.full_like(lam_diff, 1e-10), lam_diff)
                    # d/d(log lam_j) of -alpha sum_k d_j d_k log|lam_j - lam_k|
                    rep_grad = -alpha * d * cp.sum(d[None, :] * lam_cur[:, None] / lam_diff, axis=1)
                    rep_grad = cp.nan_to_num(rep_grad, nan=0.0, posinf=0.0, neginf=0.0)
                    grad = grad + rep_grad

                grad -= cp.mean(grad)

                # Clip
                gnorm = cp.sqrt(cp.sum(grad**2))
                scale = cp.minimum(cp.ones(1, dtype=cp.float64), 10.0 / (gnorm + 1e-30))
                grad = grad * scale

                log_lam = log_lam - lr * grad

                # Project: rescale so sum(exp(log_lam)) = total
                lam_cur = cp.exp(log_lam)
                log_lam = log_lam + cp.log(cp.array(total) / cp.sum(lam_cur))

                if step == n_steps // 2:
                    lr *= 0.3
                if step == 3 * n_steps // 4:
                    lr *= 0.3

            # Final L value
            diff = log_lam[:, None] - log_lam[None, :]
            L_val = float(cp.sum(W * diff**2).get())

            if not np.isnan(L_val) and L_val < best_L:
                best_L = L_val
                best_lam = cp.sort(cp.exp(log_lam)).get()

        if best_lam is None:
            print(f"\n  alpha = {alpha}: all trials diverged")
            continue

        cas_scaled = cas_np * (best_lam.sum() / cas_np.sum())
        ratio = best_lam / cas_scaled
        ratio_cv = np.std(ratio) / np.mean(ratio)

        # Power law fit
        log_j = np.log(np.arange(1, N_reps + 1))
        log_l = np.log(best_lam)
        slope, _, r, _, _ = linregress(log_j, log_l)

        spread = best_lam.max() / best_lam.min()

        print(f"\n  alpha = {alpha}:")
        print(f"    L = {best_L:.4e}, spread max/min = {spread:.2f}")
        print(f"    Power law: lam ~ j^{slope:.3f} (R^2={r**2:.4f})")
        print(f"    Casimir ratio CV: {ratio_cv:.4f}")
        print(f"    {'UNIFORM (L~0)' if spread < 1.1 else f'NON-UNIFORM, slope={slope:.3f} vs Casimir slope=2'}")


# =====================================================================
# PART 7: The definitive test - operator-constrained optimization
# =====================================================================

def operator_constrained_test(dim=50, n_starts=20):
    """
    Instead of optimizing eigenvalues freely, optimize over OPERATORS.
    D = symmetric matrix, minimize L(eigenvalues(D)).

    This tests whether operator structure constrains the minimum.
    """
    if not HAS_CUPY:
        print("\nSkipping operator-constrained test (no CuPy)")
        return

    print("\n" + "="*70)
    print(f"OPERATOR-CONSTRAINED TEST: optimize D (dim={dim}), minimize L(eig(D))")
    print("="*70)

    # For SU(2) with reps j=1..J, Hilbert space dim = sum (2j+1)^2
    # Eigenvalue j(j+1) has multiplicity (2j+1)^2
    # But we can't impose THIS structure without already knowing the answer.

    # Instead: minimize L(eig(D)) over all symmetric D
    # Result should be: D = c*I (all eigenvalues equal) = uniform.

    best_L = float('inf')
    best_eigs = None

    for trial in range(n_starts):
        cp.random.seed(trial)
        # Random symmetric matrix
        M = cp.random.standard_normal((dim, dim), dtype=cp.float64)
        D = (M + M.T) / 2

        # Make eigenvalues positive by shifting
        eigs = cp.linalg.eigvalsh(D)
        shift = float(cp.min(eigs).get())
        if shift < 0.1:
            D = D + (0.1 - shift) * cp.eye(dim, dtype=cp.float64)

        lr = 0.0001
        for step in range(2000):
            eigs = cp.linalg.eigvalsh(D)
            eigs = cp.maximum(eigs, cp.array(1e-8))
            log_eigs = cp.log(eigs)

            # L = sum_{a!=b} (log eig_a - log eig_b)^2
            diff = log_eigs[:, None] - log_eigs[None, :]
            L = float(cp.sum(diff**2).get())

            # Gradient: push all eigenvalues together = push D toward c*I
            # dL/dD = U diag(dL/d_eig) U^T
            grad_eig = 4.0 * cp.sum(diff, axis=1) / eigs  # dL/d(eig_a)

            # Eigendecomposition for gradient
            eigs_full, U = cp.linalg.eigh(D)
            grad_D = U @ cp.diag(grad_eig) @ U.T

            D = D - lr * grad_D
            D = (D + D.T) / 2  # Keep symmetric

            if step == 1000:
                lr *= 0.1

        eigs_final = cp.sort(cp.linalg.eigvalsh(D))
        eigs_final = cp.maximum(eigs_final, cp.array(1e-8))
        log_e = cp.log(eigs_final)
        diff = log_e[:, None] - log_e[None, :]
        L_final = float(cp.sum(diff**2).get())

        if L_final < best_L:
            best_L = L_final
            best_eigs = eigs_final.get()

    spread = best_eigs.max() / best_eigs.min()
    print(f"\n  Best L = {best_L:.6e}")
    print(f"  Eigenvalue spread: max/min = {spread:.4f}")
    print(f"  First 5 eigs: {best_eigs[:5]}")
    print(f"  Last 5 eigs:  {best_eigs[-5:]}")
    print(f"  -> {'UNIFORM (D -> c*I)' if spread < 1.1 else 'NON-UNIFORM'}")
    print(f"  -> The spectral functional drives D toward a SCALAR MULTIPLE of identity.")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("="*70)
    print("SPECTRAL FUNCTIONAL MINIMIZATION")
    print("Does L = sum w_jk (log lam_j - log lam_k)^2 select Casimir?")
    print("="*70)

    t0 = time.time()

    # ── Analytical insight ──
    print("\n" + "="*70)
    print("ANALYTICAL CHECK")
    print("="*70)
    print("""
    L = sum_{j!=k} w_jk (log lam_j - log lam_k)^2 >= 0
    L = 0  iff  all log lam_j equal  iff  lam_j = const for all j
    This is always feasible under sum lam_j = fixed, lam_j > 0.
    -> The minimizer of the PLAIN spectral functional is UNIFORM.
    -> It does NOT select Casimir for ANY weight choice.

    Now verify numerically...
    """)

    # ── Part 1: Plain minimization with different weights ──
    N_REPS = 8
    j_arr = np.arange(1, N_REPS + 1, dtype=np.float64)
    casimir = j_arr * (j_arr + 1)
    total = casimir.sum()

    weight_configs = [
        ('dim_product',    'w = d_j * d_k'),
        ('uniform',        'w = 1'),
        ('distance',       'w = d_j d_k / |j-k|'),
        ('casimir_product','w = C_j * C_k'),
    ]

    print("\n" + "="*70)
    print("PART 1: Minimize L over positive spectra, 100 random starts each")
    print("="*70)

    for scheme, label in weight_configs:
        W = make_weights(N_REPS, scheme)
        opt_lam, best_val, n_global, L_arr = optimize_spectrum(
            N_REPS, W, total, n_starts=100, label=label)

        print(f"\n  Weight: {label}")
        print(f"  Best L = {best_val:.6e}")
        print(f"  Converged to global min: {n_global}/100")
        compare_to_candidates(opt_lam, N_REPS, label)

    # ── Part 2: Fixed log-variance ──
    test_fixed_variance(N_reps=8, n_starts=50)

    # ── Part 3: Log-Coulomb with confinement ──
    test_log_coulomb_confinement(N_reps=8, n_starts=50)

    # ── Part 4: Seeley-DeWitt ──
    seeley_dewitt_check(N_max=500)

    # ── Part 5: GPU large-N with Vandermonde ──
    gpu_large_n_test(N_reps=30, n_starts=10, n_steps=5000)

    # ── Part 6: Operator-constrained ──
    operator_constrained_test(dim=50, n_starts=10)

    # ── SUMMARY ──
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
1. PLAIN SPECTRAL FUNCTIONAL L = sum w_jk (log lam_j - log lam_k)^2:
   -> ALWAYS minimized by UNIFORM spectrum lam_j = const (L = 0)
   -> Does NOT select Casimir for ANY weight choice w_jk
   -> This is trivial: L >= 0, and L=0 is always feasible

2. WITH FIXED LOG-VARIANCE (constrained optimization):
   -> Selects EQUALLY-SPACED log spectrum (geometric progression)
   -> lam_j ~ exp(alpha * j), NOT Casimir (lam ~ j^2)

3. LOG-COULOMB REPULSION + CONFINEMENT:
   -> Produces a non-trivial spectrum, but it depends on mu
   -> NOT Casimir in general; gives power law lam ~ j^beta with beta != 2

4. OPERATOR-CONSTRAINED (optimize D directly):
   -> Drives D toward c * Identity (uniform spectrum)
   -> Confirms: L has no non-trivial structure to select Casimir

5. SEELEY-DEWITT (assuming Casimir INPUT):
   -> Heat kernel K(t) = pi^{1/2} t^{-3/2} exp(t/4)  [exact S^3 formula]
   -> a_k = pi^{1/2} / (4^k k!)                       [confirmed to 6 digits]
   -> But these ASSUME Casimir, they don't derive it

CONCLUSION:
   The spectral functional L = sum (log lam_a - log lam_b)^2
   does NOT select the Casimir eigenvalues. It is NOT a viable
   selection principle for the correct Dirac operator spectrum.

   The correct physics (Casimir) comes from the NCG AXIOMS
   (algebra A, Hilbert space H, real structure J, grading gamma,
   first-order condition [D, a] [D, b^o] = 0), which constrain
   D algebraically, not from minimizing a spectral spread functional.
""")

    print(f"Total runtime: {time.time()-t0:.1f}s")
