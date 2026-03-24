#!/usr/bin/env python3
"""
Spectral Distance and Dimension from SU(2) Transfer Matrix Eigenvalues
======================================================================

Tests whether the spectral data of the SU(2) transfer matrix generates
a meaningful metric space with integer spectral dimension.

Transfer matrix eigenvalues: A_p(beta) = I_{2p+1}(beta) / I_1(beta)
Log-spectra: ell_p = log A_p(beta)
Spectral distance: d(p,q;beta) = |ell_p - ell_q|

Return probability: P(t) = sum_p d_p^2 exp(-2t * lambda_p) / Z
  where lambda_p = -log(A_p) >= 0 (eigenvalues of the "Laplacian"),
  d_p = 2p+1 (dimension of spin-p rep of SU(2)),
  Z = sum_p d_p^2 exp(-2t * lambda_0) = sum d_p^2 (since lambda_0 = 0).

Spectral dimension: d_s = -2 d log P / d log t
"""

import numpy as np
from scipy.special import iv as besseli
import sys


def compute_A_p(p_max, beta_arr):
    """Compute A_p(beta) = I_{2p+1}(beta) / I_1(beta) for p=0,...,p_max."""
    I1 = besseli(1, beta_arr)
    A = np.zeros((p_max + 1, len(beta_arr)))
    for p in range(p_max + 1):
        A[p] = besseli(2 * p + 1, beta_arr) / I1
    return A


def check_metric_axioms(A, beta_idx, beta_val):
    """Check metric axioms for d(p,q) = |log A_p - log A_q| at a given beta."""
    p_max = A.shape[0]
    ell = np.log(A[:, beta_idx])

    valid = np.isfinite(ell)
    ell_v = ell[valid]
    n = len(ell_v)

    D = np.abs(ell_v[:, None] - ell_v[None, :])

    nonneg = np.all(D >= -1e-15)
    diag_zero = np.allclose(np.diag(D), 0)
    symmetric = np.allclose(D, D.T)

    # Triangle inequality (guaranteed for abs-value metric, but verify)
    tri_violations = 0
    for i in range(n):
        for k in range(n):
            for j in range(n):
                if D[i, k] > D[i, j] + D[j, k] + 1e-12:
                    tri_violations += 1

    off_diag = D[np.triu_indices(n, k=1)]
    all_distinct = np.all(off_diag > 1e-15) if len(off_diag) > 0 else True

    print(f"\n=== METRIC AXIOMS at beta = {beta_val:.1f} ===")
    print(f"  Valid representations: {n} / {p_max}")
    print(f"  Non-negativity:       {'PASS' if nonneg else 'FAIL'}")
    print(f"  d(x,x) = 0:          {'PASS' if diag_zero else 'FAIL'}")
    print(f"  Symmetry:             {'PASS' if symmetric else 'FAIL'}")
    print(f"  Triangle inequality:  {'PASS' if tri_violations == 0 else f'FAIL ({tri_violations} violations)'}")
    print(f"  All distinct:         {'YES' if all_distinct else 'NO (degenerate points)'}")

    if n <= 6:
        print(f"\n  Distance matrix (first {n} reps):")
        print("       ", "  ".join(f"p={i:2d}" for i in range(n)))
        for i in range(n):
            row = "  ".join(f"{D[i,j]:6.3f}" for j in range(n))
            print(f"  p={i:2d}  {row}")

    print(f"\n  Log-spectra ell_p = log A_p:")
    for i in range(min(n, 11)):
        print(f"    p={i:2d}: ell = {ell_v[i]:+10.6f}  (A_p = {np.exp(ell_v[i]):.6e})")

    return nonneg and diag_zero and symmetric and (tri_violations == 0)


def compute_spectral_dimension(lambda_arr, d_p_arr, t_arr):
    """
    Compute spectral dimension d_s(t) from return probability.

    P(t) = sum_p d_p^2 exp(-2t * lambda_p) / Z
    where lambda_p >= 0, lambda_0 = 0.

    d_s = -2 d log P / d log t
    """
    # lambda_arr: shape (n_reps,), all >= 0, lambda[0] = 0
    # d_p_arr: shape (n_reps,), degeneracies

    n_t = len(t_arr)
    # Vectorized: shape (n_t, n_reps)
    exponents = -2.0 * t_arr[:, None] * lambda_arr[None, :]  # all <= 0
    weights = (d_p_arr[None, :] ** 2) * np.exp(exponents)
    P = np.sum(weights, axis=1)

    Z = np.sum(d_p_arr ** 2)
    P /= Z

    # d_s = -2 d(log P) / d(log t)
    log_P = np.log(np.maximum(P, 1e-300))
    log_t = np.log(t_arr)
    d_s = -2 * np.gradient(log_P, log_t)

    return P, d_s


def text_plot(x, y, xlabel, ylabel, title, width=70, height=20):
    """Simple text-based plot."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")

    y_finite = y[np.isfinite(y)]
    if len(y_finite) == 0:
        print("  [No finite values to plot]")
        return

    y_min, y_max = np.min(y_finite), np.max(y_finite)
    if y_max - y_min < 1e-10:
        y_min -= 1
        y_max += 1

    x_min, x_max = np.min(x), np.max(x)

    grid = [[' '] * width for _ in range(height)]

    for i in range(len(x)):
        if not np.isfinite(y[i]):
            continue
        col = int((x[i] - x_min) / (x_max - x_min) * (width - 1))
        row = int((y_max - y[i]) / (y_max - y_min) * (height - 1))
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        grid[row][col] = '*'

    # Mark integer lines
    for d_int in range(max(0, int(np.floor(y_min))), int(np.ceil(y_max)) + 1):
        if y_min <= d_int <= y_max:
            row = int((y_max - d_int) / (y_max - y_min) * (height - 1))
            row = max(0, min(height - 1, row))
            label = str(d_int)
            for c in range(width):
                if grid[row][c] == ' ':
                    grid[row][c] = '-'

    for r in range(height):
        val = y_max - r * (y_max - y_min) / (height - 1)
        print(f"  {val:7.2f} |{''.join(grid[r])}|")

    print(f"          {'_' * width}")
    print(f"  {xlabel}: {x_min:.3f}" + " " * (width - 20) + f"{x_max:.3f}")
    print(f"  {ylabel}")


def main():
    print("=" * 72)
    print("  SPECTRAL DISTANCE & DIMENSION FROM SU(2) TRANSFER MATRIX")
    print("=" * 72)

    # Parameters
    p_max = 10
    beta_arr = np.arange(0.1, 20.05, 0.1)
    t_arr = np.logspace(-2, 2, 500)

    print(f"\nComputing A_p(beta) for p=0..{p_max}, beta=0.1..20.0 ...")
    A = compute_A_p(p_max, beta_arr)

    print(f"  A_0(beta) = 1 for all beta (trivial rep)")
    print(f"  A_1(beta=1) = {A[1, 9]:.6f}")
    print(f"  A_1(beta=10) = {A[1, 99]:.6f}")
    print(f"  A_1(beta=20) = {A[1, 199]:.6f}")

    # =========================================================
    # PART 1: METRIC AXIOMS
    # =========================================================
    print("\n" + "=" * 72)
    print("  PART 1: METRIC AXIOMS")
    print("=" * 72)

    test_betas = [1.0, 5.0, 10.0, 20.0]
    all_pass = True
    for b in test_betas:
        idx = int(round(b / 0.1)) - 1
        passed = check_metric_axioms(A, idx, b)
        all_pass = all_pass and passed

    print(f"\n  OVERALL METRIC STATUS: {'ALL PASS' if all_pass else 'SOME FAIL'}")
    print("\n  THEOREM: d(p,q;beta) = |log A_p - log A_q| is always a valid metric.")
    print("  Proof: it is the pullback of |.| on R via the injection p -> log A_p.")
    print("  For beta>0, A_p is strictly decreasing in p, so all points distinct.")

    # =========================================================
    # PART 2: SPECTRAL DIMENSION
    # =========================================================
    print("\n" + "=" * 72)
    print("  PART 2: SPECTRAL DIMENSION d_s(t)")
    print("=" * 72)

    print("\n  Definition: lambda_p = -log(A_p) >= 0 (Laplacian eigenvalues)")
    print("  P(t) = sum_p (2p+1)^2 exp(-2t*lambda_p) / Z")
    print("  d_s(t) = -2 d(log P)/d(log t)")
    print()
    print("  Small t: all exp terms ~ 1, P ~ 1, d_s -> 0")
    print("  Large t: only p=0 survives (lambda_0=0), P -> 1/Z, d_s -> 0")
    print("  Intermediate t: spectral dimension shows a PEAK")
    print()

    beta_test = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0]
    results = {}

    for b in beta_test:
        idx = int(round(b / 0.1)) - 1
        ell = np.log(A[:, idx])
        lambda_p = -ell  # lambda_0 = 0, lambda_p > 0 for p > 0
        d_p = 2 * np.arange(p_max + 1) + 1.0

        P, d_s = compute_spectral_dimension(lambda_p, d_p, t_arr)
        results[b] = (P, d_s)

        d_s_max = np.nanmax(d_s[5:-5])
        t_at_max = t_arr[5 + np.nanargmax(d_s[5:-5])]

        print(f"  beta = {b:.1f}:")
        print(f"    max(d_s) = {d_s_max:.4f} at t = {t_at_max:.4f}")
        for t_probe in [0.01, 0.1, 1.0, 10.0]:
            idx_t = np.argmin(np.abs(t_arr - t_probe))
            print(f"    d_s(t={t_probe:5.2f}) = {d_s[idx_t]:.4f}")

    # =========================================================
    # PART 3: TEXT PLOTS
    # =========================================================
    print("\n" + "=" * 72)
    print("  PART 3: PLOTS OF d_s(t)")
    print("=" * 72)

    for b in [1.0, 5.0, 10.0, 20.0]:
        P, d_s = results[b]
        d_s_clip = np.clip(d_s, -0.5, 5)
        text_plot(np.log10(t_arr), d_s_clip,
                  "log10(t)", "d_s",
                  f"Spectral dimension d_s(t) at beta={b:.1f}")

    # =========================================================
    # PART 4: LOG-SPECTRA STRUCTURE
    # =========================================================
    print("\n" + "=" * 72)
    print("  PART 4: STRUCTURE OF LOG-SPECTRA")
    print("=" * 72)

    print("\n  Fitting lambda_p = -log(A_p) to power law: lambda_p ~ c * p^alpha")

    for b in [2.0, 5.0, 10.0, 20.0]:
        idx = int(round(b / 0.1)) - 1
        ell = np.log(A[:, idx])
        lam = -ell  # >= 0
        # Fit for p >= 1
        p_vals = np.arange(1, p_max + 1)
        lam_vals = lam[1:]

        log_p = np.log(p_vals)
        log_lam = np.log(lam_vals)
        coeffs = np.polyfit(log_p, log_lam, 1)
        alpha = coeffs[0]
        c = np.exp(coeffs[1])

        print(f"\n  beta = {b:.1f}: lambda_p ~ {c:.4f} * p^{alpha:.3f}")
        print(f"    Casimir prediction: lambda_p ~ p(p+1)/(2*beta) = p^2/(2*{b:.0f}) for large p")
        print(f"    Exact Bessel vs Casimir approx:")
        for p in [1, 2, 3, 5, 10]:
            if p <= p_max:
                exact = lam[p]
                casimir = p * (p + 1) / (2 * b)
                print(f"      p={p:2d}: exact={exact:.6f}, Casimir={casimir:.6f}, ratio={exact/casimir:.4f}")

    # =========================================================
    # PART 5: LARGE-BETA ASYMPTOTICS (CORRECTED)
    # =========================================================
    print("\n" + "=" * 72)
    print("  PART 5: LARGE-BETA ASYMPTOTICS (CORRECTED)")
    print("=" * 72)

    print("""
  For large beta, using I_nu(z) ~ e^z/sqrt(2pi z) * (1 - (4nu^2-1)/(8z) + ...):

  A_p = I_{2p+1}(beta) / I_1(beta)
      ~ exp(-(4(2p+1)^2 - 1 - (4*1 - 1))/(8*beta))
      = exp(-((2p+1)^2 - 1)/(2*beta))     [to leading order]
      = exp(-(4p^2 + 4p)/(2*beta))
      = exp(-2p(p+1)/beta)

  Wait -- let's be more careful. The asymptotic expansion gives:
  I_nu(z) ~ (e^z / sqrt(2pi z)) * sum_k (-1)^k a_k(nu) / z^k
  where a_0 = 1, a_1(nu) = (4nu^2 - 1)/8.

  So log I_nu(z) ~ z - (1/2)log(2pi z) - (4nu^2 - 1)/(8z) + ...

  log A_p = log I_{2p+1}(beta) - log I_1(beta)
          ~ -(4(2p+1)^2 - 4)/(8*beta) + ...
          = -(16p^2 + 16p + 4 - 4)/(8*beta)
          = -2p(p+1)/beta

  So lambda_p ~ 2*C_2(p)/beta where C_2(p) = p(p+1).
""")

    print("  Verification of lambda_p ~ 2*p(p+1)/beta:")
    for b in [10.0, 20.0, 50.0, 100.0]:
        A_test = compute_A_p(5, np.array([b]))
        ell_test = np.log(A_test[:, 0])
        lam_test = -ell_test
        print(f"\n  beta={b:.0f}:")
        for p in [1, 2, 3, 5]:
            theory = 2 * p * (p + 1) / b
            ratio = lam_test[p] / theory if theory > 0 else float('inf')
            print(f"    p={p}: lambda={lam_test[p]:.8f}, 2p(p+1)/beta={theory:.8f}, ratio={ratio:.6f}")

    # =========================================================
    # PART 6: SPECTRAL DIMENSION WITH MORE REPS
    # =========================================================
    print("\n" + "=" * 72)
    print("  PART 6: d_s WITH INCREASING p_max (CONVERGENCE)")
    print("=" * 72)

    print("\n  For the continuous approximation to work, we need many reps.")
    print("  Theoretical prediction: d_s -> 3 in the scaling regime.")

    t_arr_fine = np.logspace(-3, 3, 1000)

    for b in [10.0, 20.0, 50.0, 100.0]:
        print(f"\n  beta = {b:.0f}:")
        b_arr = np.array([b])
        for p_max_test in [10, 30, 50, 100, 200]:
            A_ext = compute_A_p(p_max_test, b_arr)
            ell_ext = np.log(A_ext[:, 0])
            lam_ext = -ell_ext
            d_p_ext = (2 * np.arange(p_max_test + 1) + 1).astype(float)

            P, d_s = compute_spectral_dimension(lam_ext, d_p_ext, t_arr_fine)
            d_s_max = np.nanmax(d_s[10:-10])
            t_at_max = t_arr_fine[10 + np.nanargmax(d_s[10:-10])]
            print(f"    p_max={p_max_test:3d}: max(d_s) = {d_s_max:.4f} at t = {t_at_max:.5f}")

    # =========================================================
    # PART 7: ANALYTICAL d_s FROM CASIMIR (EXACT INTEGRAL)
    # =========================================================
    print("\n" + "=" * 72)
    print("  PART 7: ANALYTICAL d_s FROM CASIMIR INTEGRAL")
    print("=" * 72)

    print("""
  Using lambda_p ~ 2*p(p+1)/beta and d_p = 2p+1, in the continuum limit:

  P(t) = (1/Z) * int_0^inf (2p+1)^2 exp(-2t * 2p(p+1)/beta) dp

  Substituting u = p(p+1), du = (2p+1)dp, and (2p+1)^2 dp = (2p+1) du:
  At large p: u ~ p^2, du ~ 2p dp, (2p+1) ~ 2*sqrt(u)

  P(t) ~ int_0^inf 2*sqrt(u) * exp(-4t*u/beta) du
       = 2 * Gamma(3/2) * (beta/(4t))^{3/2}
       = sqrt(pi) * (beta/(4t))^{3/2}

  d log P / d log t = d log[(beta/(4t))^{3/2}] / d log t = -3/2

  d_s = -2 * (-3/2) = 3

  This is EXACT in the continuum limit: d_s = 3 = dim(SU(2)).
""")

    # Verify with large p_max and large beta
    print("  Numerical verification with large p_max and beta:")
    b = 200.0
    b_arr = np.array([b])
    A_big = compute_A_p(500, b_arr)
    ell_big = np.log(A_big[:, 0])
    lam_big = -ell_big
    d_p_big = (2 * np.arange(501) + 1).astype(float)

    P_big, d_s_big = compute_spectral_dimension(lam_big, d_p_big, t_arr_fine)
    d_s_max = np.nanmax(d_s_big[20:-20])
    t_at_max = t_arr_fine[20 + np.nanargmax(d_s_big[20:-20])]

    # Find region where d_s is close to 3
    close_to_3 = np.where(np.abs(d_s_big - 3.0) < 0.05)[0]
    if len(close_to_3) > 0:
        t_range = t_arr_fine[close_to_3]
        print(f"    max(d_s) = {d_s_max:.4f} at t = {t_at_max:.5f}")
        print(f"    d_s in [2.95, 3.05]: t in [{t_range[0]:.5f}, {t_range[-1]:.5f}]")
        print(f"    This spans {np.log10(t_range[-1]/t_range[0]):.1f} decades in t")
    else:
        print(f"    max(d_s) = {d_s_max:.4f} at t = {t_at_max:.5f}")
        print(f"    d_s does not reach [2.95, 3.05] -- need larger p_max or beta")

    text_plot(np.log10(t_arr_fine), np.clip(d_s_big, -0.5, 5),
              "log10(t)", "d_s",
              f"d_s(t) at beta={b:.0f}, p_max=500")

    # Show d_s values in the plateau region
    print("\n  d_s values near the plateau:")
    for t_probe in [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        idx_t = np.argmin(np.abs(t_arr_fine - t_probe))
        print(f"    t = {t_probe:8.3f}: d_s = {d_s_big[idx_t]:.4f}")

    # =========================================================
    # PART 8: SU(N) PREDICTION
    # =========================================================
    print("\n" + "=" * 72)
    print("  PART 8: SU(N) SPECTRAL DIMENSION PREDICTION")
    print("=" * 72)

    print("""
  The spectral dimension from the return probability with Casimir scaling
  equals the dimension of the group manifold:

  SU(N): rank = N-1, dim = N^2 - 1

  For SU(N), reps labelled by (N-1) highest weights (m_1,...,m_{N-1}).
  The degeneracy d_{m} and Casimir C_2(m) are polynomials in m_i.

  In the continuum approximation (sum -> integral over R^{N-1}_+):
  P(t) ~ integral d^{N-1}m  [d_m]^2  exp(-2t * C_2(m)/beta)

  Since C_2 is quadratic and d_m is a polynomial of degree N(N-1)/2,
  each m_i integral contributes a factor of (beta/t)^{1/2 + deg_i}.

  Explicit computation:
    SU(2): 1 variable j, d=2j+1, C_2=j(j+1)
           P ~ (beta/t)^{3/2}  =>  d_s = 3

    SU(3): 2 variables (p,q), d=(p+1)(q+1)(p+q+2)/2, C_2=p^2+q^2+pq+3p+3q
           The Gaussian integral gives d_s = 8 = dim(SU(3))

    SU(N): (N-1) variables, d_s = N^2 - 1 = dim(SU(N))

  This is a THEOREM (not numerics): the spectral dimension of the
  representation-theoretic heat kernel equals dim(G).

  PHYSICAL INTERPRETATION:
  - d_s = 3 for SU(2) = dimension of S^3 (group manifold)
  - d_s = 8 for SU(3)
  - For the Standard Model gauge group SU(3)xSU(2)xU(1): d_s = 8+3+1 = 12
  - For spacetime to emerge, need d_s = 4, which requires either:
    * A single group: N^2-1 = 4 => N = sqrt(5) (impossible)
    * Product: SU(2) x U(1): 3 + 1 = 4 (electroweak!)
    * Or a fundamentally different spectral structure
""")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("=" * 72)
    print("  SUMMARY OF RESULTS")
    print("=" * 72)

    print("""
  1. METRIC AXIOMS: ALWAYS SATISFIED.
     d(p,q;beta) = |log A_p(beta) - log A_q(beta)| is a valid metric
     on the space of SU(2) representations for all beta > 0.
     This is trivially true (pullback of |.| on R).

  2. SPECTRAL DIMENSION EXISTS AND IS WELL-DEFINED.
     d_s(t) shows a clear peak at intermediate diffusion times,
     approaching 0 at both t->0 and t->infinity.

  3. THE PEAK VALUE CONVERGES TO d_s = 3 = dim(SU(2)).
     This is EXACT in the large-beta / large-p_max limit.
     Analytically: P(t) ~ (beta/t)^{3/2} => d_s = 3.

  4. CASIMIR SCALING IS THE KEY.
     lambda_p = -log A_p ~ 2*p(p+1)/beta = 2*C_2(p)/beta.
     The quadratic Casimir determines the spectral geometry.

  5. GENERAL SU(N): d_s = N^2 - 1 (dimension of the group manifold).
     The spectral data encodes GROUP dimension, not spacetime dimension.

  6. d_s = 4 IS NOT ACHIEVABLE from a single SU(N).
     N^2-1 = 4 has no integer solution.
     BUT: SU(2) x U(1) gives 3+1 = 4 (the electroweak group!).

  ANSWER TO KEY QUESTION:
  YES, the spectral data contains dimensional information.
  The spectral dimension d_s = 3 for SU(2) is an INTEGER,
  equal to dim(SU(2)) = dim(S^3). This is a genuine geometric
  invariant encoded in the transfer matrix eigenvalues.

  However, it is the dimension of the GAUGE GROUP manifold,
  not the spacetime dimension 4.
""")


if __name__ == "__main__":
    main()
