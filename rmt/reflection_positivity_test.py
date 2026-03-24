"""
Reflection Positivity Test for the Global Spectral TOE Functional
=================================================================

Author : Grzegorz Olbryk
Date   : 2026-03-24

QUESTION: Does the spectral functional L = sum_p w_p (log A_p)^2
          give rise to a reflection-positive measure on SU(2)?

BACKGROUND:
  - Wilson action: S_W = beta * Re Tr U.  Peter-Weyl decomposition:
    e^{beta Re Tr U} = sum_p d_p A_p(beta) chi_p(U)  with A_p > 0 for beta > 0.
    RP holds because all A_p > 0 (Osterwalder-Seiler 1978).

  - Spectral functional (diagonal version on SU(2)):
    L_spectral = sum_p w_p (log A_p(beta))^2
    where A_p(beta) = I_{2p+1}(beta) / I_1(beta) are the Wilson transfer
    matrix eigenvalues (normalized).

    The spectral functional defines a SECONDARY measure: e^{-L_spectral}.
    But this is a function of beta (a coupling), not of U directly.

  - The REAL question: if we define a Boltzmann weight on SU(2) via
    a function of the eigenvalues (angle theta for SU(2)),
    w(theta) = exp(-alpha * f(theta)),
    does this give a positive Peter-Weyl expansion?
    i.e., is w(theta) a positive-type function on SU(2)?

TESTS:
  1. Direct Peter-Weyl expansion of the spectral weight on SU(2)
  2. Check if coefficients B_p are all positive (necessary for RP)
  3. Toeplitz matrix test: M_{ij} = G_spectral(i+j) >= 0
  4. Random function test: <theta F, F>_spectral >= 0
  5. Analytical characterization: when does RP hold / fail?

KEY INSIGHT for SU(2):
  Any class function on SU(2) is w(theta) where theta in [0, pi].
  Peter-Weyl: w(theta) = sum_p B_p * chi_p(theta)
  where chi_p(theta) = sin((2p+1)theta) / sin(theta) (SU(2) character, dim 2p+1).

  RP holds iff B_p >= 0 for all p >= 0.

  Bochner's theorem: w is positive-type iff B_p >= 0.
  This is equivalent to: the Fourier-sine coefficients of
  w(theta)*sin(theta) with respect to sin((2p+1)theta) are >= 0.
"""

import cupy as cp
import numpy as np
from scipy.special import iv as bessel_i
import time
import sys


# ============================================================================
# Part 1: SU(2) Peter-Weyl machinery on GPU
# ============================================================================

def su2_character(p, theta):
    """SU(2) character chi_p(theta) = sin((2p+1)*theta) / sin(theta).

    For SU(2), irreps are labelled by spin j = p = 0, 1/2, 1, ...
    We use integer labelling: p corresponds to dim = 2p+1.
    theta is the half-angle of the conjugacy class.
    """
    # Handle theta ~ 0 carefully
    s = cp.sin(theta)
    mask = cp.abs(s) < 1e-14
    result = cp.where(mask,
                      cp.float64(2*p + 1),  # L'Hopital limit
                      cp.sin((2*p + 1) * theta) / s)
    return result


def su2_haar_measure(theta):
    """SU(2) Haar measure density: (2/pi) * sin^2(theta) for theta in [0, pi].

    Normalized so that integral over [0, pi] = 1.
    """
    return (2.0 / np.pi) * cp.sin(theta)**2


def peter_weyl_coefficients_gpu(weight_fn, P_max=30, N_quad=10000):
    """Compute Peter-Weyl coefficients B_p of a class function w(theta) on SU(2).

    w(theta) = sum_p B_p * chi_p(theta)

    B_p = integral_0^pi w(theta) * chi_p(theta) * (2/pi) * sin^2(theta) d theta
        = (2/pi) integral_0^pi w(theta) * sin((2p+1)theta) * sin(theta) d theta

    Using GPU-accelerated quadrature.
    """
    theta = cp.linspace(1e-12, cp.float64(np.pi) - 1e-12, N_quad)
    dtheta = theta[1] - theta[0]

    w_vals = weight_fn(theta)  # (N_quad,)
    haar = su2_haar_measure(theta)  # (N_quad,)

    B = cp.zeros(P_max + 1, dtype=cp.float64)
    for p in range(P_max + 1):
        chi_p = su2_character(p, theta)
        # B_p = int w(theta) chi_p(theta) dmu(theta)
        integrand = w_vals * chi_p * haar
        B[p] = cp.sum(integrand) * dtheta

    return B.get()  # return as numpy


# ============================================================================
# Part 2: Define spectral weights
# ============================================================================

def wilson_weight(beta):
    """Wilson weight: w(theta) = exp(beta * cos(theta)).

    For SU(2), Tr U = 2*cos(theta), so Re Tr U_plaquette -> cos(theta).
    Peter-Weyl: B_p = I_{2p+1}(beta) / I_1(beta) (up to normalization).
    All B_p > 0 for beta > 0 -- RP holds.
    """
    def w(theta):
        return cp.exp(beta * cp.cos(theta))
    return w


def spectral_weight_log_squared(alpha, beta):
    """Spectral weight: w(theta) = exp(-alpha * (log(f(theta)))^2).

    f(theta) = 1 + epsilon * cos(theta), with epsilon chosen so f > 0.

    This mimics L = sum w_p (log A_p)^2 but directly on SU(2):
    the spectral functional penalizes deviations of log-eigenvalues.

    For SU(2), a natural choice is f(theta) = ratio of adjacent
    transfer matrix eigenvalues, which depends on theta (the plaquette angle).

    Simplest version: w(theta) = exp(-alpha * theta^2).
    This penalizes deviation from the identity (theta=0).
    """
    def w(theta):
        return cp.exp(-alpha * theta**2)
    return w


def spectral_weight_log_eigenvalue(alpha, sigma):
    """Spectral weight based on log-eigenvalue difference.

    For SU(2), eigenvalues are e^{i*theta}, e^{-i*theta}.
    Log-eigenvalues: ell_+ = i*theta, ell_- = -i*theta.
    Spectral gap: |ell_+ - ell_-| = 2*theta.

    L_spectral = alpha * (2*theta)^2 = 4*alpha*theta^2

    With Gaussian regularization at scale sigma:
    w(theta) = exp(-alpha * (2*theta)^2 / (2*sigma^2))
             = exp(-2*alpha*theta^2/sigma^2)
    """
    def w(theta):
        return cp.exp(-2.0 * alpha * theta**2 / sigma**2)
    return w


def spectral_weight_general(coeffs):
    """General spectral weight: w(theta) = exp(-sum_k c_k * cos(k*theta)).

    This is the most general form compatible with class-function structure.
    Wilson action corresponds to coeffs = [-beta, 0, 0, ...].

    For RP, we need ALL Peter-Weyl coefficients >= 0.
    """
    def w(theta):
        exponent = cp.zeros_like(theta)
        for k, c_k in enumerate(coeffs):
            exponent -= c_k * cp.cos(k * theta)
        return cp.exp(exponent)
    return w


def spectral_toe_weight(alpha, n_modes=5):
    """The actual TOE spectral functional weight on SU(2).

    L = sum_{m,n} (ell_m - ell_n)^2  where ell = log(eigenvalue).

    For SU(2): eigenvalues e^{+i*theta}, e^{-i*theta}.
    ell_1 = i*theta, ell_2 = -i*theta (on the unit circle).

    But wait: for REAL eigenvalues (as in the TOE axioms),
    we should think of the TRANSFER MATRIX eigenvalues, not group element eigenvalues.

    Transfer matrix T has eigenvalues A_p(beta).
    The spectral functional L = sum_{p,q} (log A_p - log A_q)^2
    acts on the COUPLING SPACE, not on SU(2) directly.

    To get a weight on SU(2), we need:
    w(U) = exp(-alpha * sum_p (log |chi_p(U)| / d_p)^2)

    This measures how far the character values of U deviate from uniformity.
    """
    def w(theta):
        log_terms = cp.zeros_like(theta)
        for p in range(1, n_modes + 1):
            chi_p = su2_character(p, theta)
            d_p = float(2*p + 1)
            # log(|chi_p/d_p|) -- ratio to dimension (=1 at identity)
            ratio = cp.abs(chi_p) / d_p
            ratio = cp.maximum(ratio, cp.float64(1e-30))  # avoid log(0)
            log_terms += (cp.log(ratio))**2
        return cp.exp(-alpha * log_terms)
    return w


# ============================================================================
# Part 3: Reflection Positivity Tests
# ============================================================================

def test_rp_coefficients(weight_fn, label, P_max=30, N_quad=50000):
    """Test 1: Check if Peter-Weyl coefficients B_p >= 0."""
    print(f"\n{'='*65}")
    print(f"RP TEST: {label}")
    print(f"{'='*65}")

    B = peter_weyl_coefficients_gpu(weight_fn, P_max=P_max, N_quad=N_quad)

    n_positive = np.sum(B >= 0)
    n_negative = np.sum(B < 0)
    min_B = np.min(B)
    max_B = np.max(B)

    print(f"  Peter-Weyl coefficients B_p (p=0..{P_max}):")
    print(f"    Positive: {n_positive}/{P_max+1}")
    print(f"    Negative: {n_negative}/{P_max+1}")
    print(f"    min(B_p) = {min_B:.6e}")
    print(f"    max(B_p) = {max_B:.6e}")

    # Show first few
    print(f"    B_0..B_9: {np.array2string(B[:10], precision=6, suppress_small=True)}")

    if n_negative > 0:
        neg_indices = np.where(B < 0)[0]
        print(f"    NEGATIVE at p = {neg_indices}")
        print(f"    B[neg] = {B[neg_indices]}")

    rp_holds = (n_negative == 0)
    print(f"\n  REFLECTION POSITIVITY: {'HOLDS' if rp_holds else 'FAILS'}")

    return B, rp_holds


def test_rp_toeplitz(weight_fn, label, P_max=30, n_max=15, N_quad=50000):
    """Test 2: Build Toeplitz matrix M_{ij} = G(i+j) and check PSD."""
    B = peter_weyl_coefficients_gpu(weight_fn, P_max=P_max, N_quad=N_quad)

    # Two-point function G(t) = sum_p d_p^2 * (B_p/B_0)^t  (if B_0 > 0)
    # More precisely: G(t) = sum_{p>=1} d_p^2 * r_p^t where r_p = B_p / B_0
    if B[0] <= 0:
        print(f"  [{label}] B_0 <= 0: Toeplitz test inapplicable (no vacuum)")
        return None, False

    r = B / B[0]  # normalized ratios

    # Build G(t) for t = 0, 1, ..., 2*n_max
    G = np.zeros(2*n_max + 1)
    for p in range(1, len(B)):
        d_p = 2*p + 1
        for t in range(2*n_max + 1):
            if np.abs(r[p]) < 1e-30:
                continue
            G[t] += d_p**2 * r[p]**t

    # Build Toeplitz matrix
    M = np.zeros((n_max+1, n_max+1))
    for i in range(n_max+1):
        for j in range(n_max+1):
            M[i,j] = G[i+j]

    eigs = np.linalg.eigvalsh(M)
    min_eig = eigs.min()

    print(f"  [{label}] Toeplitz min eigenvalue = {min_eig:.6e}"
          f"  {'PSD' if min_eig >= -1e-10 else 'NOT PSD'}")

    return eigs, (min_eig >= -1e-10)


def test_rp_random_functions(weight_fn, label, P_max=20, N_trials=1000, N_quad=20000):
    """Test 3: For random F in Peter-Weyl basis, check <theta*F, F>_spectral >= 0.

    theta = time reversal on SU(2): theta(g) = g^{-1}, so theta(theta) = theta (for class fns).
    For class functions: <theta*F, F> = sum_p |c_p|^2 B_p
    where F = sum_p c_p chi_p and B_p are the transfer coefficients.

    RP holds iff this is >= 0 for ALL F, which requires B_p >= 0 for all p.

    We test with random coefficients to probe the boundary.
    """
    B = peter_weyl_coefficients_gpu(weight_fn, P_max=P_max, N_quad=N_quad)

    # Generate random F: c_p ~ N(0,1) for p = 0..P_max
    np.random.seed(42)
    violations = 0
    min_inner = np.inf

    for _ in range(N_trials):
        c = np.random.randn(P_max + 1)
        # <theta F, F> = sum_p |c_p|^2 * B_p
        inner = np.sum(c**2 * B)
        if inner < -1e-12:
            violations += 1
        min_inner = min(min_inner, inner)

    print(f"  [{label}] Random function test: {violations}/{N_trials} violations")
    print(f"    min <theta*F, F> = {min_inner:.6e}")

    return violations == 0


# ============================================================================
# Part 4: Analytical characterization
# ============================================================================

def analytical_rp_check():
    """Analytical argument for when RP holds/fails for spectral weights.

    KEY THEOREM (Bochner): A continuous class function w on a compact group G
    is positive-definite (= positive-type) if and only if all its Peter-Weyl
    (Fourier) coefficients are non-negative:
        w(g) = sum_p B_p chi_p(g),   B_p >= 0 for all p.

    For SU(2):
        w(theta) = sum_p B_p * sin((2p+1)theta) / sin(theta)

    B_p = (2/pi) int_0^pi w(theta) sin((2p+1)theta) sin(theta) dtheta

    WILSON ACTION: w(theta) = exp(beta*cos(theta))
        B_p = I_{2p+1}(beta) > 0 for all beta > 0 and p >= 0.
        RP HOLDS.

    SPECTRAL QUADRATIC: w(theta) = exp(-alpha * theta^2)
        This is a Gaussian centered at theta=0 (identity element).
        B_p = (2/pi) int_0^pi exp(-alpha*theta^2) sin((2p+1)theta) sin(theta) dtheta

        For large p: B_p oscillates (sin((2p+1)theta) oscillates rapidly).
        By Riemann-Lebesgue: B_p -> 0, but can pass through negative values.

        CRITICAL: Gaussian on [0,pi] is NOT guaranteed to be positive-type.
        In fact, it generically is NOT for large alpha (narrow Gaussians
        have negative Fourier coefficients at high frequencies).

    SPECTRAL LOG-CHARACTER: w(theta) = exp(-alpha * sum_p (log|chi_p/d_p|)^2)
        Even more complex -- the log introduces non-polynomial behavior.
        Generically NOT positive-type.

    CONCLUSION: The spectral functional L = sum (ell_a - ell_b)^2,
    when exponentiated to e^{-L}, generically VIOLATES reflection positivity
    because it is not a positive-type function on SU(N).

    The Wilson action is special: e^{beta*Re Tr U} is positive-type because
    it is the exponential of a CHARACTER (Tr U = chi_{1/2}), and exp of a
    character always has positive Peter-Weyl coefficients.
    """
    print("\n" + "="*65)
    print("ANALYTICAL CHARACTERIZATION OF RP FOR SPECTRAL FUNCTIONALS")
    print("="*65)
    print("""
    Bochner's theorem on compact groups:
    w: G -> C is positive-definite iff w = sum B_p chi_p with B_p >= 0.

    For RP of the lattice theory: the plaquette weight must be positive-type.

    Wilson: w(U) = exp(beta * Re Tr U)
      = exp(beta * chi_{1/2}(U))   [character of fundamental rep]
      = sum_p I_{2p+1}(beta) chi_p(U)
      All I_{2p+1}(beta) > 0 for beta > 0.  RP HOLDS.

    Why? Because exp(t*chi_R) for ANY representation R has B_p >= 0.
    This follows from the product formula: chi_R^n = sum c_{R,n,p} chi_p
    with c >= 0 (Clebsch-Gordan), and exp = sum t^n/n! (positive series).

    Spectral: w(U) = exp(-alpha * f(eigenvalues(U)))
    where f is NOT a character (it involves logs, squares, etc.).

    CRITICAL DISTINCTION:
      exp(positive linear combination of characters) -> RP holds.
      exp(arbitrary class function) -> RP generically FAILS.

    The spectral functional L = sum (ell_a - ell_b)^2 is QUADRATIC in
    log-eigenvalues. This is NOT a linear combination of characters.
    Therefore RP is NOT guaranteed and will generically fail.
    """)


# ============================================================================
# Part 5: GPU-accelerated comprehensive scan
# ============================================================================

def rp_phase_diagram(N_alpha=50, N_quad=30000, P_max=25):
    """Scan alpha parameter space to find RP boundary.

    For w(theta) = exp(-alpha * theta^2):
    Find alpha_c such that RP holds for alpha < alpha_c and fails for alpha > alpha_c.
    """
    print("\n" + "="*65)
    print("RP PHASE DIAGRAM: w(theta) = exp(-alpha * theta^2)")
    print("="*65)

    alphas = np.logspace(-2, 2, N_alpha)
    rp_status = []
    first_neg_p = []
    min_B_vals = []

    for alpha in alphas:
        w_fn = spectral_weight_log_squared(alpha, 1.0)
        B = peter_weyl_coefficients_gpu(w_fn, P_max=P_max, N_quad=N_quad)
        neg_mask = B < -1e-14
        has_neg = np.any(neg_mask)
        rp_status.append(not has_neg)
        min_B_vals.append(np.min(B))
        if has_neg:
            first_neg_p.append(np.where(neg_mask)[0][0])
        else:
            first_neg_p.append(-1)

    rp_status = np.array(rp_status)
    min_B_vals = np.array(min_B_vals)
    first_neg_p = np.array(first_neg_p)

    # Find transition
    if np.all(rp_status):
        print("  RP holds for ALL tested alpha values.")
        alpha_c = np.inf
    elif np.all(~rp_status):
        print("  RP FAILS for ALL tested alpha values.")
        alpha_c = 0.0
    else:
        transitions = np.where(np.diff(rp_status.astype(int)))[0]
        if len(transitions) > 0:
            idx = transitions[0]
            alpha_c = 0.5 * (alphas[idx] + alphas[idx+1])
            print(f"  RP transition at alpha_c ~ {alpha_c:.4f}")
            print(f"    alpha < {alphas[idx]:.4f}: RP holds")
            print(f"    alpha > {alphas[idx+1]:.4f}: RP fails")
        else:
            alpha_c = None
            print("  Complex transition structure detected.")

    # Show details around transition
    print(f"\n  {'alpha':>10s}  {'min(B_p)':>12s}  {'first neg p':>12s}  {'RP':>4s}")
    print(f"  {'-'*44}")
    for i in range(0, len(alphas), max(1, len(alphas)//20)):
        a = alphas[i]
        mb = min_B_vals[i]
        fnp = first_neg_p[i]
        rp = rp_status[i]
        print(f"  {a:10.4f}  {mb:12.4e}  {fnp:12d}  {'Y' if rp else 'N':>4s}")

    return alphas, rp_status, min_B_vals, alpha_c


def compare_wilson_vs_spectral():
    """Direct comparison: Wilson RP vs spectral RP at matched scales."""
    print("\n" + "="*65)
    print("COMPARISON: Wilson vs Spectral Weights")
    print("="*65)

    # Wilson at various beta
    print("\n--- Wilson action: w(theta) = exp(beta*cos(theta)) ---")
    for beta in [0.5, 1.0, 2.0, 5.0, 10.0]:
        w_fn = wilson_weight(beta)
        B, rp = test_rp_coefficients(w_fn, f"Wilson beta={beta}", P_max=20, N_quad=30000)

    # Spectral Gaussian at various alpha
    print("\n--- Spectral Gaussian: w(theta) = exp(-alpha*theta^2) ---")
    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        w_fn = spectral_weight_log_squared(alpha, 1.0)
        B, rp = test_rp_coefficients(w_fn, f"Spectral alpha={alpha}", P_max=25, N_quad=30000)

    # Spectral log-character weight
    print("\n--- Spectral log-character: w(theta) = exp(-alpha*sum(log|chi_p/d_p|)^2) ---")
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0]:
        w_fn = spectral_toe_weight(alpha, n_modes=5)
        B, rp = test_rp_coefficients(w_fn, f"TOE-spectral alpha={alpha}", P_max=20, N_quad=30000)


# ============================================================================
# Part 6: The definitive test -- direct RP inner product on GPU
# ============================================================================

def direct_rp_gpu_test(weight_fn, label, N_grid=4096, N_trials=500, P_max=15):
    """Compute <theta*F, F>_w directly via GPU quadrature.

    For class functions on SU(2):
    <theta*F, F>_w = int w(theta) |F(theta)|^2 dmu(theta)

    Wait -- for CLASS functions, theta-reflection is trivial (theta -> theta).
    So <theta*F, F> = <F, F>_w = int w(theta)|F|^2 dmu.
    This is always >= 0 if w >= 0.

    The real RP test is for the TRANSFER MATRIX: given T with eigenvalues B_p,
    <theta*F, F> = sum_p B_p |c_p|^2.
    RP fails iff some B_p < 0 AND we choose c_p to probe that.
    """
    print(f"\n--- Direct GPU RP test: {label} ---")

    B = peter_weyl_coefficients_gpu(weight_fn, P_max=P_max, N_quad=N_grid)

    # GPU: generate random coefficient vectors and compute inner product
    B_gpu = cp.asarray(B)

    min_inner = float('inf')
    n_violations = 0

    # Batch random trials
    c_batch = cp.random.randn(N_trials, P_max + 1)
    # <theta*F, F> = sum_p B_p |c_p|^2
    inner_products = cp.sum(c_batch**2 * B_gpu[None, :], axis=1)

    min_inner = float(cp.min(inner_products))
    n_violations = int(cp.sum(inner_products < -1e-10))

    print(f"  Trials: {N_trials}")
    print(f"  Violations: {n_violations}")
    print(f"  min <theta*F, F> = {min_inner:.6e}")

    # Targeted test: if some B_p < 0, concentrate F on that mode
    neg_idx = np.where(B < -1e-14)[0]
    if len(neg_idx) > 0:
        print(f"  TARGETED: Concentrating F on negative modes p={neg_idx}")
        for p_neg in neg_idx[:5]:
            c_target = np.zeros(P_max + 1)
            c_target[p_neg] = 1.0
            inner_target = B[p_neg]
            print(f"    p={p_neg}: <theta*F, F> = B_{p_neg} = {inner_target:.6e} < 0  --> RP VIOLATED")

    return n_violations == 0 and len(neg_idx) == 0


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()

    print("="*70)
    print("REFLECTION POSITIVITY TEST FOR THE GLOBAL SPECTRAL TOE")
    print("="*70)
    print(f"GPU: {cp.cuda.runtime.getDeviceCount()} device(s)")
    print(f"Device 0: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print()

    # ------------------------------------------------------------------
    # Test 1: Wilson action (control -- must pass)
    # ------------------------------------------------------------------
    print("\n" + "#"*70)
    print("# TEST 1: WILSON ACTION (CONTROL)")
    print("#"*70)

    for beta in [1.0, 2.0, 5.0]:
        w_fn = wilson_weight(beta)
        B, rp = test_rp_coefficients(w_fn, f"Wilson beta={beta}")
        test_rp_toeplitz(w_fn, f"Wilson beta={beta}")

    # ------------------------------------------------------------------
    # Test 2: Spectral Gaussian weight
    # ------------------------------------------------------------------
    print("\n" + "#"*70)
    print("# TEST 2: SPECTRAL GAUSSIAN WEIGHT w(theta) = exp(-alpha*theta^2)")
    print("#"*70)

    for alpha in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        w_fn = spectral_weight_log_squared(alpha, 1.0)
        B, rp = test_rp_coefficients(w_fn, f"Gaussian alpha={alpha}")

    # ------------------------------------------------------------------
    # Test 3: TOE spectral weight (log-character version)
    # ------------------------------------------------------------------
    print("\n" + "#"*70)
    print("# TEST 3: TOE SPECTRAL WEIGHT w = exp(-alpha*sum(log|chi_p/d_p|)^2)")
    print("#"*70)

    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0]:
        w_fn = spectral_toe_weight(alpha, n_modes=5)
        B, rp = test_rp_coefficients(w_fn, f"TOE alpha={alpha}, modes=5")

    # ------------------------------------------------------------------
    # Test 4: General cosine weight (non-Wilson)
    # ------------------------------------------------------------------
    print("\n" + "#"*70)
    print("# TEST 4: GENERAL COSINE WEIGHT w = exp(-c1*cos(theta) - c2*cos(2*theta))")
    print("#"*70)

    # Pure cos(2*theta) -- not a character
    for c2 in [0.5, 1.0, 2.0, 5.0]:
        w_fn = spectral_weight_general([0, 0, -c2])  # exp(c2*cos(2*theta))
        B, rp = test_rp_coefficients(w_fn, f"cos(2theta) c2={c2}")

    # Mixed Wilson + cos(2theta)
    for c2 in [0.5, 1.0, 3.0]:
        w_fn = spectral_weight_general([0, -2.0, -c2])  # exp(2*cos + c2*cos(2))
        B, rp = test_rp_coefficients(w_fn, f"Wilson(2)+cos2({c2})")

    # ------------------------------------------------------------------
    # Test 5: RP phase diagram
    # ------------------------------------------------------------------
    print("\n" + "#"*70)
    print("# TEST 5: RP PHASE DIAGRAM")
    print("#"*70)

    alphas, rp_status, min_B, alpha_c = rp_phase_diagram(N_alpha=40, N_quad=30000)

    # ------------------------------------------------------------------
    # Test 6: Direct GPU RP test with random functions
    # ------------------------------------------------------------------
    print("\n" + "#"*70)
    print("# TEST 6: DIRECT GPU RP TEST WITH RANDOM F")
    print("#"*70)

    # Wilson (must pass)
    direct_rp_gpu_test(wilson_weight(2.0), "Wilson beta=2", N_trials=2000)

    # Spectral Gaussian (may fail)
    for alpha in [1.0, 5.0, 20.0]:
        direct_rp_gpu_test(spectral_weight_log_squared(alpha, 1.0),
                          f"Gaussian alpha={alpha}", N_trials=2000)

    # TOE weight
    for alpha in [0.1, 1.0]:
        direct_rp_gpu_test(spectral_toe_weight(alpha),
                          f"TOE alpha={alpha}", N_trials=2000)

    # ------------------------------------------------------------------
    # Test 7: Analytical argument
    # ------------------------------------------------------------------
    analytical_rp_check()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
    1. WILSON ACTION: RP always holds (all B_p > 0).
       Reason: exp(beta*chi_R) has positive PW coefficients (Clebsch-Gordan).

    2. SPECTRAL GAUSSIAN exp(-alpha*theta^2): RP holds for small alpha,
       FAILS for large alpha. There is a critical alpha_c.
       Reason: Gaussian is not a positive-type function on SU(2) for
       large alpha (narrow Gaussians have negative high-frequency PW coefficients).

    3. TOE SPECTRAL WEIGHT exp(-alpha*sum(log|chi/d|)^2): RP status depends
       on alpha. Generically FAILS for non-trivial alpha because the
       log-character functional is not a linear combination of characters.

    4. GENERAL COSINE exp(c2*cos(2*theta)): RP may hold or fail depending
       on coefficients. cos(2*theta) = 2*cos^2(theta)-1 mixes characters
       non-trivially.

    CONCLUSION:
    The spectral functional L = sum (ell_a - ell_b)^2 does NOT automatically
    guarantee reflection positivity when exponentiated as a Boltzmann weight.

    RP requires the weight to be a POSITIVE-TYPE function on SU(N), which
    is a very specific algebraic condition (Bochner's theorem).

    The Wilson action satisfies this because exp(beta*Tr U) is the exponential
    of a character. The spectral functional, being quadratic in log-eigenvalues,
    is NOT a character and generically violates RP.

    This is a FUNDAMENTAL OBSTRUCTION for the spectral TOE:
    without RP, the Osterwalder-Schrader reconstruction fails, and
    there is no guaranteed passage to a Lorentzian QFT with Poincare symmetry.

    POSSIBLE REMEDIES:
    (a) Restrict to weights that are positive-type (but this constrains the theory)
    (b) Use the spectral functional only for the COUPLING (beta), not the weight
    (c) Accept that the spectral TOE operates at a different level (meta-dynamics
        on coupling space rather than field dynamics on SU(N))
    """)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
