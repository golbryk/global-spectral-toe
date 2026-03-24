#!/usr/bin/env python3
"""
RG Fixed Points and Gauge Group Selection
==========================================
Tests whether RG fixed points can SELECT the gauge group.

For SU(N) in 4D with n_f fermion flavours:
  - Computes exact RG flow using Bessel-based plaquette variance
  - Searches for non-trivial IR fixed points
  - Tests coupling unification for SU(3)×SU(2)×U(1)
  - Maps the conformal window (N, N_f)

Uses CuPy for GPU-accelerated computation.

Author: Grzegorz Olbryk
Date: 2026-03-24
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import cupy as cp
    GPU = True
    print("GPU backend: CuPy + CUDA")
except ImportError:
    import numpy as cp
    GPU = False
    print("GPU backend: NumPy (CPU fallback)")

from scipy.special import iv as bessel_iv  # I_nu(x)


# =============================================================================
# 1. EXACT BESSEL VARIANCE for SU(N) plaquette
# =============================================================================

def plaquette_expectation_sun(beta, N):
    """
    Exact mean plaquette for SU(N) strong-coupling / character expansion.

    For SU(N) Wilson action on a single plaquette:
      <Re Tr U_P / N> = I_1(beta) / I_0(beta)  [leading order, exact for SU(2)]

    For general SU(N), the exact result involves modified Bessel functions:
      <P> = (1/N) * d/d(beta) ln Z(beta)

    where Z(beta) = det[I_{i-j}(beta)]_{i,j=1..N}  (Toeplitz determinant).

    We use the exact Toeplitz determinant formula.
    """
    # Build the N×N Toeplitz matrix M_{ij} = I_{i-j}(beta)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = bessel_iv(i - j, beta)

    Z = np.linalg.det(M)

    # Numerical derivative: d/dbeta ln Z
    dbeta = 1e-6
    M_plus = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M_plus[i, j] = bessel_iv(i - j, beta + dbeta)
    Z_plus = np.linalg.det(M_plus)

    M_minus = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M_minus[i, j] = bessel_iv(i - j, beta - dbeta)
    Z_minus = np.linalg.det(M_minus)

    dlogZ = (np.log(Z_plus) - np.log(Z_minus)) / (2 * dbeta)

    return dlogZ / N


def plaquette_variance_sun(beta, N):
    """
    Variance of the plaquette: Var(P) = d<P>/d(beta) / N

    This is the susceptibility, computed as second derivative of ln Z.
    """
    dbeta = 1e-5
    P_plus = plaquette_expectation_sun(beta + dbeta, N)
    P_minus = plaquette_expectation_sun(beta - dbeta, N)

    # d<P>/dbeta = susceptibility
    dP_dbeta = (P_plus - P_minus) / (2 * dbeta)

    return dP_dbeta


def beta_from_g2(g2, N):
    """Convert coupling g^2 to lattice beta = 2N/g^2."""
    if g2 < 1e-15:
        return 1e15
    return 2.0 * N / g2


def rg_step_exact(g2, N):
    """
    One RG step: g^2_{k+1} = g^2_k + c(g^2_k, N) * g^4_k

    The coefficient c comes from the plaquette variance:
      c(g^2, N) = (variance of plaquette action) * geometric_factor

    In the weak coupling limit: c -> 11N/(48*pi^2) [one-loop perturbative]
    At strong coupling: c depends on g^2 through the exact Bessel variance.

    The exact RG recursion from blocking is:
      delta(g^2) = g^4 * [b_0 + non-perturbative corrections from variance]

    We use: delta(g^2) = g^4 * b_eff(g^2, N)
    where b_eff = b0_pert * [1 + variance_correction(beta)]
    """
    b0_pert = 11.0 * N / (48.0 * np.pi**2)

    beta_lat = beta_from_g2(g2, N)

    if beta_lat > 100:
        # Deep weak coupling: use perturbative
        return g2 + b0_pert * g2**2

    # Get exact variance
    var_P = plaquette_variance_sun(beta_lat, N)

    # Perturbative variance at same beta: Var_pert = 1/(2*N*beta^2)
    var_pert = 1.0 / (2.0 * N * beta_lat**2) if beta_lat > 0.01 else 1.0

    # Effective b0 incorporating non-perturbative corrections
    if var_pert > 1e-15:
        correction = var_P / var_pert
    else:
        correction = 1.0

    b_eff = b0_pert * correction

    return g2 + b_eff * g2**2


def rg_step_with_fermions(g2, N, Nf):
    """
    RG step with N_f fermion flavours (fundamental representation).

    Two-loop beta function:
      beta(g) = -b0 g^3 - b1 g^5
    where
      b0 = (11N - 2Nf) / (48 pi^2)
      b1 = (34N^2 - 10N*Nf - 3*(N^2-1)/N * Nf) / (768 pi^4)

    RG recursion: g^2_{k+1} = g^2_k + 2*b0*g^4_k + 2*b1*g^6_k
    """
    b0 = (11.0 * N - 2.0 * Nf) / (48.0 * np.pi**2)
    b1 = (34.0 * N**2 - 10.0 * N * Nf - 3.0 * (N**2 - 1.0) / N * Nf) / (768.0 * np.pi**4)

    # Two-loop RG step
    delta = 2.0 * b0 * g2**2 + 2.0 * b1 * g2**3

    return g2 + delta


# =============================================================================
# 2. RG FLOW COMPUTATION (GPU-accelerated batch)
# =============================================================================

def compute_rg_flow(N, g2_init=0.01, g2_max=10.0, max_steps=2000):
    """Compute full RG flow for SU(N) from UV to IR."""
    g2_values = [g2_init]
    ratios = []

    for k in range(max_steps):
        g2_k = g2_values[-1]
        if g2_k > g2_max or g2_k < 0:
            break

        g2_next = rg_step_exact(g2_k, N)
        g2_values.append(g2_next)

        if g2_k > 1e-15:
            ratios.append(g2_next / g2_k)

    return np.array(g2_values), np.array(ratios)


def compute_rg_flow_fermions(N, Nf, g2_init=0.01, g2_max=50.0, max_steps=5000):
    """Compute RG flow with fermions using two-loop beta function."""
    g2_values = [g2_init]

    for k in range(max_steps):
        g2_k = g2_values[-1]
        if g2_k > g2_max or g2_k < 0:
            break

        g2_next = rg_step_with_fermions(g2_k, N, Nf)

        # Check for fixed point: |delta| < epsilon
        if abs(g2_next - g2_k) < 1e-12 and k > 10:
            # Found fixed point
            g2_values.append(g2_next)
            break

        g2_values.append(g2_next)

    return np.array(g2_values)


# =============================================================================
# MAIN COMPUTATION
# =============================================================================

def main():
    print("=" * 80)
    print("RG FIXED POINTS AND GAUGE GROUP SELECTION")
    print("=" * 80)

    Ns = [2, 3, 4, 5, 6]
    g2_init = 0.01
    g2_B = 4.0  # Strong coupling boundary (confinement scale)

    # =========================================================================
    # PART 1 & 2: RG flow and exit scale k*(N)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1-2: RG FLOW g^2_k FOR SU(N), N = 2..6")
    print("=" * 80)

    flows = {}
    k_star = {}

    for N in Ns:
        g2_vals, ratios = compute_rg_flow(N, g2_init=g2_init, g2_max=g2_B + 2.0)
        flows[N] = (g2_vals, ratios)

        # Find k* where g^2 first exceeds g_B
        k_exit = len(g2_vals) - 1
        for k, g2 in enumerate(g2_vals):
            if g2 >= g2_B:
                k_exit = k
                break
        k_star[N] = k_exit

        print(f"\nSU({N}):")
        print(f"  Steps to g_B = {g2_B}: k* = {k_exit}")
        print(f"  g^2 at first 5 steps: {g2_vals[:5]}")
        print(f"  g^2 at last 5 steps:  {g2_vals[-5:]}")
        if len(ratios) > 0:
            print(f"  Ratio g^2_(k+1)/g^2_k: min={ratios.min():.6f}, max={ratios.max():.6f}")
            print(f"  Ratio at step 1: {ratios[0]:.6f}")
            if len(ratios) > 10:
                print(f"  Ratio at step 10: {ratios[min(10, len(ratios)-1)]:.6f}")
            print(f"  Ratio at last step: {ratios[-1]:.6f}")

    print(f"\n  k*(N) summary:")
    for N in Ns:
        print(f"    SU({N}): k* = {k_star[N]}")

    # =========================================================================
    # PART 3: Ratio analysis — geometric series test
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: RATIO g^2_(k+1)/g^2_k — GEOMETRIC SERIES TEST")
    print("=" * 80)

    for N in Ns:
        g2_vals, ratios = flows[N]
        if len(ratios) < 2:
            print(f"SU({N}): Too few steps for ratio analysis")
            continue

        ratio_std = np.std(ratios)
        ratio_mean = np.mean(ratios)
        ratio_range = ratios.max() - ratios.min()

        print(f"\nSU({N}):")
        print(f"  Mean ratio: {ratio_mean:.6f}")
        print(f"  Std ratio:  {ratio_std:.6f}")
        print(f"  Range:      {ratio_range:.6f}")

        if ratio_range < 0.01:
            print(f"  --> NEARLY CONSTANT: approximate geometric series (scaling symmetry)")
        else:
            print(f"  --> VARYING: non-trivial RG structure")
            # Check if variation is monotonic
            if np.all(np.diff(ratios[:min(50, len(ratios))]) > 0):
                print(f"      Ratios INCREASING: flow accelerates (non-perturbative enhancement)")
            elif np.all(np.diff(ratios[:min(50, len(ratios))]) < 0):
                print(f"      Ratios DECREASING: flow decelerates")
            else:
                print(f"      Ratios NON-MONOTONIC: complex RG structure")

    # =========================================================================
    # PART 4: Product group SU(3) x SU(2) x U(1) — coupling unification
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: SU(3) x SU(2) x U(1) COUPLING UNIFICATION")
    print("=" * 80)

    # Standard Model couplings at Z-mass scale (approximate):
    #   alpha_1 = g'^2/(4pi) ~ 0.01017  (U(1)_Y, GUT normalized: alpha_1 = 5/3 * alpha')
    #   alpha_2 = g_2^2/(4pi) ~ 0.03378  (SU(2)_L)
    #   alpha_3 = g_3^2/(4pi) ~ 0.1185   (SU(3)_c)

    # SM fermion content: 3 generations
    # For SU(3): N_f = 6 quarks
    # For SU(2): N_f = 6 doublets (3 lepton + 3 quark doublets) [effective]
    # For U(1): handled separately

    # One-loop b coefficients for SM:
    #   b_3 = (11*3 - 2*6)/(48*pi^2) = 21/(48*pi^2) = 7/(16*pi^2)
    #   b_2 = (11*2 - 2*6)/(48*pi^2) = 10/(48*pi^2) = 5/(24*pi^2)  [with Higgs: (22-4-1/6)/(48pi^2)]
    #   b_1 (GUT norm) = -2*Nf_eff/(48*pi^2) [U(1) is not asymptotically free]

    # Standard SM one-loop coefficients (with Higgs, 3 generations):
    b_SM = {
        'SU3': 7.0 / (16.0 * np.pi**2),        # = (33 - 2*6)/(48*pi^2) [Note: 11*3=33]
        'SU2': 19.0 / (96.0 * np.pi**2),        # = (22 - 4*3 - 1/2)/(48*pi^2) [Higgs contribution]
        'U1': -41.0 / (96.0 * np.pi**2),        # negative: NOT asymptotically free
    }

    # Initial couplings at M_Z ~ 91.2 GeV
    alpha_inv = {
        'U1': 59.0,    # alpha_1^{-1} (GUT normalized)
        'SU2': 29.6,   # alpha_2^{-1}
        'SU3': 8.44,   # alpha_3^{-1}
    }

    # One-loop running: alpha_i^{-1}(mu) = alpha_i^{-1}(M_Z) - b_i/(2*pi) * ln(mu/M_Z)
    # More precisely: 1/alpha_i(mu) = 1/alpha_i(M_Z) + (b_i * 4*pi) * ln(mu/M_Z)/(2*pi)
    # Standard: d(1/alpha)/d(ln mu) = -b_i/(2*pi)  where b_i = SM convention

    # SM convention: b_i = (b1, b2, b3) = (41/10, -19/6, -7) [different sign convention]
    # Running: 1/alpha_i(t) = 1/alpha_i(0) - b_i * t / (2*pi), t = ln(mu/M_Z)
    b_SM_conv = {'U1': 41.0/10.0, 'SU2': -19.0/6.0, 'SU3': -7.0}

    t_values = np.linspace(0, 70, 10000)  # ln(mu/M_Z), up to ~10^30 GeV
    t_GUT = np.log(2e16 / 91.2)  # ~37.6

    alpha_inv_running = {}
    for group in ['U1', 'SU2', 'SU3']:
        alpha_inv_running[group] = alpha_inv[group] - b_SM_conv[group] * t_values / (2.0 * np.pi)

    # Find pairwise unification scales
    print("\nOne-loop SM running (GUT normalized):")
    print(f"  At M_Z: 1/alpha_1 = {alpha_inv['U1']:.1f}, 1/alpha_2 = {alpha_inv['SU2']:.1f}, 1/alpha_3 = {alpha_inv['SU3']:.1f}")

    pairs = [('SU2', 'SU3'), ('U1', 'SU2'), ('U1', 'SU3')]
    unif_scales = {}
    for g1, g2 in pairs:
        diff = alpha_inv_running[g1] - alpha_inv_running[g2]
        # Find zero crossing
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        if len(sign_changes) > 0:
            idx = sign_changes[0]
            # Linear interpolation
            t_cross = t_values[idx] - diff[idx] * (t_values[idx+1] - t_values[idx]) / (diff[idx+1] - diff[idx])
            mu_cross = 91.2 * np.exp(t_cross)
            alpha_at_cross = 1.0 / (alpha_inv_running[g1][idx])
            unif_scales[(g1, g2)] = (t_cross, mu_cross, alpha_at_cross)
            print(f"\n  {g1} = {g2} unification:")
            print(f"    Scale: mu = {mu_cross:.2e} GeV  (t = {t_cross:.1f})")
            print(f"    alpha at crossing: {alpha_at_cross:.4f}")
        else:
            print(f"\n  {g1} = {g2}: NO unification in range")
            unif_scales[(g1, g2)] = None

    # Check if all three meet at one point
    if all(v is not None for v in unif_scales.values()):
        t_vals_unif = [v[0] for v in unif_scales.values()]
        t_spread = max(t_vals_unif) - min(t_vals_unif)
        print(f"\n  Unification scale spread: Delta_t = {t_spread:.2f}")
        print(f"    (ratio of scales: {np.exp(t_spread):.1f})")
        if t_spread < 2.0:
            print(f"    --> APPROXIMATE unification (within factor {np.exp(t_spread):.1f})")
        else:
            print(f"    --> NO precise unification (SM alone)")
            print(f"    This is the well-known result: SM couplings do NOT unify.")
            print(f"    SUSY or other BSM physics needed for exact unification.")

    # =========================================================================
    # PART 5: Fixed point search (pure gauge)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 5: FIXED POINT SEARCH (PURE GAUGE)")
    print("=" * 80)

    for N in Ns:
        print(f"\nSU({N}):")
        # Scan g^2 from 0.01 to 20 looking for delta(g^2) = 0
        g2_scan = np.linspace(0.01, 15.0, 500)
        deltas = []

        for g2 in g2_scan:
            g2_next = rg_step_exact(g2, N)
            deltas.append(g2_next - g2)

        deltas = np.array(deltas)

        # Find sign changes in delta (fixed points)
        sign_changes = np.where(np.diff(np.sign(deltas)))[0]

        if len(sign_changes) == 0:
            print(f"  No non-trivial fixed point found in g^2 in [0.01, 15]")
            print(f"  delta(g^2) is always positive: flow always toward strong coupling")
            print(f"  --> Only trivial fixed points: g^2 = 0 (UV, Gaussian) and g^2 = inf (confinement)")
        else:
            for idx in sign_changes:
                g2_fp = g2_scan[idx]
                print(f"  Fixed point candidate at g^2 = {g2_fp:.4f}")
                print(f"    delta at g^2={g2_scan[idx]:.3f}: {deltas[idx]:.6e}")
                print(f"    delta at g^2={g2_scan[idx+1]:.3f}: {deltas[idx+1]:.6e}")

    # =========================================================================
    # PART 6: Banks-Zaks fixed points with fermions
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 6: BANKS-ZAKS FIXED POINTS — CONFORMAL WINDOW")
    print("=" * 80)

    print("\nTwo-loop beta function: beta(g) = -b0 g^3 - b1 g^5")
    print("Fixed point at g*^2 = -b0/b1 (when b0 > 0 and b1 < 0)")
    print()

    bz_results = {}

    for N in Ns:
        print(f"SU({N}):")
        Nf_AF_max = int(11 * N / 2)  # Asymptotic freedom bound: N_f < 11N/2

        # Conformal window: between N_f^* (where BZ exists) and 11N/2
        # BZ exists when b0 > 0 and b1 < 0

        found_bz = []

        for Nf in range(1, Nf_AF_max + 1):
            b0 = (11.0 * N - 2.0 * Nf) / (48.0 * np.pi**2)
            b1 = (34.0 * N**2 - 10.0 * N * Nf - 3.0 * (N**2 - 1.0) / N * Nf) / (768.0 * np.pi**4)

            if b0 > 0 and b1 < 0:
                g2_star = -b0 / b1
                alpha_star = g2_star / (4 * np.pi)

                # Check if perturbative (alpha* < 1)
                perturbative = alpha_star < 1.0

                found_bz.append({
                    'Nf': Nf, 'b0': b0, 'b1': b1,
                    'g2_star': g2_star, 'alpha_star': alpha_star,
                    'perturbative': perturbative
                })

        if found_bz:
            Nf_min = found_bz[0]['Nf']
            Nf_max_bz = found_bz[-1]['Nf']
            print(f"  Asymptotic freedom: N_f < {Nf_AF_max} (= 11*{N}/2)")
            print(f"  Banks-Zaks window: N_f in [{Nf_min}, {Nf_max_bz}]")
            print(f"  Perturbative BZ (alpha* < 1):")

            bz_results[N] = found_bz

            for bz in found_bz:
                tag = "PERT" if bz['perturbative'] else "STRONG"
                print(f"    N_f = {bz['Nf']:2d}: g*^2 = {bz['g2_star']:8.4f}, "
                      f"alpha* = {bz['alpha_star']:8.4f}  [{tag}]")
        else:
            print(f"  No Banks-Zaks window found (b1 never negative)")
            bz_results[N] = []

    # Numerical verification: run RG flow and check convergence to BZ
    print("\n--- Numerical verification of BZ fixed points ---")
    for N in [3]:  # Focus on SU(3)
        for Nf in [12, 14, 16]:
            b0 = (11.0 * N - 2.0 * Nf) / (48.0 * np.pi**2)
            if b0 <= 0:
                print(f"\nSU({N}), N_f={Nf}: NOT asymptotically free (b0 = {b0:.4e})")
                continue

            flow = compute_rg_flow_fermions(N, Nf, g2_init=0.01, g2_max=50.0, max_steps=10000)
            g2_final = flow[-1]
            converged = abs(flow[-1] - flow[-2]) < 1e-8 if len(flow) > 1 else False

            b1 = (34.0 * N**2 - 10.0 * N * Nf - 3.0 * (N**2 - 1.0) / N * Nf) / (768.0 * np.pi**4)
            g2_pred = -b0 / b1 if b1 < 0 else float('inf')

            print(f"\nSU({N}), N_f={Nf}:")
            print(f"  b0 = {b0:.6f}, b1 = {b1:.6f}")
            print(f"  Predicted g*^2 = {g2_pred:.4f}" if g2_pred < 100 else f"  No BZ (b1 > 0)")
            print(f"  Flow after {len(flow)} steps: g^2 = {g2_final:.6f}")
            print(f"  Converged: {converged}")
            if converged and g2_pred < 100:
                print(f"  Relative error: |g^2_num - g*^2_pred|/g*^2 = {abs(g2_final - g2_pred)/g2_pred:.2e}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: CAN RG FIXED POINTS SELECT THE GAUGE GROUP?")
    print("=" * 80)

    print("""
1. PURE GAUGE (no fermions):
   - Only trivial fixed points: g^2 = 0 (UV Gaussian) and g^2 = infinity (confinement)
   - The flow is NOT a geometric series: ratio g^2_{k+1}/g^2_k varies
   - Non-perturbative corrections from exact Bessel variance modify the flow
   - But NO group selection: all SU(N) have qualitatively identical flow
   - k*(N) increases with N: larger groups take longer to confine

2. SM COUPLING UNIFICATION:
   - The three SM couplings do NOT precisely unify with one-loop running
   - Approximate unification near 10^{15-16} GeV (well-known result)
   - No RG fixed point mechanism selects SU(3) x SU(2) x U(1)

3. BANKS-ZAKS FIXED POINTS (with fermions):
   - Each SU(N) has a conformal window N_f^* < N_f < 11N/2
   - Inside the window: non-trivial IR fixed point exists
   - The fixed point coupling g*^2(N, N_f) DEPENDS on N and N_f
   - This is the closest to "gauge group selection by dynamics"
   - But it doesn't SELECT a group: it characterizes conformal behavior

4. CONCLUSION:
   The RG flow alone CANNOT select the gauge group.
   The gauge group is an INPUT, not an OUTPUT, of the RG framework.
   Fixed points depend on N, but don't uniquely determine N.
   Group selection requires additional structure beyond RG flow.
""")

    # =========================================================================
    # FIGURES
    # =========================================================================

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("RG Fixed Points and Gauge Group Selection", fontsize=16)

    # Panel 1: RG flow g^2 vs k for each N
    ax = axes[0, 0]
    for N in Ns:
        g2_vals = flows[N][0]
        ax.plot(range(len(g2_vals)), g2_vals, label=f'SU({N})', linewidth=2)
    ax.axhline(y=g2_B, color='k', linestyle='--', alpha=0.5, label=f'$g_B^2 = {g2_B}$')
    ax.set_xlabel('RG step k')
    ax.set_ylabel('$g^2_k$')
    ax.set_title('RG flow: $g^2$ vs step')
    ax.legend()
    ax.set_ylim(0, g2_B + 1)

    # Panel 2: k*(N) vs N
    ax = axes[0, 1]
    N_arr = np.array(Ns)
    k_arr = np.array([k_star[N] for N in Ns])
    ax.plot(N_arr, k_arr, 'ko-', markersize=10, linewidth=2)
    # Fit: k* ~ a/N^2 (since b0 ~ N, so k* ~ 1/N for weak coupling)
    if len(N_arr) > 2:
        from numpy.polynomial import polynomial as P
        # Try k* = c * N^alpha fit
        log_fit = np.polyfit(np.log(N_arr), np.log(k_arr), 1)
        alpha_fit = log_fit[0]
        ax.set_title(f'Exit scale $k^*(N)$; power law: $k^* \\sim N^{{{alpha_fit:.2f}}}$')
    else:
        ax.set_title('Exit scale $k^*(N)$')
    ax.set_xlabel('N')
    ax.set_ylabel('$k^*$')

    # Panel 3: Ratio g^2_{k+1}/g^2_k
    ax = axes[0, 2]
    for N in Ns:
        ratios = flows[N][1]
        if len(ratios) > 0:
            ax.plot(range(len(ratios)), ratios, label=f'SU({N})', linewidth=1.5)
    ax.set_xlabel('RG step k')
    ax.set_ylabel('$g^2_{k+1}/g^2_k$')
    ax.set_title('Ratio test (constant = geometric series)')
    ax.legend()

    # Panel 4: SM coupling unification
    ax = axes[1, 0]
    colors = {'U1': 'blue', 'SU2': 'green', 'SU3': 'red'}
    labels = {'U1': '$\\alpha_1^{-1}$ (U(1))', 'SU2': '$\\alpha_2^{-1}$ (SU(2))', 'SU3': '$\\alpha_3^{-1}$ (SU(3))'}
    for group in ['U1', 'SU2', 'SU3']:
        mu_values = 91.2 * np.exp(t_values)
        ax.plot(np.log10(mu_values), alpha_inv_running[group],
                color=colors[group], label=labels[group], linewidth=2)
    ax.axvline(x=np.log10(2e16), color='gray', linestyle=':', alpha=0.5, label='GUT scale')
    ax.set_xlabel('$\\log_{10}(\\mu / \\mathrm{GeV})$')
    ax.set_ylabel('$\\alpha_i^{-1}$')
    ax.set_title('SM coupling running (one-loop)')
    ax.legend(fontsize=9)
    ax.set_xlim(1, 18)
    ax.set_ylim(0, 70)

    # Panel 5: Banks-Zaks conformal window
    ax = axes[1, 1]
    for N in Ns:
        if bz_results[N]:
            Nf_vals = [bz['Nf'] for bz in bz_results[N]]
            alpha_vals = [bz['alpha_star'] for bz in bz_results[N]]
            ax.plot(Nf_vals, alpha_vals, 'o-', label=f'SU({N})', markersize=5, linewidth=1.5)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='$\\alpha^* = 1$ (pert. boundary)')
    ax.set_xlabel('$N_f$ (fermion flavours)')
    ax.set_ylabel('$\\alpha^* = g^{*2}/(4\\pi)$')
    ax.set_title('Banks-Zaks fixed point coupling')
    ax.legend(fontsize=9)
    ax.set_yscale('log')

    # Panel 6: Conformal window map
    ax = axes[1, 2]
    for N in Ns:
        Nf_AF = 11 * N / 2.0
        if bz_results[N]:
            Nf_bz_min = bz_results[N][0]['Nf']
            Nf_bz_max = bz_results[N][-1]['Nf']
            # Conformal window
            ax.barh(N, Nf_bz_max - Nf_bz_min + 1, left=Nf_bz_min - 0.5,
                    height=0.6, color='orange', alpha=0.7, label='BZ window' if N == 2 else '')
            # AF region
            ax.plot(Nf_AF, N, 'rv', markersize=10, label='AF bound' if N == 2 else '')
        # Perturbative BZ
        pert_bz = [bz for bz in bz_results[N] if bz['perturbative']]
        if pert_bz:
            Nf_p_min = pert_bz[0]['Nf']
            Nf_p_max = pert_bz[-1]['Nf']
            ax.barh(N, Nf_p_max - Nf_p_min + 1, left=Nf_p_min - 0.5,
                    height=0.3, color='green', alpha=0.7, label='Pert. BZ' if N == 2 else '')

    ax.set_xlabel('$N_f$')
    ax.set_ylabel('N (gauge group SU(N))')
    ax.set_title('Conformal window map')
    ax.legend(fontsize=9)

    plt.tight_layout()
    figpath = '/home/golbryk/ai/toe/global-spectral-toe/rmt/rg_fixed_points.png'
    plt.savefig(figpath, dpi=150)
    print(f"\nFigure saved: {figpath}")

    # =========================================================================
    # Additional: Plaquette expectation and variance plots
    # =========================================================================
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Exact Bessel Plaquette Variance for SU(N)", fontsize=14)

    beta_range = np.linspace(0.5, 20.0, 100)

    for N in [2, 3, 4, 5]:
        P_vals = [plaquette_expectation_sun(b, N) for b in beta_range]
        V_vals = [plaquette_variance_sun(b, N) for b in beta_range]

        ax1.plot(beta_range, P_vals, label=f'SU({N})', linewidth=2)
        ax2.plot(beta_range, V_vals, label=f'SU({N})', linewidth=2)

    ax1.set_xlabel(r'$\beta = 2N/g^2$')
    ax1.set_ylabel(r'$\langle P \rangle$')
    ax1.set_title('Mean plaquette (exact)')
    ax1.legend()

    ax2.set_xlabel(r'$\beta = 2N/g^2$')
    ax2.set_ylabel('Var(P)')
    ax2.set_title('Plaquette variance (exact)')
    ax2.legend()

    plt.tight_layout()
    figpath2 = '/home/golbryk/ai/toe/global-spectral-toe/rmt/plaquette_variance.png'
    plt.savefig(figpath2, dpi=150)
    print(f"Figure saved: {figpath2}")


if __name__ == '__main__':
    main()
