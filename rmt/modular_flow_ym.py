#!/usr/bin/env python3
"""
Modular Flow of the Yang-Mills Vacuum on an n-Plaquette Chain
=============================================================

Author : Grzegorz Olbryk
Date   : March 2026

Applies Tomita-Takesaki modular theory to the 1D SU(2) lattice Yang-Mills
vacuum.  The transfer matrix T has eigenvalues A_p(beta) with degeneracy d_p^2.

Key constructions:
  1. Ground state |Omega> = sum_p c_p |p_L> x |p_R>  (thermofield double)
  2. Reduced density matrix rho_L = diag(|c_p|^2)
  3. Modular Hamiltonian K_L = -log(rho_L)
  4. Modular frequencies omega_pq = K_p - K_q
  5. Locality / causality test via commutator growth under modular flow
  6. Emergent modular distance d(j,k)

Uses CuPy for GPU acceleration of the locality tests (large matrix operations).
"""

import numpy as np
import cupy as cp
from math import factorial
from scipy.special import iv as bessel_iv
import time
import sys

# ===========================================================================
# SU(2) Transfer Matrix Eigenvalues
# ===========================================================================

def dim_su2(p):
    """Dimension of SU(2) irrep labeled by p (spin j = p/2): d_p = p + 1."""
    return p + 1


def casimir_su2(p):
    """Quadratic Casimir for SU(2) irrep p: C_2 = j(j+1) = p(p+2)/4."""
    return p * (p + 2) / 4.0


def A_p_wilson_su2(p, beta):
    """
    Transfer matrix eigenvalue A_p(beta) for Wilson action on SU(2).

    A_p(beta) = integral_0^pi  chi_p(theta) * exp(beta * cos(theta))
                * (2/pi) sin^2(theta) dtheta

    where chi_p(theta) = sin((p+1)theta) / sin(theta).

    Normalized so A_0 = 1 at the trivial rep.

    Uses high-precision numerical quadrature.
    """
    n_quad = 4000
    theta = np.linspace(1e-12, np.pi, n_quad)
    dtheta = theta[1] - theta[0]

    # Haar measure for SU(2): (2/pi) sin^2(theta)
    haar = (2.0 / np.pi) * np.sin(theta)**2

    # Character: chi_p(theta) = sin((p+1)*theta) / sin(theta)
    chi = np.sin((p + 1) * theta) / np.sin(theta)

    # Wilson action: S = beta * (1 - cos(theta)) but for transfer matrix
    # we use exp(beta * cos(theta)) (the constant doesn't matter for ratios)
    boltzmann = np.exp(beta * np.cos(theta))

    integrand = chi * boltzmann * haar
    result = np.sum(integrand) * dtheta

    return result


def compute_transfer_eigenvalues(n_reps, beta):
    """Compute A_p(beta) for p = 0, 1, ..., n_reps-1."""
    Ap = np.array([A_p_wilson_su2(p, beta) for p in range(n_reps)])
    return Ap


# ===========================================================================
# Ground State and Modular Hamiltonian
# ===========================================================================

def build_ground_state(Ap, n_plaq, n_reps):
    """
    Ground state of n-plaquette chain.

    Z_n = sum_p d_p^2 A_p^n
    |Omega> = sum_p c_p |p_L> x |p_R>
    c_p = sqrt(d_p^2 A_p^n / Z_n)

    Returns: c_p coefficients (real, positive), Z_n
    """
    dp = np.array([dim_su2(p) for p in range(n_reps)], dtype=np.float64)

    # Use log-space to avoid overflow for large n
    log_terms = 2.0 * np.log(dp) + n_plaq * np.log(np.abs(Ap))
    log_max = np.max(log_terms)
    log_Zn = log_max + np.log(np.sum(np.exp(log_terms - log_max)))

    # c_p^2 = d_p^2 A_p^n / Z_n
    log_cp2 = log_terms - log_Zn
    cp2 = np.exp(log_cp2)
    cp_coeff = np.sqrt(cp2)

    Z_n = np.exp(log_Zn)
    return cp_coeff, cp2, Z_n


def modular_hamiltonian(cp2):
    """
    Modular Hamiltonian eigenvalues: K_p = -log(|c_p|^2) = -log(d_p^2 A_p^n / Z_n).

    Returns K_p array.
    """
    # Avoid log(0) for negligible components
    mask = cp2 > 1e-300
    K = np.full_like(cp2, np.inf)
    K[mask] = -np.log(cp2[mask])
    return K


# ===========================================================================
# Modular Flow and Frequencies
# ===========================================================================

def modular_frequencies(K, n_reps):
    """
    Modular frequencies omega_pq = K_p - K_q.

    sigma_t(|p><q|) = exp(i * omega_pq * t) |p><q|
    """
    omega = np.zeros((n_reps, n_reps))
    for p in range(n_reps):
        for q in range(n_reps):
            if np.isfinite(K[p]) and np.isfinite(K[q]):
                omega[p, q] = K[p] - K[q]
            else:
                omega[p, q] = np.nan
    return omega


# ===========================================================================
# Locality Tests — Site-Localized Operators
# ===========================================================================

def build_site_operator(site, n_plaq, n_reps):
    """
    Build a site-localized operator O_j in the transfer matrix Hilbert space.

    The Hilbert space of the n-plaquette chain is:
      H = (rep space at link 0) x (rep space at link 1) x ... x (rep space at link n)

    For SU(2), each link carries all irreps. The transfer matrix acts on the
    rep-space as T = sum_p d_p^2 A_p |p><p|.

    A plaquette operator at site j connects link j to link j+1.
    In the rep basis, a localized operator O_j is:
      O_j = sum_{p,q} M^j_{pq} |p><q|

    We define O_j via the plaquette action character at site j:
      <p|O_j|q> = delta_{p,q-1} + delta_{p,q+1}  (raising/lowering in rep space)

    This is the simplest non-diagonal localized operator (analogous to
    cos(theta_j) in the angle basis, which connects adjacent representations
    via the recurrence of Chebyshev polynomials: cos(theta)*chi_p = (chi_{p-1} + chi_{p+1})/2).
    """
    # For the full n-plaquette chain, the Hilbert space dimension is n_reps^n_plaq.
    # But since T is diagonal in rep basis, the ground state is a product state
    # in the LEFT/RIGHT bipartition.
    #
    # For the locality test, we work in the FULL chain Hilbert space.
    # State: |p1, p2, ..., p_n> where p_j labels the rep flowing through link j.
    #
    # Actually, for 1D gauge theory with open boundary conditions,
    # the physical Hilbert space after gauge fixing is labeled by a single rep p
    # at each link. The transfer matrix T acts as: T|p> = A_p |p> on this space.
    #
    # For MULTIPLE plaquettes, the full Hilbert space before tracing is:
    # |p_1, p_2, ..., p_{n-1}> where p_j is the rep at link j.
    # The transfer matrix couples adjacent reps via the 6j symbols.
    #
    # HOWEVER, for Wilson action on a 1D chain, the transfer matrix factorizes:
    # Z_n = Tr(T^n) = sum_p d_p^2 A_p^n
    # This means T is already diagonal — the chain has no spatial structure
    # beyond the rep label.
    #
    # KEY REALIZATION: To get non-trivial spatial structure, we need to
    # work with the LINK Hilbert space, not the rep-diagonal basis.
    # Each link carries L^2(SU(2)) = direct sum of V_p x V_p.
    # The full Hilbert space is L^2(SU(2))^{otimes (n+1)}.
    #
    # For computational tractability, we truncate to n_reps representations
    # and use the fact that in the rep basis, the plaquette operator at site j
    # is: P_j = sum_p d_p A_p |p><p| acting on the p-th sector between
    # links j and j+1.
    #
    # A simple localized observable: measure the "flux" at a specific link.
    # In rep basis: O_j = sum_p f(p) * (projector onto rep p at link j)
    # where f(p) = C_2(p) = p(p+2)/4 (Casimir = "electric field squared")

    # For the n-plaquette chain, the full Hilbert space is n_reps^n_plaq
    # dimensional (one rep label per plaquette).
    # But this grows too fast. Instead, we use the TRANSFER MATRIX approach:
    #
    # The chain state space is spanned by |p> (single rep), and
    # n plaquettes just give T^n. Spatial structure requires going beyond
    # the diagonal basis.
    #
    # RESOLUTION: Use the MATRIX ELEMENTS of the transfer matrix.
    # In the full group element basis, the transfer matrix is:
    #   T(U, U') = exp(beta * Re Tr(U U'^dag))
    # This is NOT diagonal — it couples different group elements.
    # In the Peter-Weyl basis, it becomes diagonal.
    #
    # For locality, we work in the tensor product of representation spaces.
    # The key insight: the ground state |Omega> lives in the product of
    # n copies of the rep space. We can define site operators as those
    # that act on a specific tensor factor.

    pass  # See the matrix-based implementation below


def build_chain_hilbert_space(n_plaq, n_reps, Ap, beta):
    """
    Build the full chain Hilbert space and site-localized operators.

    For an n-plaquette OPEN chain with SU(2) gauge group:
    - Physical Hilbert space after gauge fixing: one rep label per link
    - n_plaq plaquettes = n_plaq + 1 links, but boundary conditions
      reduce to n_plaq - 1 internal degrees of freedom for periodic,
      or a single rep for open (1D is trivial after gauge fixing).

    For NON-TRIVIAL modular structure, we use the THERMAL STATE approach:
    - State: rho = T^n / Z_n (thermal state of the transfer matrix)
    - Subalgebra: operators localized on the LEFT half of the chain
    - Modular flow acts on this subalgebra

    In the Peter-Weyl basis, rho = diag(d_p^2 A_p^n / Z_n).

    The LEFT/RIGHT split for modular theory uses the thermofield double:
    |Omega> = sum_p sqrt(d_p^2 A_p^n / Z_n) |p>_L |p>_R

    Site-localized operators in the tensor product H_L x H_R:
    - LEFT operator at "position j": O_j^L = f_j(p) acting on H_L
      where f_j encodes the position dependence
    - For the THERMAL interpretation, "position" = imaginary time
      t_E = j * a (lattice spacing a = 1)

    The crucial operator: O(t_E) = T^{t_E/a} O T^{-t_E/a} in imaginary time.

    In the diagonal basis:
      <p|O(t_E)|q> = (A_p/A_q)^{t_E/a} * <p|O|q>

    For the modular flow (REAL time t):
      sigma_t(<p|O|q>) = exp(i * omega_pq * t) * <p|O|q>
      omega_pq = K_p - K_q = log(A_q^n/A_p^n * d_q^2/d_p^2)  [from c_p^2 ratios]

    We define site operators via imaginary-time translation:
      O_j := O at imaginary time t_E = j (in lattice units)

    Then in modular flow:
      [sigma_t(O_j), O_k] measures whether modular flow connects sites j and k.
    """
    pass  # See implementation in main analysis


# ===========================================================================
# GPU-accelerated commutator computation
# ===========================================================================

def compute_commutator_norm_gpu(omega_pq, O_j_matrix, O_k_matrix, t_values):
    """
    Compute ||[sigma_t(O_j), O_k]||_F as function of t.

    sigma_t(O_j)_{pq} = exp(i * omega_pq * t) * (O_j)_{pq}

    [sigma_t(O_j), O_k]_{pr} = sum_q [exp(i*omega_pq*t) * O_j_{pq} * O_k_{qr}
                                      - O_k_{pq} * exp(i*omega_qr*t) * O_j_{qr}]

    Returns: array of ||[...]||_F for each t value
    """
    omega_gpu = cp.asarray(omega_pq)
    Oj_gpu = cp.asarray(O_j_matrix)
    Ok_gpu = cp.asarray(O_k_matrix)

    n = omega_gpu.shape[0]
    norms = np.zeros(len(t_values))

    for it, t in enumerate(t_values):
        # Phase matrix
        phase = cp.exp(1j * omega_gpu * t)

        # sigma_t(O_j)
        Oj_t = phase * Oj_gpu

        # Commutator
        comm = Oj_t @ Ok_gpu - Ok_gpu @ Oj_t

        # Frobenius norm
        norms[it] = float(cp.sqrt(cp.sum(cp.abs(comm)**2)).real)

    return norms


# ===========================================================================
# Site operator construction (imaginary-time based)
# ===========================================================================

def build_site_operators_imaginary_time(Ap, n_plaq, n_reps):
    """
    Build site-localized operators using imaginary-time displacement.

    The base operator O is chosen as the "electric field" (Casimir operator):
      O_{pq} = delta_{pq} * C_2(p)  (diagonal)

    But diagonal operators have trivial commutators!

    Instead, use the PLAQUETTE operator, which in rep basis connects
    adjacent reps via the fusion rules:
      P_{pq} = <p| Tr U_plaq |q>

    For SU(2), the fundamental character chi_1(theta) = 2*cos(theta),
    and the coupling between reps p and q via a single plaquette is:
      P_{pq} = A_p * delta_{pq}  (in the diagonal basis)

    This is STILL diagonal because the transfer matrix diagonalizes
    the plaquette operator.

    KEY INSIGHT: For non-trivial commutators, we need OFF-DIAGONAL operators.
    The natural choice is the CREATION/ANNIHILATION operator in rep space:

      (a^+)_{pq} = sqrt(p+1) * delta_{p, q+1}   (raise rep)
      (a^-)_{pq} = sqrt(p) * delta_{p, q-1}      (lower rep)

    These are the "electric flux" raising/lowering operators.
    They are localized because they correspond to adding/removing
    a unit of flux at a specific link.

    The imaginary-time displaced version:
      O_j = T^j * a^+ * T^{-j}
      (O_j)_{pq} = (A_p / A_q)^j * (a^+)_{pq}
    """
    # Base operator: raising operator in rep space
    # (a^+)_{pq} = sqrt(d_q) * delta_{p, q+1}
    # More physically: the matrix element of cos(theta) between reps
    # Using the recurrence: cos(theta) * chi_p = (chi_{p+1} + chi_{p-1})/2
    # So <p|cos(theta)|q> = (1/2) * delta_{|p-q|, 1}

    operators = {}

    for j in range(n_plaq):
        O_j = np.zeros((n_reps, n_reps), dtype=complex)
        for p in range(n_reps):
            for q in range(n_reps):
                # cos(theta) matrix element in rep basis
                base_element = 0.0
                if abs(p - q) == 1:
                    base_element = 0.5

                if base_element != 0 and Ap[q] > 1e-300 and Ap[p] > 1e-300:
                    # Imaginary-time displacement by j units
                    ratio = Ap[p] / Ap[q]
                    O_j[p, q] = base_element * ratio**j

        operators[j] = O_j

    return operators


# ===========================================================================
# Modular distance from commutator onset
# ===========================================================================

def find_modular_distance(comm_norms, t_values, threshold=0.01):
    """
    Find the modular distance = earliest time t where ||[sigma_t(O_j), O_k]|| > threshold.

    threshold is relative to max norm.
    """
    max_norm = np.max(comm_norms)
    if max_norm < 1e-15:
        return np.inf

    abs_threshold = threshold * max_norm
    idx = np.where(comm_norms > abs_threshold)[0]
    if len(idx) == 0:
        return np.inf
    return t_values[idx[0]]


# ===========================================================================
# Main Analysis
# ===========================================================================

def analyze_single_config(n_plaq, beta, n_reps=20):
    """Full analysis for one (n_plaq, beta) configuration."""

    print(f"\n{'='*80}")
    print(f"  n = {n_plaq} plaquettes, beta = {beta}, SU(2), n_reps = {n_reps}")
    print(f"{'='*80}")

    # ---- Step 1: Transfer matrix eigenvalues ----
    t0 = time.time()
    Ap = compute_transfer_eigenvalues(n_reps, beta)

    # Normalize so A_0 = 1 (doesn't affect physics)
    A0 = Ap[0]
    Ap_norm = Ap / A0

    print(f"\n  Transfer matrix eigenvalues A_p / A_0:")
    for p in range(min(10, n_reps)):
        dp = dim_su2(p)
        C2 = casimir_su2(p)
        print(f"    p={p:2d}  d_p={dp:3d}  C_2={C2:8.3f}  A_p/A_0 = {Ap_norm[p]:.6e}")

    # ---- Step 2: Ground state ----
    cp_coeff, cp2, Z_n = build_ground_state(Ap_norm, n_plaq, n_reps)

    print(f"\n  Ground state coefficients |c_p|^2 (d_p^2 * A_p^n / Z_n):")
    print(f"  Z_n = {Z_n:.6e}")
    n_significant = 0
    for p in range(min(15, n_reps)):
        if cp2[p] > 1e-20:
            n_significant += 1
            print(f"    p={p:2d}  |c_p|^2 = {cp2[p]:.6e}  c_p = {cp_coeff[p]:.6e}")
    print(f"  Number of significant reps (|c_p|^2 > 1e-20): {n_significant}")

    # ---- Step 3: Modular Hamiltonian ----
    K = modular_hamiltonian(cp2)

    print(f"\n  Modular Hamiltonian eigenvalues K_p = -log|c_p|^2:")
    for p in range(min(10, n_reps)):
        if np.isfinite(K[p]):
            print(f"    p={p:2d}  K_p = {K[p]:12.6f}")

    # ---- Step 4: Modular frequencies ----
    omega = modular_frequencies(K, n_reps)

    print(f"\n  Modular frequencies omega_pq = K_p - K_q (first 6x6 block):")
    n_show = min(6, n_significant)
    header = "  p\\q  " + "".join(f"{q:>12d}" for q in range(n_show))
    print(header)
    for p in range(n_show):
        row = f"  {p:3d}  "
        for q in range(n_show):
            if np.isfinite(omega[p, q]):
                row += f"{omega[p,q]:12.4f}"
            else:
                row += f"{'nan':>12s}"
        print(row)

    # ---- Step 4b: Physical interpretation of frequencies ----
    print(f"\n  Physical interpretation of modular frequencies:")
    print(f"  {'p':>3s} {'q':>3s} {'omega_pq':>12s} {'n*log(A_q/A_p)':>16s} {'C2(p)-C2(q)':>14s} {'ratio omega/dC2':>16s}")

    for p in range(min(6, n_significant)):
        for q in range(p+1, min(6, n_significant)):
            if np.isfinite(omega[p, q]):
                dC2 = casimir_su2(p) - casimir_su2(q)
                # omega_pq should equal n*log(A_q/A_p) + 2*log(d_q/d_p)
                log_ratio_A = n_plaq * np.log(Ap_norm[q] / Ap_norm[p]) if Ap_norm[p] > 0 and Ap_norm[q] > 0 else np.nan
                log_ratio_d = 2.0 * np.log(dim_su2(q) / dim_su2(p))
                expected = -(log_ratio_A + log_ratio_d)  # K_p - K_q = -log(c_p^2) + log(c_q^2)
                ratio = omega[p, q] / dC2 if abs(dC2) > 1e-10 else np.nan
                print(f"  {p:3d} {q:3d} {omega[p,q]:12.4f} {log_ratio_A:16.4f} {dC2:14.4f} {ratio:16.6f}")

    # Check if omega_pq is proportional to Casimir difference
    omegas_finite = []
    dC2s = []
    for p in range(min(8, n_significant)):
        for q in range(p+1, min(8, n_significant)):
            if np.isfinite(omega[p, q]):
                omegas_finite.append(omega[p, q])
                dC2s.append(casimir_su2(p) - casimir_su2(q))

    if len(omegas_finite) > 2:
        omegas_arr = np.array(omegas_finite)
        dC2_arr = np.array(dC2s)
        # Linear regression
        if np.std(dC2_arr) > 1e-10:
            slope = np.sum(omegas_arr * dC2_arr) / np.sum(dC2_arr**2)
            residuals = omegas_arr - slope * dC2_arr
            R2 = 1 - np.sum(residuals**2) / np.sum((omegas_arr - np.mean(omegas_arr))**2)
            print(f"\n  Linear fit omega_pq = alpha * (C2(p) - C2(q)):")
            print(f"    alpha = {slope:.6f}")
            print(f"    R^2   = {R2:.6f}")
            if R2 > 0.99:
                print(f"    ** EXCELLENT FIT: modular frequency ~ Casimir **")
                print(f"    ** This means K = alpha * C_2 (modular Hamiltonian = Casimir) **")
            elif R2 > 0.95:
                print(f"    ** GOOD FIT: strong Casimir correlation **")
            else:
                print(f"    ** POOR FIT: modular frequencies are NOT simply Casimir **")

    # ---- Step 5: Locality / Causality Test ----
    print(f"\n  Step 5: Locality Test — Commutator Growth under Modular Flow")
    print(f"  " + "-"*70)

    n_active = min(n_significant, n_reps)

    # Build site operators using imaginary-time displacement
    operators = build_site_operators_imaginary_time(Ap_norm, n_plaq, n_active)

    # Time grid for modular flow
    t_values = np.linspace(0, 5.0, 500)

    # Compute commutator norms for all pairs of sites
    print(f"\n  Computing ||[sigma_t(O_j), O_k]|| for site pairs...")

    # Truncate omega to active reps
    omega_active = omega[:n_active, :n_active]

    comm_results = {}
    for j in range(n_plaq):
        for k in range(j+1, n_plaq):
            O_j = operators[j][:n_active, :n_active]
            O_k = operators[k][:n_active, :n_active]

            norms = compute_commutator_norm_gpu(omega_active, O_j, O_k, t_values)
            comm_results[(j, k)] = norms

    # Report
    print(f"\n  Commutator norms ||[sigma_t(O_j), O_k]||_F:")
    print(f"  {'(j,k)':>8s} {'sep':>5s} {'max norm':>12s} {'t at max':>10s} {'t_onset(1%)':>12s}")

    distances = {}
    for (j, k), norms in sorted(comm_results.items()):
        sep = k - j
        max_norm = np.max(norms)
        t_max = t_values[np.argmax(norms)]
        t_onset = find_modular_distance(norms, t_values, threshold=0.01)
        distances[(j, k)] = t_onset
        print(f"  ({j},{k}){' '*(5-len(f'({j},{k})'))} {sep:5d} {max_norm:12.6e} {t_max:10.4f} {t_onset:12.4f}")

    # ---- Step 6: Emergent Metric ----
    print(f"\n  Step 6: Emergent Modular Metric")
    print(f"  " + "-"*70)

    # Check if d(j,k) is proportional to |j-k|
    seps = []
    d_vals = []
    for (j, k), d in distances.items():
        if np.isfinite(d) and d < 100:
            seps.append(k - j)
            d_vals.append(d)

    if len(seps) > 2:
        seps_arr = np.array(seps, dtype=float)
        d_arr = np.array(d_vals)

        # Linear fit
        if np.std(seps_arr) > 0:
            c_modular = np.sum(seps_arr * d_arr) / np.sum(seps_arr**2)
            residuals = d_arr - c_modular * seps_arr
            R2 = 1 - np.sum(residuals**2) / np.sum((d_arr - np.mean(d_arr))**2)

            print(f"  Fit: d_modular(j,k) = c * |j-k|")
            print(f"    c (modular speed of light inverse) = {c_modular:.6f}")
            print(f"    v_modular (speed of light) = {1.0/c_modular:.6f}" if c_modular > 1e-10 else "    v_modular = inf")
            print(f"    R^2 = {R2:.6f}")

            if R2 > 0.95:
                print(f"    ** METRIC STRUCTURE: d(j,k) ~ |j-k| (1D flat metric) **")
            elif R2 > 0.8:
                print(f"    ** APPROXIMATE metric structure **")
            else:
                print(f"    ** NO clean metric structure **")

        # Print distance matrix
        print(f"\n  Modular distance matrix d(j,k):")
        header = "  j\\k  " + "".join(f"{k:>10d}" for k in range(n_plaq))
        print(header)
        for j in range(n_plaq):
            row = f"  {j:3d}  "
            for k in range(n_plaq):
                if j == k:
                    row += f"{'0':>10s}"
                elif (min(j,k), max(j,k)) in distances:
                    d = distances[(min(j,k), max(j,k))]
                    if np.isfinite(d) and d < 100:
                        row += f"{d:10.4f}"
                    else:
                        row += f"{'inf':>10s}"
                else:
                    row += f"{'---':>10s}"
            print(row)
    else:
        print("  Not enough finite distances for metric analysis.")

    # ---- Step 7: Detailed frequency analysis ----
    print(f"\n  Step 7: Modular Spectrum Analysis")
    print(f"  " + "-"*70)

    # Entanglement entropy from the reduced state
    S_ent = -np.sum(cp2[cp2 > 1e-300] * np.log(cp2[cp2 > 1e-300]))
    print(f"  Entanglement entropy S = -Tr(rho_L log rho_L) = {S_ent:.6f}")
    print(f"  Effective dimension (exp(S)) = {np.exp(S_ent):.4f}")

    # Modular spectrum = K eigenvalues
    K_finite = K[np.isfinite(K)]
    if len(K_finite) > 1:
        K_min = np.min(K_finite)
        K_gaps = np.diff(np.sort(K_finite))
        print(f"  K_min = {K_min:.6f}")
        print(f"  K gaps: {K_gaps[:8]}")

        # Compare to n * log(A_0/A_p) + 2*log(d_0/d_p)
        print(f"\n  Decomposition: K_p = n*log(A_0/A_p) + 2*log(d_0/d_p) + const")
        for p in range(min(8, n_significant)):
            if Ap_norm[p] > 1e-300:
                contrib_A = -n_plaq * np.log(Ap_norm[p])
                contrib_d = -2.0 * np.log(dim_su2(p))
                contrib_Z = np.log(Z_n)
                total = contrib_A + contrib_d + contrib_Z
                print(f"    p={p:2d}: K_p={K[p]:10.4f}  =  {contrib_A:8.4f} (action) + {contrib_d:8.4f} (degeneracy) + {contrib_Z:8.4f} (normalization)")

    # ---- Scaling with n ----
    print(f"\n  Key ratio: omega_01 / n = {omega[0,1]/n_plaq:.6f}" if np.isfinite(omega[0,1]) else "")
    print(f"  Key ratio: K_1 - K_0 = {K[1]-K[0]:.6f}" if (np.isfinite(K[0]) and np.isfinite(K[1])) else "")

    elapsed = time.time() - t0
    print(f"\n  Computation time: {elapsed:.2f}s")

    return {
        'Ap': Ap_norm,
        'cp2': cp2,
        'K': K,
        'omega': omega,
        'distances': distances,
        'S_ent': S_ent,
        'n_significant': n_significant,
        'comm_results': comm_results,
        't_values': t_values,
    }


# ===========================================================================
# Scaling Analysis
# ===========================================================================

def scaling_analysis(all_results):
    """Analyze how modular structure scales with n and beta."""

    print(f"\n{'='*80}")
    print(f"  SCALING ANALYSIS")
    print(f"{'='*80}")

    # Organize by beta
    betas = sorted(set(beta for (n, beta) in all_results.keys()))
    ns = sorted(set(n for (n, beta) in all_results.keys()))

    # ---- omega_01 scaling with n ----
    print(f"\n  omega_01 vs n (at fixed beta):")
    print(f"  {'beta':>6s}" + "".join(f"{'n='+str(n):>14s}" for n in ns) + f"{'omega/n ratio':>16s}")

    for beta in betas:
        row = f"  {beta:6.1f}"
        ratios = []
        for n in ns:
            if (n, beta) in all_results:
                r = all_results[(n, beta)]
                om = r['omega'][0, 1] if np.isfinite(r['omega'][0, 1]) else np.nan
                row += f"{om:14.4f}"
                ratios.append(om / n if not np.isnan(om) else np.nan)
            else:
                row += f"{'---':>14s}"

        # Check if omega_01 / n is constant
        valid_ratios = [r for r in ratios if not np.isnan(r)]
        if len(valid_ratios) > 1:
            cv = np.std(valid_ratios) / np.mean(valid_ratios) if np.mean(valid_ratios) > 1e-10 else np.inf
            row += f"  CV={cv:.4f}"
            if cv < 0.05:
                row += " LINEAR"
        print(row)

    # ---- Entanglement entropy scaling ----
    print(f"\n  Entanglement entropy S vs n:")
    print(f"  {'beta':>6s}" + "".join(f"{'n='+str(n):>14s}" for n in ns))
    for beta in betas:
        row = f"  {beta:6.1f}"
        for n in ns:
            if (n, beta) in all_results:
                S = all_results[(n, beta)]['S_ent']
                row += f"{S:14.6f}"
            else:
                row += f"{'---':>14s}"
        print(row)

    # ---- Casimir proportionality vs beta ----
    print(f"\n  Casimir proportionality alpha (omega ~ alpha * dC2) vs beta:")
    for beta in betas:
        for n in ns:
            if (n, beta) in all_results:
                r = all_results[(n, beta)]
                K = r['K']
                n_sig = r['n_significant']

                omegas = []
                dC2s = []
                for p in range(min(6, n_sig)):
                    for q in range(p+1, min(6, n_sig)):
                        if np.isfinite(r['omega'][p, q]):
                            omegas.append(r['omega'][p, q])
                            dC2s.append(casimir_su2(p) - casimir_su2(q))

                if len(omegas) > 2:
                    om_arr = np.array(omegas)
                    dc_arr = np.array(dC2s)
                    if np.std(dc_arr) > 1e-10:
                        slope = np.sum(om_arr * dc_arr) / np.sum(dc_arr**2)
                        resid = om_arr - slope * dc_arr
                        R2 = 1 - np.sum(resid**2) / np.sum((om_arr - np.mean(om_arr))**2)
                        print(f"    n={n:3d}, beta={beta:5.1f}: alpha={slope:10.4f}, R^2={R2:.6f}")

    # ---- Modular speed vs beta ----
    print(f"\n  Modular distances d(0,k) vs separation k:")
    for beta in betas:
        for n in ns:
            if (n, beta) in all_results:
                r = all_results[(n, beta)]
                dists = r['distances']
                pairs = [(j, k, d) for (j, k), d in dists.items() if j == 0 and np.isfinite(d) and d < 100]
                if pairs:
                    pairs.sort(key=lambda x: x[1])
                    line = f"    n={n:3d}, beta={beta:5.1f}: "
                    for j, k, d in pairs[:6]:
                        line += f"d(0,{k})={d:.4f}  "
                    print(line)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print()
    print("=" * 80)
    print("  TOMITA-TAKESAKI MODULAR FLOW OF THE YANG-MILLS VACUUM")
    print("  SU(2) Wilson Action on n-Plaquette Chain")
    print("=" * 80)
    print()
    print("  Modular Hamiltonian K_L = -log(rho_L)")
    print("  Modular flow: sigma_t(A) = rho^{it} A rho^{-it}")
    print("  Key question: Does modular flow generate causal structure?")
    print()

    # Configuration grid
    n_values = [4, 8, 16, 32]
    beta_values = [1.0, 2.0, 5.0, 10.0]
    n_reps = 25  # Truncation in rep space

    all_results = {}

    for beta in beta_values:
        for n in n_values:
            result = analyze_single_config(n, beta, n_reps=n_reps)
            all_results[(n, beta)] = result

    # Scaling analysis
    scaling_analysis(all_results)

    # ===========================================================================
    # FINAL SUMMARY
    # ===========================================================================
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY: MODULAR FLOW OF THE YANG-MILLS VACUUM")
    print(f"{'='*80}")

    print(f"""
  1. MODULAR HAMILTONIAN K_p = n * log(A_0/A_p) + 2*log(d_0/d_p) + const
     - Dominated by the action contribution n*log(A_0/A_p)
     - Degeneracy 2*log(d_p) provides a subdominant correction
     - K grows with rep p (higher reps are "hotter")

  2. MODULAR FREQUENCIES omega_pq = K_p - K_q
     - Test: is omega_pq proportional to Casimir difference C_2(p) - C_2(q)?
     - If yes: K = alpha * C_2 = alpha * "electric field squared"
       This would mean modular time = PHYSICAL time (Bisognano-Wichmann analog)

  3. CAUSALITY from commutator growth:
     - If [sigma_t(O_j), O_k] = 0 for |t| < d(j,k)/c
       → LIGHT CONE structure from modular flow
     - This would establish emergent METRIC from the thermal state alone

  4. SCALING with n:
     - omega_01 ~ n * log(A_0/A_1): GROWS linearly with chain length
     - This means modular "temperature" ~ n (extensive)
     - The modular metric should stabilize as n → inf
""")

    # Check key results
    # Casimir test at beta=2, n=8
    if (8, 2.0) in all_results:
        r = all_results[(8, 2.0)]
        K = r['K']
        n_sig = r['n_significant']

        omegas = []
        dC2s = []
        for p in range(min(6, n_sig)):
            for q in range(p+1, min(6, n_sig)):
                if np.isfinite(r['omega'][p, q]):
                    omegas.append(r['omega'][p, q])
                    dC2s.append(casimir_su2(p) - casimir_su2(q))

        if len(omegas) > 2:
            om_arr = np.array(omegas)
            dc_arr = np.array(dC2s)
            slope = np.sum(om_arr * dc_arr) / np.sum(dc_arr**2)
            resid = om_arr - slope * dc_arr
            R2 = 1 - np.sum(resid**2) / np.sum((om_arr - np.mean(om_arr))**2)

            if R2 > 0.99:
                print(f"  *** RESULT: K_L = {slope:.4f} * C_2 + const  (R^2 = {R2:.6f}) ***")
                print(f"  *** Modular Hamiltonian IS the Casimir (= Laplacian on SU(2)) ***")
                print(f"  *** This is the 1D analog of Bisognano-Wichmann! ***")
            elif R2 > 0.9:
                print(f"  ** RESULT: K_L approximately proportional to C_2 (R^2 = {R2:.4f}) **")
                print(f"  ** Modular flow approximately = Casimir dynamics **")
            else:
                print(f"  RESULT: K_L is NOT simply proportional to C_2 (R^2 = {R2:.4f})")
                print(f"  The modular Hamiltonian has non-trivial structure beyond Casimir.")

    print(f"\n  Computation complete.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
