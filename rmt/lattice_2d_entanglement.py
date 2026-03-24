#!/usr/bin/env python3
"""
2D SU(2) Yang-Mills Entanglement Entropy via Transfer Matrix + Monte Carlo
===========================================================================

Computes entanglement entropy of the 2D SU(2) lattice gauge vacuum across
horizontal strips of an Lx x Lt lattice with periodic BC.

Two methods:
  1. Transfer matrix (exact): character expansion, analytic in reps
  2. Monte Carlo (replica trick): Renyi S_2 from swap operator on doubled system

Key result: S_EE vs boundary length |dA| tests area law, extracts G_N_eff.

SU(2) conventions (quaternion parametrisation):
  U = a0*I + i*(a1*s1 + a2*s2 + a3*s3),  |a|^2 = 1
  Wilson action: S = -beta * Sum_p Re Tr(U_p) / 2 = -beta * Sum_p a0(U_p)
  Quaternion dot: a0*b0 - a1*b1 - a2*b2 - a3*b3  (MINUS signs required)
  Action change: dS = -beta * (dot_new - dot_old)  (NOT -2*beta)

Link array layout:
  links: shape (2, Lt, Lx, 4)
  - axis 0: direction mu (0=spatial/x, 1=temporal/t)
  - axis 1: temporal coordinate t
  - axis 2: spatial coordinate x
  - axis 3: quaternion components (a0, a1, a2, a3)

  When slicing links[mu], the result has shape (Lt, Lx, 4):
  - axis 0: t,  axis 1: x,  axis 2: quaternion

Author: Grzegorz Olbryk
Date: 2026-03-24
"""

import numpy as np
import time

try:
    import cupy as cp
    GPU = True
except ImportError:
    print("CuPy not available, falling back to NumPy (slow)")
    import numpy as cp
    GPU = False


# ============================================================================
# SU(2) quaternion algebra (all vectorised, GPU-compatible)
# ============================================================================

def su2_mul(U, V):
    """Quaternion multiplication U*V for SU(2) elements."""
    a0, a1, a2, a3 = U[..., 0], U[..., 1], U[..., 2], U[..., 3]
    b0, b1, b2, b3 = V[..., 0], V[..., 1], V[..., 2], V[..., 3]
    return cp.stack([
        a0*b0 - a1*b1 - a2*b2 - a3*b3,
        a0*b1 + a1*b0 + a2*b3 - a3*b2,
        a0*b2 - a1*b3 + a2*b0 + a3*b1,
        a0*b3 + a1*b2 - a2*b1 + a3*b0
    ], axis=-1)


def su2_dag(U):
    """Hermitian conjugate U^dag = (a0, -a1, -a2, -a3)."""
    result = U.copy()
    result[..., 1:] *= -1
    return result


def su2_dot(U, V):
    """Quaternion dot product = Re Tr(U V^dag)/2 = a0*b0 - a.b."""
    return (U[..., 0]*V[..., 0] - U[..., 1]*V[..., 1]
            - U[..., 2]*V[..., 2] - U[..., 3]*V[..., 3])


def su2_trace(U):
    """Re Tr(U)/2 = a0."""
    return U[..., 0]


def random_su2_near_id(shape, eps, rng):
    """Random SU(2) near identity. eps controls spread (0=id, ~1=large)."""
    a123 = (rng.random((*shape, 3)) * 2 - 1) * eps
    norm_sq = cp.sum(a123**2, axis=-1, keepdims=True)
    # Rescale if |a123| >= 1 to keep on S^3
    too_big = (norm_sq >= 1.0).squeeze(-1)
    if cp.any(too_big):
        a123[too_big] *= 0.9 / cp.sqrt(norm_sq[too_big])
        norm_sq[too_big] = cp.sum(a123[too_big]**2, axis=-1, keepdims=True)
    a0 = cp.sqrt(cp.maximum(1.0 - norm_sq, 0.0))
    return cp.concatenate([a0, a123], axis=-1)


# ============================================================================
# 2D lattice gauge theory
# ============================================================================
#
# Plaquette at site (t,x):
#   U_p(t,x) = U_x(t,x) * U_t(t, x+1) * U_x(t+1, x)^dag * U_t(t, x)^dag
#
# For links[mu] with shape (Lt, Lx, 4):
#   axis 0 = t, axis 1 = x
#   roll(., -1, axis=0) = shift t -> t+1
#   roll(., -1, axis=1) = shift x -> x+1

def plaquettes(links, Lx, Lt):
    """All plaquettes. Returns (Lt, Lx, 4)."""
    Ux = links[0]                                    # U_x(t, x)
    Ut_xp1 = cp.roll(links[1], -1, axis=1)          # U_t(t, x+1)
    Ux_tp1_dag = su2_dag(cp.roll(links[0], -1, axis=0))  # U_x(t+1, x)^dag
    Ut_dag = su2_dag(links[1])                       # U_t(t, x)^dag
    return su2_mul(su2_mul(su2_mul(Ux, Ut_xp1), Ux_tp1_dag), Ut_dag)


def action(links, Lx, Lt, beta):
    """Wilson action S = -beta * sum_p Re Tr(U_p)/2."""
    P = plaquettes(links, Lx, Lt)
    return float(-beta * cp.sum(su2_trace(P)))


def avg_plaquette(links, Lx, Lt):
    """<Re Tr(U_p)/2> averaged over all plaquettes."""
    P = plaquettes(links, Lx, Lt)
    return float(cp.mean(su2_trace(P)))


def all_staples(links, Lx, Lt):
    """Compute staples for every link. Returns shape (2, Lt, Lx, 4).

    For link U_mu(t,x), the staple Sigma_mu(t,x) is such that:
      S_plaquettes_involving_U = -beta * su2_dot(U_mu, Sigma_mu) + const

    Each link touches 2 plaquettes (in 2D), giving forward + backward staple.

    Axis conventions for links[mu] (shape Lt, Lx, 4):
      axis 0 = t, axis 1 = x
    """
    stap = cp.zeros_like(links)

    # ---- mu=0 (spatial / x-links) ----
    # Plaquette at (t,x) [forward in t]:
    #   U_p = U_x(t,x) * U_t(t,x+1) * U_x(t+1,x)^dag * U_t(t,x)^dag
    #   Staple for U_x(t,x): Sigma_fwd = U_t(t,x+1) * U_x(t+1,x)^dag * U_t(t,x)^dag
    # But su2_dot(U, Sigma) = Re Tr(U Sigma^dag)/2, and
    # Re Tr(U_p)/2 = Re Tr(U_x * [U_t(t,x+1) * U_x(t+1,x)^dag * U_t(t,x)^dag])/2
    # = su2_dot(U_x, su2_dag(U_t(t,x+1) * U_x(t+1,x)^dag * U_t(t,x)^dag))
    # Wait: su2_dot(A, B) = Re Tr(A B^dag)/2.
    # We need: Re Tr(A * C)/2 where C = U_t(t,x+1) * U_x(t+1,x)^dag * U_t(t,x)^dag
    # = su2_dot(A, su2_dag(C)) = su2_dot(A, U_t(t,x) * U_x(t+1,x) * U_t(t,x+1)^dag)
    #
    # Hmm, let's just define staple = C directly (not C^dag), so that
    # Re Tr(U_p)/2 = Re Tr(U_x * staple)/2 != su2_dot.
    # Actually: Re Tr(A*B)/2 = a0*b0 - a1*b1 - ... = su2_dot(A, su2_dag(B))? No.
    # su2_dot(A, B) = a0*b0 - a1*b1 - a2*b2 - a3*b3 = Re Tr(A * B^dag)/2.
    # So Re Tr(A*C)/2 = su2_dot(A, su2_dag(C))?
    # su2_dot(A, C^dag) = a0*c0 + a1*c1 + a2*c2 + a3*c3 -- NO that's wrong.
    # su2_dag(C) = (c0, -c1, -c2, -c3).
    # su2_dot(A, su2_dag(C)) = a0*c0 - a1*(-c1) - a2*(-c2) - a3*(-c3)
    #                        = a0*c0 + a1*c1 + a2*c2 + a3*c3
    # That's the EUCLIDEAN dot, not the quaternion dot.
    # But Re Tr(A*C)/2 for SU(2): A*C = (a0c0-a.c, ...), so Re Tr/2 = a0*c0 - a.c
    # = quaternion dot of A and C = su2_dot(A, C).
    #
    # So: Re Tr(U_x * staple)/2 = su2_dot(U_x, staple) where staple IS the product
    # of the other 3 links in order.

    Ut = links[1]                                    # U_t(t, x)
    Ut_xp1 = cp.roll(links[1], -1, axis=1)          # U_t(t, x+1)
    Ux_tp1 = cp.roll(links[0], -1, axis=0)          # U_x(t+1, x)
    Ut_tm1 = cp.roll(links[1], 1, axis=0)           # U_t(t-1, x)
    Ut_tm1_xp1 = cp.roll(Ut_tm1, -1, axis=1)       # U_t(t-1, x+1)
    Ux_tm1 = cp.roll(links[0], 1, axis=0)           # U_x(t-1, x)

    # Forward staple (plaquette at (t,x)):
    # C_fwd = U_t(t, x+1) * U_x(t+1, x)^dag * U_t(t, x)^dag
    sf = su2_mul(Ut_xp1, su2_mul(su2_dag(Ux_tp1), su2_dag(Ut)))

    # Backward staple (plaquette at (t-1, x)):
    # U_p(t-1,x) = U_x(t-1,x) * U_t(t-1,x+1) * U_x(t,x)^dag * U_t(t-1,x)^dag
    # For U_x(t,x)^dag: the plaquette contains U_x(t,x)^dag as the 3rd factor.
    # Re Tr(U_p) = Re Tr(... * U_x(t,x)^dag * ...) -- need to cycle.
    # Re Tr(U_x(t,x)^dag * U_t(t-1,x)^dag * U_x(t-1,x) * U_t(t-1,x+1))
    # = su2_dot(U_x(t,x)^dag, C_bwd) where C_bwd = U_t(t-1,x)^dag * U_x(t-1,x) * U_t(t-1,x+1)
    # But we want su2_dot(U_x(t,x), staple_bwd).
    # su2_dot(U^dag, C) = Re Tr(U^dag * C)/2 = Re Tr(C * U^dag)/2 = su2_dot(C, U^dag)
    # Hmm. Actually su2_dot is symmetric? Let me check:
    # su2_dot(A,B) = a0b0 - a1b1 - a2b2 - a3b3 = su2_dot(B,A). YES, symmetric.
    # And Re Tr(AB)/2 = su2_dot(A, B) (for SU(2) quaternions).
    # But Re Tr(A^dag B)/2 = su2_dot(A^dag, B) = a0b0 + a1b1 + a2b2 + a3b3 (Euclidean).
    # So Re Tr(U_p(t-1,x))/2 = Re Tr(U_x(t-1,x) U_t(t-1,x+1) U_x(t,x)^dag U_t(t-1,x)^dag)/2
    # = su2_dot(U_x(t,x)^dag, U_t(t-1,x)^dag * U_x(t-1,x) * U_t(t-1,x+1))
    # We need this expressed as f(U_x(t,x)):
    # su2_dot(U^dag, C) = u0*c0 + u1*c1 + u2*c2 + u3*c3 -- not the same form.
    #
    # Let me use a DIFFERENT approach. The total action change when replacing
    # U_x(t,x) -> U_x'(t,x) is:
    # dS = -beta * [su2_trace(U_p_new(t,x)) - su2_trace(U_p_old(t,x))
    #              + su2_trace(U_p_new(t-1,x)) - su2_trace(U_p_old(t-1,x))]
    #
    # For plaquette (t,x): U_p = U_x(t,x) * rest_fwd
    # su2_trace(U_p) = su2_dot(U_x(t,x), rest_fwd)  [since su2_trace = component 0 of product]
    # Wait no: su2_trace(A*B) = (AB)[0] = a0b0 - a1b1 - a2b2 - a3b3 = su2_dot(A, B).
    # So su2_trace(U_x * rest) = su2_dot(U_x, rest).
    # And dS_fwd = -beta * (su2_dot(U_x_new, rest_fwd) - su2_dot(U_x_old, rest_fwd))
    #
    # For plaquette (t-1,x): U_p = U_x(t-1,x) * U_t(t-1,x+1) * U_x(t,x)^dag * U_t(t-1,x)^dag
    # Cycle: = U_t(t-1,x)^dag * U_x(t-1,x) * U_t(t-1,x+1) * U_x(t,x)^dag
    # su2_trace(U_p) = su2_trace(rest_bwd * U_x(t,x)^dag) = su2_dot(rest_bwd, U_x(t,x)^dag)
    # where rest_bwd = U_t(t-1,x)^dag * U_x(t-1,x) * U_t(t-1,x+1)
    #
    # su2_dot(C, U^dag) = c0*u0 + c1*u1 + c2*u2 + c3*u3 (Euclidean dot!)
    # This is NOT the same as su2_dot(U, something).
    #
    # So we can't simply add the two staples. We need:
    # dS = -beta * [su2_dot(U_new, Sigma_fwd) - su2_dot(U_old, Sigma_fwd)
    #             + eucl_dot(U_new, Sigma_bwd) - eucl_dot(U_old, Sigma_bwd)]
    # where Sigma_bwd involves the daggered link.
    #
    # Actually, let me reconsider. The issue is that in plaquette (t-1,x),
    # U_x(t,x) appears DAGGERED. Let me write:
    #
    # Re Tr(U_p(t-1,x))/2 = Re Tr(... U_x(t,x)^dag ...)/2
    #
    # Using cyclicity of trace:
    # = Re Tr(U_x(t,x)^dag * V)/2 where V = U_t(t-1,x)^dag * U_x(t-1,x) * U_t(t-1,x+1)
    # = Re Tr(V * U_x^dag)/2 = Re Tr((U_x * V^dag)^dag)/2 = Re Tr(U_x * V^dag)/2
    #   (since Re Tr(M) = Re Tr(M^dag) for any matrix)
    # = su2_dot(U_x, V^dag)
    # where V^dag = U_t(t-1,x+1)^dag * U_x(t-1,x)^dag * U_t(t-1,x)
    #
    # So the backward staple for su2_dot formulation is:
    # Sigma_bwd = V^dag = U_t(t-1,x+1)^dag * U_x(t-1,x)^dag * U_t(t-1,x)

    sb = su2_mul(su2_dag(Ut_tm1_xp1), su2_mul(su2_dag(Ux_tm1), Ut_tm1))

    stap[0] = sf + sb

    # ---- mu=1 (temporal / t-links) ----
    # Plaquette at (t,x) [forward in x]:
    #   U_p = U_x(t,x) * U_t(t,x+1) * U_x(t+1,x)^dag * U_t(t,x)^dag
    #   U_t(t,x) appears daggered (4th factor).
    #   Cycle: Re Tr(U_t(t,x)^dag * U_x(t,x) * U_t(t,x+1) * U_x(t+1,x)^dag)/2
    #   = Re Tr(U_t(t,x) * [U_x(t+1,x) * U_t(t,x+1)^dag * U_x(t,x)^dag])/2
    #   = su2_dot(U_t(t,x), [U_x(t+1,x) * U_t(t,x+1)^dag * U_x(t,x)^dag])
    # Hmm wait, let me redo. Using Re Tr(A^dag B)/2 = eucl_dot but Re Tr(AB)/2 = su2_dot.
    # Re Tr(U_t^dag * C)/2 where C = U_x(t,x) * U_t(t,x+1) * U_x(t+1,x)^dag
    # = Re Tr(C * U_t^dag)/2 = Re Tr((U_t * C^dag)^dag)/2 = Re Tr(U_t * C^dag)/2
    # = su2_dot(U_t, C^dag)
    # C^dag = U_x(t+1,x) * U_t(t,x+1)^dag * U_x(t,x)^dag

    Ux = links[0]                                    # U_x(t, x)
    Ux_tp1 = cp.roll(links[0], -1, axis=0)          # U_x(t+1, x)
    Ut_xp1 = cp.roll(links[1], -1, axis=1)          # U_t(t, x+1)
    Ux_xm1 = cp.roll(links[0], 1, axis=1)           # U_x(t, x-1)
    Ux_tp1_xm1 = cp.roll(Ux_tp1, 1, axis=1)        # U_x(t+1, x-1)
    Ut_xm1 = cp.roll(links[1], 1, axis=1)           # U_t(t, x-1)

    # Forward staple from plaquette (t, x):
    # Sigma_fwd = C^dag = U_x(t+1,x) * U_t(t,x+1)^dag * U_x(t,x)^dag
    sf1 = su2_mul(Ux_tp1, su2_mul(su2_dag(Ut_xp1), su2_dag(Ux)))

    # Backward staple from plaquette (t, x-1):
    # U_p(t,x-1) = U_x(t,x-1) * U_t(t,x) * U_x(t+1,x-1)^dag * U_t(t,x-1)^dag
    # U_t(t,x) is the 2nd factor.
    # Cycle: Re Tr(U_t(t,x) * U_x(t+1,x-1)^dag * U_t(t,x-1)^dag * U_x(t,x-1))/2
    # = su2_dot(U_t(t,x), [U_x(t+1,x-1)^dag * U_t(t,x-1)^dag * U_x(t,x-1)])
    # Wait, but the product after U_t(t,x) is:
    # U_x(t+1,x-1)^dag * U_t(t,x-1)^dag * U_x(t,x-1)
    # and Re Tr(A * B)/2 = su2_dot(A, B).
    # So Sigma = U_x(t+1,x-1)^dag * U_t(t,x-1)^dag * U_x(t,x-1)
    sb1 = su2_mul(su2_dag(Ux_tp1_xm1), su2_mul(su2_dag(Ut_xm1), Ux_xm1))

    stap[1] = sf1 + sb1

    return stap


def metropolis_sweep(links, Lx, Lt, beta, eps, rng):
    """One full Metropolis sweep with checkerboard decomposition.
    Returns acceptance rate."""
    accepted = 0
    total = 0

    for mu in [0, 1]:
        for parity in [0, 1]:
            # Compute staples for this mu
            stap = all_staples(links, Lx, Lt)

            # Parity mask: (t + x) % 2 == parity
            t_idx = cp.arange(Lt)[:, None]
            x_idx = cp.arange(Lx)[None, :]
            mask = ((t_idx + x_idx) % 2 == parity)  # (Lt, Lx)

            n_sites = int(cp.sum(mask))
            if n_sites == 0:
                continue

            # Propose R*U for all sites
            R = random_su2_near_id((Lt, Lx), eps, rng)
            U_new = su2_mul(R, links[mu])

            # Action change: dS = -beta * (dot_new - dot_old)
            dot_old = su2_dot(links[mu], stap[mu])
            dot_new = su2_dot(U_new, stap[mu])
            dS = -beta * (dot_new - dot_old)

            # Accept/reject
            accept = (dS <= 0) | (cp.log(rng.random((Lt, Lx))) < -dS)
            accept = accept & mask

            links[mu] = cp.where(accept[..., None], U_new, links[mu])

            accepted += int(cp.sum(accept))
            total += n_sites

    return accepted / max(total, 1)


# ============================================================================
# PART 1: EXACT TRANSFER MATRIX (character expansion)
# ============================================================================
#
# 2D SU(2) Yang-Mills on a torus (periodic Lx x Lt).
#
# Character expansion: the partition function is
#   Z = sum_j [I_{2j+1}(beta)]^{Lx*Lt}   (for genus-1 torus)
# where j = 0, 1/2, 1, 3/2, ... and I_n is modified Bessel of 1st kind.
#
# The transfer matrix in the gauge-invariant sector is diagonal:
#   T_j = [I_{2j+1}(beta)]^Lx
# with the ground state at j=0 (since I_1 > I_n for n>1 at any beta>0).
#
# Gauge-invariant state in rep j (spatial Wilson loop):
#   |Psi_j> = (1/sqrt(2j+1)) sum_{m=-j}^{j} |j,m>^{otimes Lx}
# This is a GHZ state. Tracing over ANY subregion gives rho_A with
# eigenvalues 1/(2j+1) (each with multiplicity 1), so:
#   S_EE(j) = log(2j+1) -- TOPOLOGICAL, independent of region size!
#
# At finite temperature (finite Lt), the thermal state mixes reps:
#   rho = sum_j p_j |Psi_j><Psi_j|,  p_j = T_j^Lt / Z
#
# Total entanglement:
#   S_EE = H({p_j}) + sum_j p_j log(2j+1)
# = classical mixing entropy + average topological entropy.

def exact_entanglement(Lx, Lt, beta, j_max=8):
    """Exact entanglement entropy for 2D SU(2) YM on Lx x Lt torus.

    Returns S_EE, H_classical, S_topological, p_j array, j_vals.
    """
    from scipy.special import iv

    num_j = int(2 * j_max) + 1
    j_vals = np.arange(num_j) / 2.0
    d_j = (2 * j_vals + 1).astype(int)

    # Transfer eigenvalue per rep: T_j = I_{2j+1}(beta)^Lx
    # Thermal weight: w_j = T_j^Lt = I_{2j+1}(beta)^{Lx*Lt}
    log_w = np.zeros(num_j)
    for k, j in enumerate(j_vals):
        n = int(2*j + 1)
        bess = float(iv(n, beta))
        if bess > 0:
            log_w[k] = Lx * Lt * np.log(bess)
        else:
            log_w[k] = -1e30

    # Normalise in log space
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    Z = np.sum(w)
    p_j = w / Z

    # Entanglement entropy
    H_cl = 0.0
    S_top = 0.0
    for k in range(num_j):
        if p_j[k] > 1e-30:
            H_cl -= p_j[k] * np.log(p_j[k])
            S_top += p_j[k] * np.log(d_j[k])

    S_EE = H_cl + S_top
    return S_EE, H_cl, S_top, p_j, j_vals


# ============================================================================
# PART 2: MONTE CARLO REPLICA TRICK for Renyi S_2
# ============================================================================
#
# Tr(rho_A^2) = Z_swap / Z^2 where Z_swap is the partition function
# on a 2-sheeted branched cover with branch cut at region A.
#
# Measured by: run two replicas, swap their links in region A at t=0,
# measure exp(-Delta S) where Delta S = S_swapped - S_unswapped.

def mc_renyi_2(links1, links2, Lx, Lt, region_A_sites, beta,
               n_meas, eps, rng):
    """Measure Renyi S_2 = -log Tr(rho_A^2) via replica trick.

    Thermalises between measurements. Returns S_2, error.
    """
    ratios = []

    for _ in range(n_meas):
        # Evolve both replicas
        for _ in range(10):
            metropolis_sweep(links1, Lx, Lt, beta, eps, rng)
            metropolis_sweep(links2, Lx, Lt, beta, eps, rng)

        # Actions before swap
        S1_before = action(links1, Lx, Lt, beta)
        S2_before = action(links2, Lx, Lt, beta)

        # Swap spatial links at t=0 in region A
        links1_s = links1.copy()
        links2_s = links2.copy()
        for x in region_A_sites:
            for mu in [0, 1]:
                tmp = links1_s[mu, 0, x].copy()
                links1_s[mu, 0, x] = links2_s[mu, 0, x].copy()
                links2_s[mu, 0, x] = tmp

        # Actions after swap
        S1_after = action(links1_s, Lx, Lt, beta)
        S2_after = action(links2_s, Lx, Lt, beta)

        dS = (S1_after + S2_after) - (S1_before + S2_before)
        ratios.append(np.exp(-dS))

    ratios = np.array(ratios)
    tr_rho2 = np.mean(ratios)
    tr_rho2_err = np.std(ratios) / np.sqrt(len(ratios))

    if tr_rho2 > 1e-10:
        S2 = -np.log(tr_rho2)
        S2_err = tr_rho2_err / tr_rho2
    else:
        S2 = float('inf')
        S2_err = float('inf')

    return S2, S2_err, tr_rho2, tr_rho2_err


# ============================================================================
# MAIN ANALYSES
# ============================================================================

def sanity_check():
    """Verify MC plaquette against exact SU(2) result."""
    from scipy.special import iv

    print("=" * 72)
    print("SANITY CHECK: PLAQUETTE EXPECTATION VALUE")
    print("=" * 72)

    rng = cp.random.default_rng(42)
    Lx, Lt = 6, 6
    betas = [1.0, 2.0, 4.0, 8.0]
    n_therm = 500
    n_meas = 200
    eps = 0.3

    # First verify action/staple on trivial config
    links_test = cp.zeros((2, Lt, Lx, 4))
    links_test[..., 0] = 1.0
    S_cold = action(links_test, Lx, Lt, 2.0)
    print(f"\nCold start action (beta=2, {Lx}x{Lt}): {S_cold:.4f}")
    print(f"  Expected: -beta * Lx * Lt * 1.0 = {-2.0 * Lx * Lt:.1f}")

    P_cold = avg_plaquette(links_test, Lx, Lt)
    print(f"  Cold plaquette: {P_cold:.4f} (should be 1.0)")

    # Verify staple for cold config
    stap = all_staples(links_test, Lx, Lt)
    print(f"  Staple[0] at (0,0): {stap[0, 0, 0].get()}")
    print(f"  Should be ~(2, 0, 0, 0) (sum of 2 identity staples)")

    print(f"\nLattice: {Lx}x{Lt}, thermalisation: {n_therm}, measurements: {n_meas}")
    print(f"{'beta':>6s} {'<plaq> MC':>12s} {'<plaq> exact':>14s} {'|diff|':>10s} {'acc':>6s}")

    for beta in betas:
        exact = float(iv(2, beta)) / float(iv(1, beta))

        links = cp.zeros((2, Lt, Lx, 4))
        links[..., 0] = 1.0

        # Thermalise
        for _ in range(n_therm):
            metropolis_sweep(links, Lx, Lt, beta, eps, rng)

        # Measure
        plaqs = []
        accs = []
        for _ in range(n_meas):
            acc = metropolis_sweep(links, Lx, Lt, beta, eps, rng)
            plaqs.append(avg_plaquette(links, Lx, Lt))
            accs.append(acc)

        mc = np.mean(plaqs)
        mc_err = np.std(plaqs) / np.sqrt(len(plaqs))
        print(f"{beta:6.1f} {mc:12.6f} {exact:14.6f} {abs(mc-exact):10.6f} {np.mean(accs):6.3f}")


def exact_analysis():
    """Run exact transfer matrix entanglement for various parameters."""
    print("\n" + "=" * 72)
    print("EXACT TRANSFER MATRIX: THERMAL ENTANGLEMENT")
    print("=" * 72)

    betas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    Lxs = [4, 6, 8]
    Lts = [4, 8, 16]
    j_max = 8

    for Lt in Lts:
        print(f"\n--- Lt = {Lt} (T = 1/(a*{Lt})) ---")
        print(f"{'beta':>6s}", end="")
        for Lx in Lxs:
            print(f"  {'Lx='+str(Lx):>12s}", end="")
        print(f"  {'j_dom':>6s} {'p_dom':>8s}")

        for beta in betas:
            print(f"{beta:6.1f}", end="")
            for Lx in Lxs:
                S, H, St, pj, jv = exact_entanglement(Lx, Lt, beta, j_max)
                print(f"  {S:12.6f}", end="")
            # Show dominant rep for largest Lx
            S, H, St, pj, jv = exact_entanglement(Lxs[-1], Lt, beta, j_max)
            jd = jv[np.argmax(pj)]
            pd = np.max(pj)
            print(f"  {jd:6.1f} {pd:8.4f}")

    # Key physics
    print("\n--- AREA LAW TEST ---")
    print("For 2D SU(2) YM, S_EE is INDEPENDENT of region geometry:")
    print("  S_EE(strip of width w) = S_EE(strip of width w') for all w, w'")
    print("  because the gauge-invariant state |Psi_j> is GHZ-like.")
    print("  Entanglement is TOPOLOGICAL, not geometric.")
    print("  This means: area law S ~ |dA| is trivially satisfied (|dA| = const = 2)")
    print("  but there is no geometric content to extract.")


def newton_constant():
    """Extract effective G_N from entanglement."""
    print("\n" + "=" * 72)
    print("EFFECTIVE NEWTON'S CONSTANT: G_N_eff = |dA| / (4 S_EE)")
    print("=" * 72)

    betas = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    Lx = 8
    Lt = 4
    dA = 2  # boundary length for 1D ring
    j_max = 8

    print(f"\nLx={Lx}, Lt={Lt}, |dA|={dA}")
    print(f"{'beta':>6s} {'S_EE':>10s} {'G_N_eff':>10s} {'H_cl':>10s} {'S_top':>10s}")

    for beta in betas:
        S, H, St, pj, jv = exact_entanglement(Lx, Lt, beta, j_max)
        G = dA / (4 * S) if S > 1e-10 else float('inf')
        print(f"{beta:6.1f} {S:10.6f} {G:10.4f} {H:10.6f} {St:10.6f}")

    print("\nPhysics:")
    print("  Large beta (weak coupling): j=0 dominates, S_EE -> 0, G_N -> inf")
    print("  Small beta (strong coupling): many reps mix, S_EE > 0, G_N finite")
    print("  BUT 2D YM is solvable & topological: no emergent gravity")
    print("  G_N_eff has no geometric meaning in 2D")


def mc_entanglement():
    """MC replica-trick measurement of Renyi S_2."""
    print("\n" + "=" * 72)
    print("MONTE CARLO: RENYI S_2 VIA REPLICA TRICK")
    print("=" * 72)

    rng = cp.random.default_rng(12345)
    Lx, Lt = 6, 6
    betas = [2.0, 4.0, 8.0]
    n_therm = 500
    n_meas = 100
    eps = 0.3

    print(f"\nLattice: {Lx}x{Lt}, thermalise: {n_therm}, measure: {n_meas}")

    for beta in betas:
        print(f"\n--- beta = {beta:.1f} ---")

        # Init cold start
        links1 = cp.zeros((2, Lt, Lx, 4))
        links1[..., 0] = 1.0
        links2 = links1.copy()

        # Thermalise
        t0 = time.time()
        for _ in range(n_therm):
            metropolis_sweep(links1, Lx, Lt, beta, eps, rng)
            metropolis_sweep(links2, Lx, Lt, beta, eps, rng)
        print(f"  Thermalised in {time.time()-t0:.1f}s, "
              f"<plaq>={avg_plaquette(links1, Lx, Lt):.4f}")

        # Measure S_2 for strips of width 1, 2, 3
        print(f"  {'width':>5s} {'|dA|':>5s} {'S_2':>10s} {'err':>8s} {'Tr rho^2':>10s}")
        for w in range(1, Lx // 2 + 1):
            region_A = list(range(w))
            S2, S2_err, tr2, tr2_err = mc_renyi_2(
                links1, links2, Lx, Lt, region_A, beta, n_meas, eps, rng)
            boundary = 2  # ring: always 2 boundary points
            print(f"  {w:5d} {boundary:5d} {S2:10.4f} {S2_err:8.4f} {tr2:10.6f}")

        # Exact comparison
        S_exact, _, _, _, _ = exact_entanglement(Lx, Lt, beta, j_max=8)
        print(f"  Exact S_EE = {S_exact:.6f}")


def area_law_test():
    """Test S_EE vs strip width (should be constant for topological theory)."""
    print("\n" + "=" * 72)
    print("AREA LAW vs VOLUME LAW TEST (MC)")
    print("=" * 72)

    rng = cp.random.default_rng(77777)
    Lx, Lt = 6, 4
    betas = [2.0, 8.0]
    n_therm = 500
    n_meas = 100
    eps = 0.3

    print(f"\nLattice: {Lx}x{Lt}")
    print("If topological: S_2 independent of strip width w")
    print("If volume law: S_2 ~ w")

    for beta in betas:
        print(f"\n--- beta = {beta:.1f} ---")
        links1 = cp.zeros((2, Lt, Lx, 4))
        links1[..., 0] = 1.0
        links2 = links1.copy()

        for _ in range(n_therm):
            metropolis_sweep(links1, Lx, Lt, beta, eps, rng)
            metropolis_sweep(links2, Lx, Lt, beta, eps, rng)

        plaq = avg_plaquette(links1, Lx, Lt)
        print(f"  <plaq> = {plaq:.4f}")

        s2_list = []
        for w in range(1, Lx // 2 + 1):
            S2, S2_err, _, _ = mc_renyi_2(
                links1, links2, Lx, Lt, list(range(w)), beta, n_meas, eps, rng)
            s2_list.append(S2)
            print(f"    w={w}: S_2 = {S2:.4f} +/- {S2_err:.4f}")

        if len(s2_list) >= 2:
            slope = (s2_list[-1] - s2_list[0]) / (len(s2_list) - 1)
            print(f"  Slope dS_2/dw = {slope:.4f}", end="")
            if abs(slope) < 0.15:
                print(" --> TOPOLOGICAL (flat)")
            else:
                print(" --> NON-FLAT")


# ============================================================================

if __name__ == "__main__":
    print("2D SU(2) YANG-MILLS ENTANGLEMENT STRUCTURE")
    print("=" * 72)
    print(f"GPU: {'CuPy' if GPU else 'NumPy fallback'}\n")

    # 1. Sanity check
    sanity_check()

    # 2. Exact analysis
    exact_analysis()

    # 3. Newton's constant
    newton_constant()

    # 4. MC replica trick
    mc_entanglement()

    # 5. Area law test
    area_law_test()

    # Final summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("""
1. TOPOLOGICAL ENTANGLEMENT: 2D SU(2) YM ground state has S_EE = 0.
   At finite T, S_EE = H({p_j}) + sum_j p_j log(2j+1) is purely
   topological (independent of region geometry).

2. GHZ STRUCTURE: gauge-invariant state in rep j is
   |Psi_j> = (1/sqrt(2j+1)) sum_m |j,m>^{otimes L}
   giving S_EE(j) = log(2j+1) independent of bipartition.

3. AREA LAW: trivially satisfied (|dA| = 2 for any strip on a ring),
   but carries no geometric information.

4. G_N_eff: formally divergent at T=0 (S=0), finite at T>0 but has
   no geometric interpretation in 2D.

5. TRANSITION TO GEOMETRY: requires d >= 3 where gauge theory has
   local propagating degrees of freedom. The 2D -> 3D transition
   is where entanglement becomes geometric (area law with non-trivial
   proportionality constant).
""")
