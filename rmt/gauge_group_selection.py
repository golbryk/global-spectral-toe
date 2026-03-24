"""
Gauge Group Selection via Convexity Growth Constant
====================================================

Tests whether the RG convexity constant c in the inequality
    g^2_{k+1} >= g^2_k + c * g^4_k
selects the Standard Model gauge group SU(3) x SU(2) x U(1).

The variance of the normalized plaquette under Wilson action drives RG flow:
    c_eff(G) = n_p * d * ln(L_b) * Var(Re Tr U_p / dim)
             = 6 * 4 * ln(2) * Var   [4D, blocking factor L_b=2]

Exact formulas:
  U(1):  Bessel functions I_n(beta)
  SU(N): Toeplitz determinant det[I_{i-j}(beta)], mpmath high precision
  SO(N): Haar MC on GPU (CuPy) with importance sampling

Author: Grzegorz Olbryk
Date: 2026-03-24
"""

import sys
import os
os.environ.setdefault('CUDA_PATH', '/usr/local/cuda-13.1')
os.environ['PATH'] = os.environ.get('PATH', '') + ':/usr/local/cuda-13.1/bin'

import numpy as np
from scipy.special import iv as bessel_iv
import mpmath
import cupy as cp
import time

# Flush all prints immediately
def pr(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)

# 30 decimal digits is enough for N<=10, beta<=100
mpmath.mp.dps = 30

dev = cp.cuda.runtime.getDeviceProperties(0)
pr(f"GPU: {dev['name'].decode()}, VRAM: {dev['totalGlobalMem']/1e9:.1f} GB")

# ==========================================================================
# EXACT FORMULAS
# ==========================================================================

def u1_plaquette_stats(beta):
    """U(1): <cos theta> and Var(cos theta) under exp(beta cos theta)."""
    b = mpmath.mpf(beta)
    I0 = mpmath.besseli(0, b)
    I1 = mpmath.besseli(1, b)
    I2 = mpmath.besseli(2, b)
    mean = float(I1 / I0)
    mean_sq = float(mpmath.mpf('0.5') + I2 / (2 * I0))
    var = mean_sq - mean**2
    return mean, var


# Cache for Toeplitz log-partition values
_logZ_cache = {}

def sun_log_partition(N, beta):
    """
    SU(N) partition function: ln det[I_{i-j}(beta)]_{i,j=0..N-1}
    Cached and computed with mpmath.
    """
    key = (N, f"{beta:.15e}")
    if key in _logZ_cache:
        return _logZ_cache[key]

    b = mpmath.mpf(beta)
    mat = mpmath.matrix(N, N)
    for i in range(N):
        for j in range(N):
            mat[i, j] = mpmath.besseli(i - j, b)
    det_val = mpmath.det(mat)
    if det_val <= 0:
        result = float('-inf')
    else:
        result = float(mpmath.log(det_val))
    _logZ_cache[key] = result
    return result


def sun_plaquette_stats(N, beta):
    """
    SU(N) Wilson plaquette stats via Toeplitz determinant.
    <Re Tr U / N> = (1/N) d/d(beta) ln Z
    Var(Re Tr U / N) = (1/N^2) d^2/d(beta)^2 ln Z
    Central finite differences with adaptive step.
    """
    dbeta = max(abs(beta) * 1e-6, 1e-8)

    logZ_p = sun_log_partition(N, beta + dbeta)
    logZ_m = sun_log_partition(N, beta - dbeta)
    logZ_0 = sun_log_partition(N, beta)

    d1 = (logZ_p - logZ_m) / (2 * dbeta)
    d2 = (logZ_p - 2 * logZ_0 + logZ_m) / (dbeta**2)

    mean = d1 / N
    var = d2 / (N**2)
    return mean, var


def son_plaquette_stats_gpu(N, beta, n_samples=200000):
    """SO(N) plaquette stats via Haar MC on GPU with Boltzmann reweighting."""
    rng = cp.random.default_rng(seed=42)
    batch = min(n_samples, 40000)
    n_batches = (n_samples + batch - 1) // batch
    all_traces = []

    for b_idx in range(n_batches):
        bs = min(batch, n_samples - b_idx * batch)
        if bs <= 0:
            break
        U_prod = cp.eye(N, dtype=cp.float64).reshape(1, N, N).repeat(bs, axis=0)
        for _ in range(4):
            G = rng.standard_normal((bs, N, N), dtype=cp.float64)
            Q, R = cp.linalg.qr(G)
            d = cp.sign(cp.diagonal(R, axis1=1, axis2=2))
            Q = Q * d[:, cp.newaxis, :]
            dets = cp.linalg.det(Q)
            Q[dets < 0, :, 0] *= -1
            U_prod = cp.matmul(U_prod, Q)
        all_traces.append(cp.trace(U_prod, axis1=1, axis2=2).real.get())

    traces = np.concatenate(all_traces)
    x = traces / N
    log_w = beta * traces
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    Z = np.mean(w)
    mean = np.mean(w * x) / Z
    mean_sq = np.mean(w * x**2) / Z
    var = mean_sq - mean**2
    return float(mean), float(var)


# ==========================================================================
# CONVEXITY CONSTANT
# ==========================================================================

def c_eff(var_plaq):
    """c_eff = 6 * 4 * ln(2) * Var (4D, L_b=2)."""
    return 6 * 4 * np.log(2) * var_plaq


def casimir_fund(N):
    """Quadratic Casimir for fundamental of SU(N)."""
    return (N**2 - 1) / (2.0 * N)


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    t_start = time.time()

    pr("\n" + "=" * 80)
    pr("GAUGE GROUP SELECTION VIA CONVEXITY GROWTH CONSTANT c(G)")
    pr("=" * 80)

    g2_values = [0.5, 1.0, 2.0, 4.0]

    simple_groups = [
        ('U(1)',  1, 'U1'),
        ('SU(2)', 2, 'SU'),
        ('SU(3)', 3, 'SU'),
        ('SU(4)', 4, 'SU'),
        ('SU(5)', 5, 'SU'),
        ('SU(6)', 6, 'SU'),
    ]

    all_results = {}

    # ==================================================================
    # PART 1: Simple groups
    # ==================================================================
    pr("\n" + "-" * 80)
    pr("PART 1: Simple gauge groups (exact, mpmath 30-digit)")
    pr("-" * 80)

    for g2 in g2_values:
        pr(f"\n  g^2 = {g2}")
        pr(f"  {'Group':>8s}  {'beta':>8s}  {'<plaq>':>10s}  {'Var(plaq)':>14s}  "
           f"{'c_eff':>14s}  {'k*(g0=0.1)':>12s}")
        pr(f"  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*14}  {'-'*12}")

        for name, N, gtype in simple_groups:
            if gtype == 'U1':
                beta = 1.0 / g2
            else:
                beta = 2.0 * N / g2

            if gtype == 'U1':
                mean, var = u1_plaquette_stats(beta)
            else:
                mean, var = sun_plaquette_stats(N, beta)

            c = c_eff(var)
            k_star = 1.0 / (c * 0.1) if c > 0 else float('inf')

            all_results[(name, g2)] = {'beta': beta, 'mean': mean, 'var': var,
                                       'c_eff': c, 'k_star': k_star}

            pr(f"  {name:>8s}  {beta:8.2f}  {mean:10.6f}  {var:14.6e}  "
               f"{c:14.6e}  {k_star:12.1f}")

    pr(f"\n  [Part 1 took {time.time()-t_start:.1f}s]")

    # ==================================================================
    # PART 2: Product groups (common coupling)
    # ==================================================================
    pr("\n" + "-" * 80)
    pr("PART 2: Product groups at common coupling (unification scenario)")
    pr("-" * 80)

    for g2 in g2_values:
        pr(f"\n  g^2 = {g2}")

        # SM = SU(3) x SU(2) x U(1)
        sm_var = sum(all_results[(s, g2)]['var'] for s in ['SU(3)', 'SU(2)', 'U(1)'])
        sm_c = c_eff(sm_var)

        # Pati-Salam = SU(4) x SU(2) x SU(2)
        ps_var = all_results[('SU(4)', g2)]['var'] + 2 * all_results[('SU(2)', g2)]['var']
        ps_c = c_eff(ps_var)

        # Trinification-like = SU(3) x SU(3) x SU(3)
        tri_var = 3 * all_results[('SU(3)', g2)]['var']
        tri_c = c_eff(tri_var)

        groups = [
            ('SM: SU(3)xSU(2)xU(1)', sm_var, sm_c),
            ('PS: SU(4)xSU(2)^2', ps_var, ps_c),
            ('Tri: SU(3)^3', tri_var, tri_c),
        ]
        for name, N, gtype in simple_groups:
            r = all_results[(name, g2)]
            groups.append((name, r['var'], r['c_eff']))

        groups.sort(key=lambda x: x[2], reverse=True)

        pr(f"  {'Rank':>4s}  {'Group':>28s}  {'Var_total':>14s}  {'c_eff':>14s}  {'k*':>10s}")
        pr(f"  {'-'*4}  {'-'*28}  {'-'*14}  {'-'*14}  {'-'*10}")

        for rank, (gname, var, c) in enumerate(groups, 1):
            k = 1.0 / (c * 0.1) if c > 0 else float('inf')
            marker = "  <--" if 'SM' in gname else ""
            pr(f"  {rank:4d}  {gname:>28s}  {var:14.6e}  {c:14.6e}  {k:10.1f}{marker}")

    # ==================================================================
    # PART 3: Physical couplings at M_Z
    # ==================================================================
    pr("\n" + "-" * 80)
    pr("PART 3: SM at physical couplings (M_Z scale)")
    pr("-" * 80)

    phys = [
        ('SU(3)_c', 3, 'SU', 1.484),
        ('SU(2)_L', 2, 'SU', 0.424),
        ('U(1)_Y',  1, 'U1', 0.1278),
    ]

    pr(f"\n  {'Sector':>10s}  {'g^2':>8s}  {'beta':>8s}  {'<plaq>':>10s}  "
       f"{'Var':>14s}  {'c_eff':>14s}")
    pr(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*14}  {'-'*14}")

    total_var = 0.0
    for sname, N, gtype, g2 in phys:
        if gtype == 'U1':
            beta = 1.0 / g2
            mean, var = u1_plaquette_stats(beta)
        else:
            beta = 2.0 * N / g2
            mean, var = sun_plaquette_stats(N, beta)
        c = c_eff(var)
        total_var += var
        pr(f"  {sname:>10s}  {g2:8.4f}  {beta:8.3f}  {mean:10.6f}  "
           f"{var:14.6e}  {c:14.6e}")

    total_c = c_eff(total_var)
    k_star = 1.0 / (total_c * 0.1) if total_c > 0 else float('inf')
    pr(f"\n  SM TOTAL:  Var = {total_var:.6e},  c_eff = {total_c:.6e},  k* = {k_star:.1f}")

    # Compare with hypothetical GUTs at THEIR natural coupling
    pr(f"\n  Comparison with GUTs at matching couplings:")
    # SU(5) GUT at alpha_GUT ~ 1/40 => g^2 = 4*pi/40 = 0.314
    g2_gut = 0.314
    beta_su5 = 2.0 * 5 / g2_gut
    mean_su5, var_su5 = sun_plaquette_stats(5, beta_su5)
    c_su5 = c_eff(var_su5)
    pr(f"  SU(5) at g^2_GUT = {g2_gut}:  beta = {beta_su5:.1f},  "
       f"Var = {var_su5:.6e},  c = {c_su5:.6e}")

    # ==================================================================
    # PART 4: Scaling c(SU(N)) vs N
    # ==================================================================
    pr("\n" + "-" * 80)
    pr("PART 4: Scaling of c_eff with N for SU(N)")
    pr("-" * 80)

    for g2 in [1.0, 2.0, 4.0]:
        pr(f"\n  g^2 = {g2}")
        pr(f"  {'N':>4s}  {'beta':>8s}  {'Var':>14s}  {'c_eff':>14s}  "
           f"{'c/N^2':>14s}  {'Var*N^2':>14s}")
        pr(f"  {'-'*4}  {'-'*8}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}")

        for N in range(2, 9):
            beta = 2.0 * N / g2
            mean, var = sun_plaquette_stats(N, beta)
            c = c_eff(var)
            pr(f"  {N:4d}  {beta:8.2f}  {var:14.6e}  {c:14.6e}  "
               f"{c/N**2:14.6e}  {var*N**2:14.6e}")

    # ==================================================================
    # PART 5: SO(10) via GPU MC
    # ==================================================================
    pr("\n" + "-" * 80)
    pr("PART 5: SO(10) GUT via GPU Monte Carlo")
    pr("-" * 80)

    for g2 in [1.0, 2.0, 4.0]:
        beta_so10 = 10.0 / g2
        pr(f"\n  g^2 = {g2}, beta = {beta_so10:.1f}")
        t0 = time.time()
        mean_so10, var_so10 = son_plaquette_stats_gpu(10, beta_so10, n_samples=200000)
        c_so10 = c_eff(var_so10)
        dt = time.time() - t0
        pr(f"  SO(10): <plaq> = {mean_so10:.6f}, Var = {var_so10:.6e}, "
           f"c = {c_so10:.6e}  [{dt:.1f}s]")

    # ==================================================================
    # PART 6: Perturbative analysis
    # ==================================================================
    pr("\n" + "-" * 80)
    pr("PART 6: Perturbative (weak-coupling) analysis")
    pr("-" * 80)

    pr("""
  At weak coupling (large beta), Var(Re Tr U_p / N) ~ C_2(fund) / (2*N*beta).
  With beta = 2N/g^2: Var ~ C_2/(4N^2) * g^2.
  So c_eff^pert = 6*4*ln(2) * C_2/(4N^2) * g^2 = 6*ln(2)*C_2/N^2 * g^2.
  The GROUP-INTRINSIC factor is: c_0 = 6*ln(2)*C_2(fund)/N^2.
""")

    pr(f"  {'Group':>28s}  {'C_2':>8s}  {'dim':>6s}  {'c_0 = 6ln2*C2/N^2':>18s}")
    pr(f"  {'-'*28}  {'-'*8}  {'-'*6}  {'-'*18}")

    pert_data = []
    for N in [2, 3, 4, 5, 6]:
        C2 = casimir_fund(N)
        dim = N**2 - 1
        c0 = 6 * np.log(2) * C2 / N**2
        name = f"SU({N})"
        pr(f"  {name:>28s}  {C2:8.4f}  {dim:6d}  {c0:18.6e}")
        pert_data.append((name, c0))

    # U(1)
    c0_u1 = 6 * np.log(2) * 1.0
    pr(f"  {'U(1)':>28s}  {1.0:8.4f}  {1:6d}  {c0_u1:18.6e}")
    pert_data.append(('U(1)', c0_u1))

    # SM: sum of sectors
    c0_sm = 6 * np.log(2) * (casimir_fund(3)/9 + casimir_fund(2)/4 + 1.0)
    pr(f"  {'SM: SU(3)xSU(2)xU(1)':>28s}  {'sum':>8s}  {12:6d}  {c0_sm:18.6e}")
    pert_data.append(('SM', c0_sm))

    # PS
    c0_ps = 6 * np.log(2) * (casimir_fund(4)/16 + 2*casimir_fund(2)/4)
    pr(f"  {'PS: SU(4)xSU(2)^2':>28s}  {'sum':>8s}  {18:6d}  {c0_ps:18.6e}")
    pert_data.append(('PS', c0_ps))

    # Tri
    c0_tri = 6 * np.log(2) * 3 * casimir_fund(3) / 9
    pr(f"  {'Tri: SU(3)^3':>28s}  {'sum':>8s}  {24:6d}  {c0_tri:18.6e}")
    pert_data.append(('Tri', c0_tri))

    # SU(5) GUT
    c0_su5 = 6 * np.log(2) * casimir_fund(5) / 25
    pr(f"  {'SU(5) GUT':>28s}  {casimir_fund(5):8.4f}  {24:6d}  {c0_su5:18.6e}")
    pert_data.append(('SU(5)', c0_su5))

    pert_data.sort(key=lambda x: x[1], reverse=True)
    pr(f"\n  PERTURBATIVE RANKING (c_0 descending):")
    for i, (name, c0) in enumerate(pert_data, 1):
        marker = "  <-- SM" if name == 'SM' else ""
        pr(f"  {i:4d}. {name:>28s}  c_0 = {c0:.6e}{marker}")

    # ==================================================================
    # PART 7: Summary
    # ==================================================================
    pr("\n" + "=" * 80)
    pr("SUMMARY AND CONCLUSIONS")
    pr("=" * 80)

    # Final ranking at g^2 = 2.0 (all stable)
    g2_ref = 2.0
    pr(f"\n  FINAL RANKING at g^2 = {g2_ref}:")

    summary = []
    for name, N, gtype in simple_groups:
        r = all_results[(name, g2_ref)]
        summary.append((name, r['var'], r['c_eff']))

    sm_var = sum(all_results[(s, g2_ref)]['var'] for s in ['SU(3)', 'SU(2)', 'U(1)'])
    summary.append(('SM: SU(3)xSU(2)xU(1)', sm_var, c_eff(sm_var)))

    ps_var = all_results[('SU(4)', g2_ref)]['var'] + 2*all_results[('SU(2)', g2_ref)]['var']
    summary.append(('PS: SU(4)xSU(2)^2', ps_var, c_eff(ps_var)))

    tri_var = 3 * all_results[('SU(3)', g2_ref)]['var']
    summary.append(('Tri: SU(3)^3', tri_var, c_eff(tri_var)))

    summary.sort(key=lambda x: x[2], reverse=True)

    pr(f"  {'Rank':>4s}  {'Group':>28s}  {'Var':>14s}  {'c_eff':>14s}")
    pr(f"  {'-'*4}  {'-'*28}  {'-'*14}  {'-'*14}")
    for i, (name, var, c) in enumerate(summary, 1):
        marker = "  <-- SM" if 'SM' in name else ""
        pr(f"  {i:4d}  {name:>28s}  {var:14.6e}  {c:14.6e}{marker}")

    # Key ratios
    sm_entry = [x for x in summary if 'SM' in x[0]][0]
    su5_entry = [x for x in summary if x[0] == 'SU(5)'][0]
    u1_entry = [x for x in summary if x[0] == 'U(1)'][0]
    pr(f"\n  c(SM) / c(SU(5)) = {sm_entry[2]/su5_entry[2]:.2f}")
    pr(f"  c(SM) / c(U(1))  = {sm_entry[2]/u1_entry[2]:.2f}")

    pr(f"""
  CONCLUSIONS:
  ============
  1. c_eff is NOT maximized by the SM group. U(1) always wins for
     simple groups (fewest constraints = largest fluctuations).

  2. For simple SU(N), c_eff DECREASES monotonically with N:
     Var ~ C_2(fund)/(2*N*beta) ~ (N^2-1)/(4*N^2*beta), and
     beta = 2N/g^2, so Var ~ (N^2-1)/(8*N^3) * g^2 ~ 1/(8N) for large N.
     Larger groups are more stable (slower RG flow).

  3. Product groups SUM variances of independent sectors.
     SM = SU(3) + SU(2) + U(1) beats any single group with dim >= 12.
     The U(1) factor dominates the SM's total variance.

  4. SM > SU(5) GUT > SO(10) GUT in c_eff:
     Unification REDUCES c because it replaces multiple small-N factors
     with a single large-N group. The fragmented structure flows faster.

  5. The SM does NOT maximize c among ALL product groups.
     U(1)^12 (12 abelian factors) would have c ~ 12 * c(U(1)) >> c(SM).
     The SM is NOT selected by maximizing c.

  6. PHYSICAL INTERPRETATION: The convexity constant c measures how fast
     the lattice coupling grows under RG blocking. It depends on the
     GROUP only through Var(plaquette). The SM's relatively large c
     (compared to GUTs) means it exits the perturbative regime faster,
     which is related to confinement -- but this is a CONSEQUENCE of
     having small gauge groups, not a selection principle.

  7. The quantity c does NOT select the SM gauge group.
     There is no maximum, minimum, or special value of c at G_SM.
     The pattern is simply: more factors and smaller N = larger c.
""")

    pr(f"Total runtime: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
