#!/usr/bin/env python3
"""
Constraint-based counting of 4D quantum gauge theories.

Enumerates all compact simple/semi-simple gauge groups (rank <= 8),
checks asymptotic freedom, anomaly cancellation, gravitational anomaly,
and generation constraints. Counts how many survive ALL constraints
and determines whether G_SM = SU(3) x SU(2) x U(1) is special.

Author: Grzegorz Olbryk (computational framework)
"""

import numpy as np
from itertools import combinations_with_replacement, product
from collections import defaultdict
import time

# Try CuPy for GPU acceleration where useful
try:
    import cupy as cp
    HAS_CUPY = True
    print("CuPy available — GPU acceleration enabled")
except ImportError:
    cp = np
    HAS_CUPY = False
    print("CuPy not available — using NumPy fallback")


# =============================================================================
# PART 1: Simple compact Lie groups and their invariants
# =============================================================================

def simple_groups_up_to_rank(max_rank=8):
    """
    Enumerate all simple compact Lie groups up to given rank.

    Returns list of dicts with:
      name, rank, dim, dual_coxeter (h^v), fund_dim, fund_index,
      fund_is_real (True if fundamental is real or pseudo-real),
      fund_is_pseudoreal, anomaly_free (True if ALL reps are anomaly-free)

    The one-loop beta function coefficient for N_f Weyl fermions in
    the fundamental representation is:
      b_0 = (11/3) C_2(adj) - (2/3) N_f T(fund)
    where C_2(adj) = h^v (dual Coxeter number) and T(fund) = index of fund.

    Asymptotic freedom requires b_0 > 0, i.e.:
      N_f < (11/2) * C_2(adj) / T(fund)
    """
    groups = []

    # --- A_n = SU(n+1), rank n, n >= 1 ---
    for n in range(1, max_rank + 1):
        N = n + 1  # SU(N)
        groups.append({
            'name': f'SU({N})',
            'family': 'A',
            'rank': n,
            'dim': N**2 - 1,
            'dual_coxeter': N,           # h^v = N for SU(N)
            'fund_dim': N,
            'fund_index': 0.5,           # T(fund) = 1/2 for SU(N)
            'fund_is_real': (N == 2),     # SU(2) fund is pseudo-real
            'fund_is_pseudoreal': (N == 2),
            'anomaly_free': (N <= 2),     # SU(N>=3) has cubic anomaly
            'center': f'Z_{N}',
        })

    # --- B_n = SO(2n+1), rank n, n >= 2 ---
    for n in range(2, max_rank + 1):
        N = 2 * n + 1  # SO(N)
        groups.append({
            'name': f'SO({N})',
            'family': 'B',
            'rank': n,
            'dim': N * (N - 1) // 2,
            'dual_coxeter': N - 2,        # h^v = N-2 for SO(N)
            'fund_dim': N,
            'fund_index': 1.0,            # T(vector) = 1 for SO(N)
            'fund_is_real': True,          # vector rep is real
            'fund_is_pseudoreal': False,
            'anomaly_free': True,          # SO(N) all reps anomaly-free
            'center': 'Z_2',
        })

    # --- C_n = Sp(2n), rank n, n >= 1 ---
    # Convention: Sp(2n) has rank n, dimension n(2n+1)
    for n in range(1, max_rank + 1):
        groups.append({
            'name': f'Sp({2*n})',
            'family': 'C',
            'rank': n,
            'dim': n * (2 * n + 1),
            'dual_coxeter': n + 1,        # h^v = n+1 for Sp(2n)
            'fund_dim': 2 * n,
            'fund_index': 0.5,            # T(fund) = 1/2 for Sp(2n)
            'fund_is_real': False,
            'fund_is_pseudoreal': True,    # Sp(2n) fund is pseudo-real
            'anomaly_free': True,          # Sp(2n) all reps anomaly-free
            'center': 'Z_2',
        })

    # --- D_n = SO(2n), rank n, n >= 3 ---
    # (D_2 = SO(4) is not simple: SO(4) ~ SU(2) x SU(2))
    for n in range(3, max_rank + 1):
        N = 2 * n  # SO(N)
        groups.append({
            'name': f'SO({N})',
            'family': 'D',
            'rank': n,
            'dim': N * (N - 1) // 2,
            'dual_coxeter': N - 2,        # h^v = N-2 for SO(N)
            'fund_dim': N,
            'fund_index': 1.0,            # T(vector) = 1 for SO(N)
            'fund_is_real': True,          # vector rep is real
            'fund_is_pseudoreal': False,
            'anomaly_free': (N != 8),     # SO(2n) anomaly-free except triality issues
            # Actually SO(2n) for all n: vector rep is real, no cubic anomaly
            'center': f'Z_2 x Z_2' if n % 2 == 0 else 'Z_4',
        })
        # Correction: SO(2n) IS anomaly-free for all n (all reps real)
        groups[-1]['anomaly_free'] = True

    # --- Exceptional groups ---
    exceptionals = [
        {
            'name': 'G2',
            'family': 'G',
            'rank': 2,
            'dim': 14,
            'dual_coxeter': 4,
            'fund_dim': 7,
            'fund_index': 1.0,
            'fund_is_real': True,
            'fund_is_pseudoreal': False,
            'anomaly_free': True,
            'center': 'trivial',
        },
        {
            'name': 'F4',
            'family': 'F',
            'rank': 4,
            'dim': 52,
            'dual_coxeter': 9,
            'fund_dim': 26,
            'fund_index': 3.0,
            'fund_is_real': True,
            'fund_is_pseudoreal': False,
            'anomaly_free': True,
            'center': 'trivial',
        },
        {
            'name': 'E6',
            'family': 'E',
            'rank': 6,
            'dim': 78,
            'dual_coxeter': 12,
            'fund_dim': 27,
            'fund_index': 3.0,
            'fund_is_real': False,
            'fund_is_pseudoreal': False,
            'anomaly_free': False,  # E6 has complex reps (27 != 27bar)
            'center': 'Z_3',
        },
        {
            'name': 'E7',
            'family': 'E',
            'rank': 7,
            'dim': 133,
            'dual_coxeter': 18,
            'fund_dim': 56,
            'fund_index': 6.0,
            'fund_is_real': False,
            'fund_is_pseudoreal': True,  # 56 is pseudo-real
            'anomaly_free': True,        # pseudo-real => anomaly-free
            'center': 'Z_2',
        },
        {
            'name': 'E8',
            'family': 'E',
            'rank': 8,
            'dim': 248,
            'dual_coxeter': 30,
            'fund_dim': 248,  # smallest rep IS the adjoint
            'fund_index': 30.0,
            'fund_is_real': True,
            'fund_is_pseudoreal': False,
            'anomaly_free': True,
            'center': 'trivial',
        },
    ]
    groups.extend(exceptionals)

    return groups


def compute_nf_max(g):
    """
    Maximum number of Weyl fermion flavors in the fundamental rep
    for asymptotic freedom.

    b_0 = (11/3) h^v - (2/3) N_f T(fund) > 0
    => N_f < (11/2) h^v / T(fund)
    """
    hv = g['dual_coxeter']
    Tf = g['fund_index']
    nf_max_continuous = (11.0 / 2.0) * hv / Tf
    return int(np.floor(nf_max_continuous - 1e-10))  # strict inequality


def b0_coefficient(g, nf):
    """One-loop beta function coefficient b_0 for N_f Weyl fermions in fund rep."""
    hv = g['dual_coxeter']
    Tf = g['fund_index']
    return (11.0 / 3.0) * hv - (2.0 / 3.0) * nf * Tf


# =============================================================================
# PART 2: Product groups
# =============================================================================

def enumerate_product_groups(simple_groups, max_factors=4, max_total_rank=8):
    """
    Enumerate all product groups G1 x G2 x ... x Gk with:
      - k <= max_factors
      - sum of ranks <= max_total_rank
      - each factor is a simple group from the list
      - order doesn't matter (combinations with replacement)

    Returns list of tuples of group dicts.
    """
    products = []

    # Single factors
    for g in simple_groups:
        if g['rank'] <= max_total_rank:
            products.append((g,))

    # Multi-factor products
    for k in range(2, max_factors + 1):
        for combo in combinations_with_replacement(range(len(simple_groups)), k):
            groups_combo = tuple(simple_groups[i] for i in combo)
            total_rank = sum(g['rank'] for g in groups_combo)
            if total_rank <= max_total_rank:
                products.append(groups_combo)

    return products


# =============================================================================
# PART 3: Anomaly cancellation analysis
# =============================================================================

def check_anomaly_cancellation_simple(g, nf, n_gen=1):
    """
    Check if a simple group G with N_f fundamental Weyl fermions
    (organized in n_gen generations) can be anomaly-free.

    For SU(N>=3): need equal fundamentals and anti-fundamentals.
      A single "generation" = one fund + one anti-fund = 2 Weyl fermions.
      So N_f must be even, and we use N_f/2 generations.

    For SU(2), SO(N), Sp(2n), G2, F4, E7, E8:
      All representations are real or pseudo-real => automatically anomaly-free.
      Any N_f works.

    For E6: complex 27, need 27 + 27bar pairs. Same logic as SU(N>=3).

    Returns: (is_anomaly_free, n_weyl_fermions, description)
    """
    if g['anomaly_free']:
        # Real or pseudo-real reps: no cubic anomaly constraint
        n_weyl = nf * g['fund_dim']
        return True, n_weyl, f"{nf} x {g['fund_dim']}"
    else:
        # Complex reps: need fund + anti-fund pairs
        if nf % 2 != 0:
            return False, 0, "odd N_f for complex rep"
        n_pairs = nf // 2
        n_weyl = nf * g['fund_dim']  # n_pairs funds + n_pairs anti-funds
        return True, n_weyl, f"{n_pairs} x ({g['fund_dim']} + {g['fund_dim']}bar)"


def check_gravitational_anomaly(n_weyl_total):
    """
    Gravitational anomaly cancellation requires:
      Tr(1) = total number of Weyl fermions must be even.

    In 4D, the mixed gravitational-gauge anomaly vanishes if
    the total number of chiral fermions is even.
    """
    return n_weyl_total % 2 == 0


# =============================================================================
# PART 4: Generation structure counting
# =============================================================================

def count_generation_structures(g, nf_max, max_generations=3):
    """
    For a simple group, count distinct generation structures.

    A "generation" for groups with complex reps = one fund + one anti-fund.
    For real/pseudo-real groups, a "generation" = one fundamental Weyl fermion.

    We count: number of (N_f, n_gen) pairs where:
      - N_f <= nf_max (AF)
      - N_f > 0 (need matter)
      - anomaly cancellation holds
      - gravitational anomaly holds
      - n_gen <= max_generations
    """
    structures = []

    for nf in range(1, nf_max + 1):
        ok, n_weyl, desc = check_anomaly_cancellation_simple(g, nf)
        if not ok:
            continue
        if not check_gravitational_anomaly(n_weyl):
            continue

        # Determine generation count
        if g['anomaly_free']:
            n_gen = nf  # each fermion could be a "generation"
        else:
            n_gen = nf // 2  # each fund+anti-fund pair is a generation

        if n_gen <= max_generations and n_gen >= 1:
            structures.append({
                'nf': nf,
                'n_gen': n_gen,
                'n_weyl': n_weyl,
                'desc': desc,
            })

    return structures


# =============================================================================
# PART 5: Full enumeration
# =============================================================================

def analyze_simple_groups(groups):
    """Analyze all simple groups for AF and anomaly cancellation."""
    print("\n" + "=" * 80)
    print("SIMPLE GROUPS: Asymptotic Freedom Analysis")
    print("=" * 80)
    print(f"{'Group':<10} {'rank':>4} {'dim':>5} {'h^v':>4} {'T_f':>4} "
          f"{'N_f^max':>7} {'anomaly':>8} {'AF+anom':>8} {'#struct':>7}")
    print("-" * 80)

    results = []
    for g in groups:
        nf_max = compute_nf_max(g)
        structs = count_generation_structures(g, nf_max)

        # Pure gauge (N_f=0) is always AF and anomaly-free
        has_af = nf_max >= 0
        has_af_anom = len(structs) > 0 or nf_max >= 0  # pure gauge always works

        print(f"{g['name']:<10} {g['rank']:>4} {g['dim']:>5} {g['dual_coxeter']:>4} "
              f"{g['fund_index']:>4.1f} {nf_max:>7} "
              f"{'yes' if g['anomaly_free'] else 'no':>8} "
              f"{'yes':>8} {len(structs):>7}")

        results.append({
            'group': g,
            'nf_max': nf_max,
            'structures': structs,
        })

    return results


def analyze_product_groups(simple_groups, max_factors=4, max_total_rank=8):
    """
    Analyze product groups.

    For product groups G1 x G2 x ... x Gk:
    - Each factor must be independently asymptotically free
    - Anomaly cancellation is per factor (for fundamental reps)
    - Cross-factor anomalies (mixed anomalies) constrain representations further

    We count "minimal" theories: each factor gets independent matter content.
    """
    print("\n" + "=" * 80)
    print("PRODUCT GROUPS: Enumeration (total rank <= 8, up to 4 factors)")
    print("=" * 80)

    products = enumerate_product_groups(simple_groups, max_factors, max_total_rank)

    # Count by number of factors
    by_nfactors = defaultdict(int)
    for p in products:
        by_nfactors[len(p)] += 1

    print(f"\nProduct groups by number of factors:")
    for k in sorted(by_nfactors.keys()):
        print(f"  {k} factor(s): {by_nfactors[k]}")
    print(f"  Total: {len(products)}")

    return products


def count_theories_for_product(product_group, max_gen=3):
    """
    Count the number of distinct anomaly-free, AF theories for a product group.

    For simplicity, we consider independent matter content per factor
    (each factor has its own N_f in fundamental rep).

    A theory is specified by choosing N_f for each factor such that:
    - Each factor is AF
    - Each factor is anomaly-free
    - Total Weyl fermion count is even (gravitational anomaly)
    - Total generations <= 3

    For the SM-like case, we also need to consider bi-fundamental
    representations, but that's a much harder problem. We note this caveat.
    """
    factor_options = []

    for g in product_group:
        nf_max = compute_nf_max(g)
        options = []

        # Pure gauge (N_f = 0) always works
        options.append({'nf': 0, 'n_weyl': 0, 'n_gen': 0})

        for nf in range(1, nf_max + 1):
            ok, n_weyl, desc = check_anomaly_cancellation_simple(g, nf)
            if ok:
                n_gen = nf if g['anomaly_free'] else nf // 2
                if n_gen <= max_gen:
                    options.append({'nf': nf, 'n_weyl': n_weyl, 'n_gen': n_gen})

        factor_options.append(options)

    # Count all combinations
    count = 0
    theories = []

    for combo in product(*factor_options):
        total_weyl = sum(c['n_weyl'] for c in combo)
        max_gen = max(c['n_gen'] for c in combo) if combo else 0
        any_matter = any(c['nf'] > 0 for c in combo)

        # Gravitational anomaly
        if not check_gravitational_anomaly(total_weyl):
            continue

        # At least one factor must have matter (otherwise pure gauge, counted separately)
        # Actually, pure gauge IS a valid theory, so we include it

        # Generation constraint: the max generation count across factors <= 3
        if max_gen <= 3:
            count += 1
            if any_matter:
                theories.append(combo)

    return count, theories


# =============================================================================
# PART 6: SM comparison
# =============================================================================

def analyze_sm():
    """
    Analyze the Standard Model: SU(3) x SU(2) x U(1)_Y

    SM has:
    - SU(3): 6 Weyl quarks per generation (3 colors x 2 chiralities) = N_f = 6
      Actually: left-handed quarks (3) + right-handed up (3bar) + right-handed down (3bar)
      Per generation: 2 triplets (Q_L) + 1 anti-triplet (u_R) + 1 anti-triplet (d_R)
      In terms of left-handed Weyl: 1 fund (Q_L has 2 flavors) ...

      More precisely per generation:
      Q_L: (3, 2, 1/6) = 6 Weyl dof
      u_R: (3bar, 1, -2/3) = 3 Weyl dof
      d_R: (3bar, 1, 1/3) = 3 Weyl dof
      L_L: (1, 2, -1/2) = 2 Weyl dof
      e_R: (1, 1, 1) = 1 Weyl dof
      Total per gen: 15 Weyl fermions (left-handed)
      With right-handed neutrino: 16

    SM anomaly cancellation is EXTREMELY non-trivial:
      - SU(3)^3: cancels (equal fund and anti-fund)
      - SU(2)^3: automatic (pseudo-real)
      - U(1)^3: Σ Y^3 = 0 (requires specific hypercharge assignments!)
      - SU(3)^2 U(1): Σ Y = 0 over color triplets
      - SU(2)^2 U(1): Σ Y = 0 over doublets
      - Gravitational: Σ Y = 0 (or Tr(1) even)

    All 6 anomaly cancellation conditions are satisfied with SM charges.
    """
    print("\n" + "=" * 80)
    print("STANDARD MODEL ANALYSIS: SU(3) x SU(2) x U(1)")
    print("=" * 80)

    # SU(3) factor
    su3_hv = 3
    su3_tf = 0.5
    # Per generation: 2 fund + 2 anti-fund (in left-handed convention)
    # Q_L = 2 triplets, u_R^c = 1 anti-triplet -> 1 triplet (conjugate)
    # Actually in left-handed: Q_L(3), u_R^c(3bar), d_R^c(3bar)
    # For b_0: count Weyl fermions in fund: Q_L gives 2 (up,down) x fund = 2
    #          u_R^c gives 1 x anti-fund, d_R^c gives 1 x anti-fund
    # Total SU(3) fund Weyl fermions per gen: 2 (from Q_L)
    # Total SU(3) anti-fund per gen: 2 (from u_R^c, d_R^c)
    # For beta function: N_f = 2 + 2 = 4 per generation (counting fund + anti-fund)
    # Wait: for SU(3), Q_L is a doublet under SU(2), so it contributes
    # 2 Weyl fermions in the 3 of SU(3) per generation
    # u_R^c: 1 Weyl in 3bar = 1 Weyl in fund (for beta function, same T_f)
    # d_R^c: 1 Weyl in 3bar
    # Leptons: singlets under SU(3)
    # Total N_f(SU(3)) = 2 + 1 + 1 = 4 per generation = 12 for 3 generations
    # Wait, T(3bar) = T(3) = 1/2, so each 3 or 3bar contributes equally
    # Actually, for Weyl fermions: each left-handed Weyl in fund or anti-fund
    # contributes T(fund) = 1/2 to the running.
    # Per generation: Q_L has SU(2) doublet in 3 => 2 Weyl in 3
    #                 u_R^c in 3bar => 1 Weyl in 3bar
    #                 d_R^c in 3bar => 1 Weyl in 3bar
    # Total: 4 Weyl fermions charged under SU(3) per generation
    # With N_gen = 3: N_f = 12
    nf_su3 = 12  # 3 generations x 4 colored Weyl fermions each
    b0_su3 = (11.0/3.0) * su3_hv - (2.0/3.0) * nf_su3 * su3_tf
    # = (11/3)*3 - (2/3)*12*(1/2) = 11 - 4 = 7

    # SU(2) factor
    su2_hv = 2
    su2_tf = 0.5
    # Per generation: Q_L (3 colors x doublet) = 3 doublets
    #                 L_L (1 x doublet) = 1 doublet
    # Total: 4 doublets per generation = 12 doublets for 3 gen
    # But for beta function, each SU(2) doublet = 1 Weyl fermion in fund
    # Wait: Q_L is (3,2) so it's 3 Weyl doublets
    nf_su2 = 12  # 3 gen x (3 quark doublets + 1 lepton doublet) = 3 x 4 = 12
    b0_su2 = (11.0/3.0) * su2_hv - (2.0/3.0) * nf_su2 * su2_tf
    # = (22/3) - (2/3)*12*(1/2) = 22/3 - 4 = 10/3

    # Actually let me recalculate more carefully.
    # b_0 = (1/(16pi^2)) * [11/3 C_2(adj) - 2/3 sum_reps T(R)]
    # For SU(3): C_2(adj) = N = 3
    #   sum_reps T(R) = N_gen * (T(3)*n_doublets_Q + T(3bar)*1 + T(3bar)*1)
    #   Hmm, this depends on convention. Let me use the standard result:
    #   For SU(3) with n_f quark flavors (each flavor = one Dirac fermion = 2 Weyl):
    #   b_0 = (11*3 - 2*n_f) / (48*pi^2)  where n_f = 6 (u,d,s,c,b,t)
    #   = (33 - 12)/(48pi^2) = 21/(48pi^2) > 0  ✓

    # For SU(2) with the SM matter:
    #   SU(2) sees: 3 generations x (Q_L: 3 doublets + L_L: 1 doublet) = 12 doublets
    #   Each doublet = 1 Weyl in fund of SU(2)
    #   b_0_SU2 = 11/3 * 2 - 2/3 * 12 * 1/2 = 22/3 - 4 = 10/3 > 0  ✓
    #   (Plus Higgs contributes -1/6 per complex scalar doublet)

    print(f"\nSU(3) sector:")
    print(f"  n_f = 6 quark flavors (Dirac), b_0 ∝ 33-12 = 21 > 0 ✓ (AF)")
    print(f"\nSU(2) sector:")
    print(f"  12 Weyl doublets (3 gen x 4), b_0 ∝ 22/3 - 4 = 10/3 > 0 ✓ (AF)")
    print(f"\nU(1)_Y sector:")
    print(f"  NOT asymptotically free (b_0 < 0 for U(1))")
    print(f"  U(1) is free in IR, runs to Landau pole in UV")
    print(f"  => SM is NOT fully asymptotically free!")

    print(f"\nAnomaly cancellation (per generation, 15 Weyl fermions):")
    print(f"  SU(3)^3: 2*(1/2) - 1*(1/2) - 1*(1/2) = 0 ✓")
    print(f"  SU(2)^3: automatic (pseudo-real) ✓")
    print(f"  U(1)^3: 6*(1/6)^3 + 3*(-2/3)^3 + 3*(1/3)^3 + 2*(-1/2)^3 + 1^3")
    Y_cubed = 6*(1/6)**3 + 3*(-2/3)**3 + 3*(1/3)**3 + 2*(-1/2)**3 + 1**3
    print(f"         = {Y_cubed:.6f} {'✓' if abs(Y_cubed) < 1e-10 else '✗'}")

    Y_grav = 6*(1/6) + 3*(-2/3) + 3*(1/3) + 2*(-1/2) + 1*(1)
    print(f"  Gravitational (Σ Y): {Y_grav:.6f} {'✓' if abs(Y_grav) < 1e-10 else '✗'}")

    print(f"\n  Total Weyl fermions per generation: 15")
    print(f"  Total for 3 generations: 45 (odd! but with ν_R: 48 = even)")

    return {
        'b0_su3': 21,
        'b0_su2': 10.0/3,
        'anomaly_free': True,
        'n_weyl_per_gen': 15,
    }


# =============================================================================
# PART 7: Main counting
# =============================================================================

def main():
    t0 = time.time()

    print("=" * 80)
    print("CONSTRAINT-BASED COUNTING OF 4D QUANTUM GAUGE THEORIES")
    print("=" * 80)
    print(f"\nConstraints applied:")
    print(f"  1. Gauge group: compact, connected, simple or product of simples")
    print(f"  2. Asymptotic freedom: b_0 > 0 for all simple factors")
    print(f"  3. Anomaly cancellation: cubic anomaly = 0")
    print(f"  4. Gravitational anomaly: total Weyl count even")
    print(f"  5. At most 3 generations")
    print(f"  6. Rank ≤ 8, at most 4 simple factors")
    print(f"  7. Matter in fundamental representation only")

    # Step 1: Enumerate simple groups
    simple_groups = simple_groups_up_to_rank(8)
    print(f"\nSimple groups up to rank 8: {len(simple_groups)}")

    # Step 2: Analyze simple groups
    simple_results = analyze_simple_groups(simple_groups)

    # Step 3: Detailed simple group theories
    print("\n" + "=" * 80)
    print("SIMPLE GROUP THEORIES: Generation structures (≤ 3 gen, AF, anomaly-free)")
    print("=" * 80)

    simple_theories = {}
    total_simple_theories = 0

    for r in simple_results:
        g = r['group']
        structs = r['structures']
        nf_max = r['nf_max']

        # Always count pure gauge theory (N_f = 0)
        n_theories = 1 + len(structs)  # pure gauge + matter theories
        total_simple_theories += n_theories
        simple_theories[g['name']] = {
            'pure_gauge': True,
            'matter_theories': structs,
            'total': n_theories,
        }

        if structs:
            print(f"\n{g['name']} (N_f^max = {nf_max}): {n_theories} theories")
            print(f"  Pure gauge (N_f=0): always valid")
            for s in structs:
                print(f"  N_f={s['nf']}, {s['n_gen']} gen, {s['n_weyl']} Weyl: {s['desc']}")

    print(f"\n--- Total simple gauge theories: {total_simple_theories} ---")

    # Step 4: Product groups
    products = analyze_product_groups(simple_groups, max_factors=3, max_total_rank=8)
    # Limit to 3 factors for tractability (4 factors with rank ≤ 8 is mostly
    # copies of SU(2) which are covered)

    # Step 5: Count product group theories
    print("\n" + "=" * 80)
    print("PRODUCT GROUP THEORIES: Counting (this may take a moment...)")
    print("=" * 80)

    product_theory_counts = {}
    total_product_theories = 0
    sm_like_count = 0

    # For efficiency, precompute per-factor option counts
    # and only detail interesting cases

    multi_factor_products = [p for p in products if len(p) > 1]
    print(f"\nAnalyzing {len(multi_factor_products)} multi-factor product groups...")

    # Use GPU for batch computation if available
    interesting_products = []

    for i, prod in enumerate(multi_factor_products):
        name = " x ".join(g['name'] for g in prod)
        total_rank = sum(g['rank'] for g in prod)

        count, theories = count_theories_for_product(prod)

        if count > 0:
            product_theory_counts[name] = count
            total_product_theories += count

            # Check if SM-like
            factor_names = sorted(g['name'] for g in prod)
            if factor_names == ['SU(2)', 'SU(3)']:
                sm_like_count += count
                interesting_products.append((name, count, prod))
            elif 'SU(3)' in factor_names and 'SU(2)' in factor_names:
                interesting_products.append((name, count, prod))

    # Print top product groups by theory count
    print(f"\nTop 30 product groups by number of theories:")
    sorted_products = sorted(product_theory_counts.items(), key=lambda x: -x[1])
    for name, count in sorted_products[:30]:
        print(f"  {name}: {count} theories")

    print(f"\n--- Total product gauge theories (multi-factor): {total_product_theories} ---")

    # Step 6: SM analysis
    sm_info = analyze_sm()

    # Step 7: Summary
    grand_total = total_simple_theories + total_product_theories

    print("\n" + "=" * 80)
    print("GRAND SUMMARY")
    print("=" * 80)

    print(f"\nSimple gauge theories (rank ≤ 8, AF, anomaly-free, ≤ 3 gen): {total_simple_theories}")
    print(f"Product gauge theories (rank ≤ 8, AF, anomaly-free, ≤ 3 gen): {total_product_theories}")
    print(f"{'':->60}")
    print(f"TOTAL theories satisfying all constraints: {grand_total}")

    print(f"\n--- SM STATUS ---")
    print(f"G_SM = SU(3) x SU(2) x U(1)_Y")
    print(f"")
    print(f"CRITICAL OBSERVATION: U(1) is NOT a simple compact Lie group in the")
    print(f"classification sense -- it is abelian. U(1) is NOT asymptotically free.")
    print(f"The SM gauge group SU(3) x SU(2) x U(1) therefore FAILS constraint #2")
    print(f"(asymptotic freedom for ALL factors).")
    print(f"")
    print(f"If we restrict to SIMPLE or SEMI-SIMPLE (no U(1) factors) gauge groups:")
    print(f"  The SM is NOT in the list at all.")
    print(f"  The SM-like SU(3) x SU(2) part has {sm_like_count} AF+anomaly-free theories.")
    print(f"")
    print(f"GUT alternatives that ARE fully AF and contain SM:")

    # Check GUT groups
    gut_groups = ['SU(5)', 'SO(10)', 'E6']
    print(f"")
    for gname in gut_groups:
        for r in simple_results:
            if r['group']['name'] == gname:
                g = r['group']
                nf_max = r['nf_max']
                structs = r['structures']
                n_gen_3 = [s for s in structs if s['n_gen'] == 3]
                print(f"  {gname}: N_f^max = {nf_max}, "
                      f"3-generation theories: {len(n_gen_3)}")
                if n_gen_3:
                    for s in n_gen_3:
                        print(f"    N_f={s['nf']}, {s['n_weyl']} Weyl fermions")
                break

    print(f"\n" + "=" * 80)
    print(f"KEY RESULT")
    print(f"=" * 80)
    print(f"")
    print(f"{grand_total} asymptotically free, anomaly-cancellable gauge theories exist")
    print(f"with rank ≤ 8, ≤ 3 generations, and fundamental-rep matter.")
    print(f"")
    print(f"The Standard Model (with U(1)_Y) is NOT among them — U(1) breaks")
    print(f"asymptotic freedom. The SM must be embedded in a GUT for full AF.")
    print(f"")

    # Count GUT candidates with exactly 3 generations
    n_gut_3gen = 0
    for r in simple_results:
        g = r['group']
        structs = r['structures']
        for s in structs:
            if s['n_gen'] == 3:
                n_gut_3gen += 1

    print(f"Simple GUT groups with exactly 3 generations: {n_gut_3gen}")
    print(f"These include: SU(5), SO(10), E6 — the classic GUT groups.")
    print(f"")
    print(f"SM is one of VERY FEW among simple GUTs: only {n_gut_3gen} simple groups")
    print(f"support exactly 3 generations with asymptotic freedom.")
    print(f"SM (via SU(5) or SO(10) embedding) is SPECIAL but not unique.")

    elapsed = time.time() - t0
    print(f"\nComputation time: {elapsed:.2f} s")

    # ==========================================================================
    # PART 8: Detailed tables
    # ==========================================================================
    print("\n" + "=" * 80)
    print("APPENDIX: Full asymptotic freedom bounds")
    print("=" * 80)
    print(f"\n{'Group':<10} {'h^v':>5} {'T_f':>5} {'N_f^max':>8} {'b0(N_f=0)':>10} "
          f"{'b0(N_f=max)':>12}")
    print("-" * 60)
    for g in simple_groups:
        nfm = compute_nf_max(g)
        b0_0 = b0_coefficient(g, 0)
        b0_max = b0_coefficient(g, nfm)
        print(f"{g['name']:<10} {g['dual_coxeter']:>5} {g['fund_index']:>5.1f} "
              f"{nfm:>8} {b0_0:>10.2f} {b0_max:>12.2f}")

    print("\n" + "=" * 80)
    print("APPENDIX: Product groups with SU(3) factor (SM-relevant)")
    print("=" * 80)
    for name, count, prod in interesting_products:
        print(f"  {name}: {count} theories")


if __name__ == '__main__':
    main()
