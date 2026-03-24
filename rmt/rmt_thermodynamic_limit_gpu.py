"""
RMT Thermodynamic Limit — GPU (CuPy) version for RTX5070 12GB
==============================================================

Computes the level-spacing ratio r for the actual TOE operator:

    M = L_ring + alpha * Phi_Arnold

where:
  L_ring = ring (1D nearest-neighbour) Laplacian (sparse, N×N)
  Phi_Arnold = diagonal phase matrix from Arnold cat map orbit

This is the ACTUAL operator used in the TOE, not a proxy.

Scans N = 500, 1000, 2000, 4000, 6000, 8000 at alpha = 0.3
(alpha where semi-Poisson behaviour was previously seen at N=400-800).

RTX5070 12GB: N=8000 requires ~1.2 GB for eigendecomp — should fit.

Usage:
    python rmt_thermodynamic_limit_gpu.py

Output: prints r(N) table + saves RESULT_083_rmt_thermodynamic_gpu.md
"""

import numpy as np
import sys
import time
import os

import cupy as cp
import cupy.linalg as cpla
GPU_AVAILABLE = True
print(f"CuPy+cuSOLVER: {cp.__version__}  |  Device: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
mempool = cp.get_default_memory_pool()

# ---------------------------------------------------------------------------

def build_toe_operator_gpu(N, alpha=0.3):
    """
    Build TOE operator entirely on GPU using CuPy vectorised ops.
    No Python loops over matrix elements.

    Returns Hermitian cupy array (N, N) complex128.
    """
    # --- Ring Laplacian (on GPU) ---
    idx = cp.arange(N, dtype=cp.int32)
    row = cp.concatenate([idx, idx, idx])
    col = cp.concatenate([idx, (idx + 1) % N, (idx - 1) % N])
    val = cp.concatenate([cp.full(N, 2.0), cp.full(N, -1.0), cp.full(N, -1.0)])
    # Build dense from COO (N usually fits in 12 GB: N=10000 → 1.6 GB)
    L = cp.zeros((N, N), dtype=cp.complex128)
    L[row, col] = val.astype(cp.complex128)

    # --- Arnold cat map phases (generate on CPU, copy once) ---
    # CPU loop over N ints is negligible vs GPU eigendecomp
    x_arr = np.empty(N, dtype=np.int64)
    x, y = 1, 0
    for k in range(N):
        x_arr[k] = x
        x, y = (x + y) % N, (x + 2 * y) % N
    phases_cpu = (2.0 * np.pi / N) * x_arr
    phases_gpu = cp.asarray(phases_cpu)

    diag_phase = cp.exp(1j * phases_gpu)          # (N,) complex128
    # Phi = diag(diag_phase); Phi_herm = alpha*(Phi+Phi†)/2 = alpha*Re(Phi) on diag
    # (since Phi is diagonal, Phi†=conj(Phi))
    Phi_herm_diag = alpha * cp.real(diag_phase)   # real part of diagonal
    L[idx, idx] += Phi_herm_diag.astype(cp.complex128)

    return L


def compute_r_statistic_gpu(eigs_gpu):
    """
    Level-spacing ratio r on GPU.
    eigs_gpu: sorted real cupy array of eigenvalues.
    """
    spacings = cp.diff(eigs_gpu)
    threshold = 1e-12 * cp.abs(eigs_gpu).max()
    mask = spacings > threshold
    spacings = spacings[mask]
    if spacings.size < 10:
        return np.nan
    s1 = spacings[:-1]
    s2 = spacings[1:]
    ratios = cp.minimum(s1, s2) / cp.maximum(s1, s2)
    return float(cp.mean(ratios))


def run_gpu(N, alpha):
    """Build and diagonalise entirely on GPU."""
    M = build_toe_operator_gpu(N, alpha)
    eigs = cpla.eigvalsh(M)     # returns sorted real eigenvalues (cupy)
    r = compute_r_statistic_gpu(eigs)
    del M, eigs
    mempool.free_all_blocks()
    return r


# ---------------------------------------------------------------------------
# Reference values
GOE_R   = 0.5307   # Wigner-Dyson GOE
POISSON_R = 0.3863  # Poisson
SEMI_P_R  = 0.4167  # Semi-Poisson (approximate)

# Scan parameters
ALPHA = 0.3   # previously showed semi-Poisson behaviour at N=400-800
# RTX5070 12GB: N=10000 dense complex128 = 1.6GB matrix + ~4GB workspace → fits
N_VALUES = [500, 1000, 2000, 4000, 6000, 8000, 10000]

# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TOE RMT THERMODYNAMIC LIMIT — GPU SCAN")
    print(f"alpha = {ALPHA}")
    print(f"N values: {N_VALUES}")
    print(f"GPU: {'YES (CuPy)' if GPU_AVAILABLE else 'NO (numpy fallback)'}")
    print("=" * 70)
    print(f"\nReference: GOE r={GOE_R:.4f}, Poisson r={POISSON_R:.4f}, Semi-Poisson r~{SEMI_P_R:.4f}\n")

    results = []

    for N in N_VALUES:
        mem_est_gb = (N * N * 16) / 1e9  # complex128 bytes
        print(f"N={N:6d}  (matrix {mem_est_gb:.2f} GB)", end="", flush=True)

        t0 = time.time()
        try:
            r = run_gpu(N, ALPHA)
            dt = time.time() - t0

            dist_goe    = abs(r - GOE_R)
            dist_poisson = abs(r - POISSON_R)
            dist_semip  = abs(r - SEMI_P_R)
            regime = "GOE" if dist_goe < dist_poisson and dist_goe < dist_semip else \
                     ("Poisson" if dist_poisson < dist_semip else "semi-Poisson")

            print(f"  r={r:.4f}  [{regime}]  {dt:.1f}s")
            results.append((N, r, dt))

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((N, np.nan, np.nan))

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'N':>8}  {'r':>8}  {'vs GOE':>10}  {'vs Poisson':>12}  {'vs SemiP':>10}")
    print("-" * 70)
    for N, r, dt in results:
        if np.isnan(r):
            print(f"{N:>8}  {'---':>8}")
        else:
            print(f"{N:>8}  {r:.4f}  {r-GOE_R:+.4f}      {r-POISSON_R:+.4f}        {r-SEMI_P_R:+.4f}")

    print(f"\nGOE:         r = {GOE_R:.4f}")
    print(f"Poisson:     r = {POISSON_R:.4f}")
    print(f"Semi-Poisson r ~ {SEMI_P_R:.4f}")

    # Extrapolate trend
    valid = [(N, r) for N, r, _ in results if not np.isnan(r) and N >= 1000]
    if len(valid) >= 3:
        Ns = np.array([v[0] for v in valid], dtype=float)
        rs = np.array([v[1] for v in valid])
        # Fit r = a + b/N
        A = np.column_stack([np.ones_like(Ns), 1.0/Ns])
        coeff, _, _, _ = np.linalg.lstsq(A, rs, rcond=None)
        r_inf = coeff[0]
        print(f"\nExtrapolation r(N→∞) = {r_inf:.4f}  (fit r = {coeff[0]:.4f} + {coeff[1]:.1f}/N)")
        if abs(r_inf - GOE_R) < 0.02:
            conclusion = "→ GOE in thermodynamic limit"
        elif abs(r_inf - POISSON_R) < 0.02:
            conclusion = "→ Poisson in thermodynamic limit"
        elif abs(r_inf - SEMI_P_R) < 0.02:
            conclusion = "→ Semi-Poisson in thermodynamic limit (pseudo-integrable CONFIRMED)"
        else:
            conclusion = f"→ intermediate regime r_inf={r_inf:.4f}"
        print(f"Conclusion: {conclusion}")
    else:
        r_inf = None
        conclusion = "insufficient data for extrapolation"

    # Write result file
    result_lines = [
        "# RESULT_083 — RMT Thermodynamic Limit (GPU, actual TOE operator)",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}",
        f"**Script:** `global-spectral-toe/rmt/rmt_thermodynamic_limit_gpu.py`",
        f"**Hardware:** RTX5070 12GB (CuPy)" if GPU_AVAILABLE else "**Hardware:** CPU fallback",
        f"**alpha:** {ALPHA}",
        "",
        "## Operator",
        "",
        "```",
        "M = L_ring + alpha * (Phi_Arnold + Phi_Arnold†) / 2",
        "L_ring: 1D ring Laplacian (nearest-neighbour, periodic)",
        "Phi_Arnold: diagonal phase from Arnold cat map orbit",
        "```",
        "",
        "## Results",
        "",
        f"| N | r | vs GOE ({GOE_R:.4f}) | vs Poisson ({POISSON_R:.4f}) |",
        "|---|---|---|---|",
    ]
    for N, r, _ in results:
        if np.isnan(r):
            result_lines.append(f"| {N} | — | — | — |")
        else:
            result_lines.append(f"| {N} | {r:.4f} | {r-GOE_R:+.4f} | {r-POISSON_R:+.4f} |")

    result_lines += [
        "",
        "## Extrapolation",
        "",
        f"Fit r = a + b/N  →  r(∞) = {r_inf:.4f}" if r_inf is not None else "Extrapolation: insufficient data",
        f"**Conclusion:** {conclusion}",
        "",
        "## Physical interpretation",
        "",
        "- If r(∞) → Poisson: TOE operator is integrable in thermodynamic limit",
        "  (consistent with Stokes constraint, but r=0.43 is finite-N crossover only)",
        "- If r(∞) → GOE: chaotic thermodynamic limit — contradicts Stokes pseudo-integrability",
        "- If r(∞) → semi-Poisson (~0.42): pseudo-integrable CONFIRMED at thermodynamic level",
        "",
        "## Status for paper",
        "",
        "This directly determines whether Thm 5.2 (r ∈ (0.386, 0.530)) is a finite-N",
        "statement or a genuine thermodynamic prediction.",
    ]

    outpath = os.path.expanduser("~/research/results/RESULT_083_rmt_thermodynamic_gpu.md")
    with open(outpath, "w") as f:
        f.write("\n".join(result_lines) + "\n")
    print(f"\nResult saved to: {outpath}")


if __name__ == "__main__":
    main()
