#!/usr/bin/env python3
"""Quick test: does minimizing L = Σ w_jk (log λ_j - log λ_k)² give Casimir?"""
import cupy as cp
import numpy as np
import time

t0 = time.time()
print("=" * 60)
print("SPECTRAL FUNCTIONAL MINIMIZATION — QUICK TEST")
print("=" * 60)

N = 12  # representations j=1,...,N

# Target: Casimir j(j+1)
casimir = np.array([j*(j+1) for j in range(1, N+1)], dtype=np.float64)

def spectral_L_cpu(log_lam, w):
    diff = log_lam[:, None] - log_lam[None, :]
    return np.sum(w * diff**2)

def grad_L_cpu(log_lam, w):
    diff = log_lam[:, None] - log_lam[None, :]
    return 2.0 * np.sum(w * diff, axis=1)

dims = np.array([2*j+1 for j in range(1, N+1)], dtype=np.float64)

weight_configs = {
    "w=d_j*d_k": np.outer(dims, dims),
    "w=1": np.ones((N, N)),
    "w=d_j*d_k/dist": np.outer(dims, dims) / (np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) + 1),
}

for label, w in weight_configs.items():
    np.fill_diagonal(w, 0)

    best_L = float('inf')
    best_log_lam = None

    for trial in range(200):
        # Init: random positive sorted spectrum
        log_lam = np.sort(np.abs(np.random.randn(N))) * 0.3 + np.arange(N) * 0.1
        log_lam = log_lam - log_lam[0]  # fix first

        lr = 0.00001  # very small lr
        for step in range(8000):
            g = grad_L_cpu(log_lam, w)
            g[0] = 0  # fix first
            # Clip gradient
            gnorm = np.linalg.norm(g)
            if gnorm > 10:
                g = g * 10 / gnorm
            log_lam -= lr * g

        L_val = spectral_L_cpu(log_lam, w)
        if np.isfinite(L_val) and L_val < best_L:
            best_L = L_val
            best_log_lam = log_lam.copy()

    if best_log_lam is None:
        print(f"\n--- {label} --- FAILED (no finite solution)")
        continue

    lam_opt = np.exp(best_log_lam)
    lam_opt = lam_opt / lam_opt[0]
    cas_norm = casimir / casimir[0]

    print(f"\n--- {label} ---")
    print(f"  L_min = {best_L:.6f}")
    print(f"  {'j':>4}  {'λ_opt':>10}  {'Casimir':>10}  {'ratio':>10}")
    for j in range(N):
        r = lam_opt[j] / cas_norm[j] if cas_norm[j] > 0 else float('nan')
        print(f"  {j+1:>4}  {lam_opt[j]:>10.4f}  {cas_norm[j]:>10.4f}  {r:>10.4f}")

    ratios = lam_opt / cas_norm
    cv = np.std(ratios) / np.mean(ratios)
    is_linear = np.corrcoef(np.arange(N), lam_opt)[0, 1]
    is_casimir = np.corrcoef(casimir, lam_opt)[0, 1]
    print(f"  Ratio CV = {cv:.4f} ({'~CASIMIR' if cv < 0.1 else 'NOT Casimir'})")
    print(f"  Corr with linear: {is_linear:.6f}")
    print(f"  Corr with Casimir j(j+1): {is_casimir:.6f}")

    # Also check: is minimizer UNIFORM (all log_lam equal)?
    log_spread = np.std(best_log_lam)
    print(f"  log-spread: {log_spread:.6f} ({'UNIFORM' if log_spread < 0.01 else 'non-uniform'})")

print(f"\nTime: {time.time()-t0:.1f}s")
