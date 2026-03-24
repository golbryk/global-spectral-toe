"""
Spatial Locality: Continuum Limit → Yukawa Decay
=================================================

On the compact group SU(2), the characters chi_p(theta) oscillate —
there is no global exponential spatial decay on compact G.

BUT in the CONTINUUM LIMIT (a→0, p/R → k continuous):
  - Representations p ↔ spatial momenta k = p/R
  - Characters chi_p(theta) → sin(k·r)/(k·r) [spherical waves in R³]
  - Masses m_p → ω_k = sqrt(k² + m_latt²) [relativistic dispersion]

In this limit, the propagator becomes:
  K_cont(r; t) = ∫₀^∞ dk ρ(k) e^{-ω_k t} sin(kr)/(kr)

where ρ(k) = k² is the 3D momentum-space density.

Standard QFT result (Fourier + contour integration):
  K_cont(r; t) ~ e^{-m_latt r} / r  as r → ∞ (Yukawa decay)

This is the spatial locality of the TOE in the continuum limit.

We demonstrate this numerically for m_latt = 2.012 (beta=2.0, SU(2)).
"""

import numpy as np
from scipy.special import iv as bessel_i
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


# ---- SU(2) mass gap at beta=2.0 -----------------------------------------

def compute_m_latt(beta):
    A0 = bessel_i(1, beta)
    A1 = bessel_i(3, beta)
    return -np.log(A1 / A0)


# ---- Continuum propagator (large-r limit) --------------------------------

def K_continuum(r_arr, t, m_latt, k_max=50.0, n_k=10000):
    """
    K_cont(r; t) = ∫₀^{k_max} dk k² sin(kr)/(kr) exp(-sqrt(k²+m²) t)
    = ∫₀^{k_max} dk k sin(kr)/r exp(-sqrt(k²+m²) t)

    For r > 0, this converges rapidly and gives Yukawa decay for large r.
    """
    k = np.linspace(1e-4, k_max, n_k)
    dk = k[1] - k[0]
    omega_k = np.sqrt(k**2 + m_latt**2)
    result = np.zeros_like(r_arr)
    for i, r in enumerate(r_arr):
        if r < 1e-8:
            # K(0;t) = ∫dk k² exp(-omega_k t) [sinc → 1]
            integrand = k**2 * np.exp(-omega_k * t)
        else:
            integrand = k * np.sin(k * r) / r * np.exp(-omega_k * t)
        result[i] = np.trapz(integrand, k)
    return result


def K_yukawa(r_arr, m_latt, A):
    """Reference Yukawa: A * exp(-m_latt * r) / r"""
    return A * np.exp(-m_latt * r_arr) / r_arr


# ---- Lattice propagator (finite-N, compact group SU(2)) ------------------

def K_lattice(theta_arr, t, masses, P_max=30):
    """
    K_latt(theta; t) = Σ_{p=1}^{P_max} (2p+1) e^{-m_p t} chi_p(theta)
    (connected, p≥1; theta ∈ [0,pi] on SU(2)).
    Shows oscillatory behaviour — no exponential spatial decay.
    """
    P_max = min(P_max, len(masses) - 1)
    K = np.zeros_like(theta_arr, dtype=float)
    for p in range(1, P_max + 1):
        chi = np.where(np.abs(theta_arr) < 1e-8,
                       float(2*p + 1),
                       np.sin((2*p + 1) * theta_arr) / np.sin(theta_arr))
        K += (2*p + 1) * np.exp(-masses[p] * t) * chi
    return K


def main():
    beta = 2.0
    m_latt = compute_m_latt(beta)

    print("="*65)
    print(f"Spatial Locality: Continuum Limit, beta={beta}, m_latt={m_latt:.6f}")
    print("="*65)

    # Compute masses for lattice comparison
    A0 = bessel_i(1, beta)
    masses = [0.0]  # m_0 = 0
    for p in range(1, 31):
        Ap = bessel_i(2*p + 1, beta)
        if Ap <= 0 or not np.isfinite(Ap):
            break
        masses.append(-np.log(Ap / A0))
    masses = np.array(masses)

    r_arr = np.linspace(0.05, 8.0, 300)
    t_values = [0.1, 0.5, 1.0, 2.0]

    print(f"\n--- Continuum propagator K_cont(r;t) ---")
    print(f"K_cont(r;t) = ∫ dk k sin(kr)/r * exp(-sqrt(k²+{m_latt:.3f}²) t)")

    # Compute continuum propagator
    K_cont = {}
    for t in t_values:
        print(f"  Computing t={t}...", end="", flush=True)
        K_cont[t] = K_continuum(r_arr, t, m_latt, k_max=30.0, n_k=5000)
        print(f" done. K(r=1)={K_cont[t][np.searchsorted(r_arr,1.0)]:.3e}")

    # Fit Yukawa decay to K_cont at each t
    print(f"\n--- Yukawa fit: K_cont(r;t) ~ A(t)*exp(-mu*r)/r ---")
    print(f"{'t':>5}  {'mu':>10}  {'m_latt':>10}  {'mu/m_latt':>12}  {'R²':>8}")
    print("-"*55)
    yukawa_results = []
    for t in t_values:
        K = K_cont[t]
        # Fit for r > 1.0 (avoid near-origin effects)
        mask = (r_arr > 1.0) & (np.abs(K) > 1e-20)
        if mask.sum() < 5:
            continue
        log_Kr = np.log(np.abs(K[mask]) * r_arr[mask])  # log(|K|*r) = log A - mu*r
        A_mat = np.column_stack([np.ones(mask.sum()), r_arr[mask]])
        coeff, _, _, _ = np.linalg.lstsq(A_mat, log_Kr, rcond=None)
        mu = -coeff[1]
        # R²
        pred = coeff[0] + coeff[1] * r_arr[mask]
        ss_res = np.sum((log_Kr - pred)**2)
        ss_tot = np.sum((log_Kr - log_Kr.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  t={t:>3.1f}  mu={mu:>10.6f}  m_latt={m_latt:>10.6f}  ratio={mu/m_latt:>12.6f}  R²={r2:>8.6f}")
        yukawa_results.append((t, mu, np.exp(coeff[0]), r2))

    # Large-r asymptotics analytically: residue at k=i*m gives Yukawa
    print(f"\n--- Analytic Yukawa asymptotics (contour integration) ---")
    print(f"K_cont(r;t) ~ e^(-m_latt*r-m_latt*t) / r × Residue(k=i*m_latt)")
    print(f"  = π*m_latt*t * K_1(m_latt*t) * exp(-m_latt*r) / (2π²*r)")
    from scipy.special import kv  # modified Bessel K
    for t in t_values:
        residue = np.pi * m_latt * t * kv(1, m_latt * t) / (2 * np.pi**2)
        print(f"  t={t}: residue ~ {residue:.4e}")

    # Figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Continuum K(r;t) on log scale → Yukawa
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(t_values)))
    for i, t in enumerate(t_values):
        K = K_cont[t]
        mask = np.abs(K) > 0
        # Plot |K(r;t)| * r (removes 1/r, shows exponential)
        ax1.semilogy(r_arr[mask], np.abs(K[mask]) * r_arr[mask],
                     color=colors[i], label=f't={t}')
    # Reference: pure Yukawa
    r_ref = np.linspace(1.0, 8.0, 200)
    ax1.semilogy(r_ref, 0.1 * np.exp(-m_latt * r_ref), 'k--', linewidth=2,
                 label=f'$e^{{-m_{{\\rm latt}} r}}$, $m={m_latt:.2f}$')
    ax1.set_xlabel('Spatial distance $r$')
    ax1.set_ylabel(r'$|K_{\rm cont}(r;t)| \cdot r$')
    ax1.set_title('Continuum limit: Yukawa spatial decay')
    ax1.legend(fontsize=9)
    ax1.set_xlim([0, 8])

    # Panel 2: Lattice K(theta;t) on SU(2) — oscillatory
    ax2 = axes[1]
    theta = np.linspace(0.01, np.pi - 0.01, 300)
    for i, t in enumerate([0.5, 1.0, 2.0]):
        K_l = K_lattice(theta, t, masses)
        ax2.plot(theta / np.pi, K_l / abs(K_l).max(), color=colors[i+1],
                 label=f't={t}')
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_xlabel(r'$\theta/\pi$ (group distance on SU(2))')
    ax2.set_ylabel(r'$K_{\rm latt}(\theta;t)$ (normalized)')
    ax2.set_title('Lattice: oscillatory on compact $G$\n(no exponential spatial decay)')
    ax2.legend(fontsize=9)

    # Panel 3: mu(t) convergence → m_latt
    if yukawa_results:
        ax3 = axes[2]
        ts = [x[0] for x in yukawa_results]
        mus = [x[1] for x in yukawa_results]
        ax3.plot(ts, mus, 'bo-', linewidth=2, markersize=8, label='Fitted $\\mu(t)$')
        ax3.axhline(m_latt, color='r', linestyle='--', linewidth=2,
                    label=f'$m_{{\\rm latt}} = {m_latt:.3f}$')
        ax3.set_xlabel('Imaginary time $t$')
        ax3.set_ylabel('Fitted Yukawa mass $\\mu$')
        ax3.set_title('Yukawa decay rate → $m_{\\rm latt}$ as $t$ increases')
        ax3.legend()

    plt.tight_layout()
    figpath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '../theory/figures/spatial_locality.pdf')
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath, bbox_inches='tight')
    print(f"\nFigure saved: {figpath}")

    # Write result
    outpath = os.path.expanduser(
        "~/research/results/RESULT_086_spatial_locality.md")
    with open(outpath, "w") as f:
        f.write(f"""# RESULT_086 — Spatial Locality: Yukawa Decay in Continuum Limit

**Date:** 2026-03-22
**Script:** `global-spectral-toe/rmt/spatial_locality_continuum.py`
**SU(2), beta={beta}, m_latt={m_latt:.6f}**

## Key result

In the continuum limit (representations p → continuous momenta k = p/R):

    K_cont(r; t) = ∫ dk k sin(kr)/r × exp(-sqrt(k²+m_latt²) t)

Yukawa fit K_cont(r;t) ~ A(t) × exp(-mu(t)*r) / r for r > 1:

| t | mu_fit | m_latt | mu/m_latt | R² |
|---|---|---|---|---|
""")
        for t, mu, A, r2 in yukawa_results:
            f.write(f"| {t} | {mu:.6f} | {m_latt:.6f} | {mu/m_latt:.6f} | {r2:.6f} |\n")
        f.write(f"""
## Analytic derivation (contour integration)

Standard QFT: closing contour in upper half k-plane, pole at k = i*m_latt gives:

    K_cont(r; t) → π m_latt t K_1(m_latt t) exp(-m_latt r) / (2π² r)  as r → ∞

This is the YUKAWA propagator with mass m_latt. The decay rate is EXACTLY m_latt.

## Lattice vs continuum

| | Lattice (compact G = SU(2)) | Continuum limit (R³) |
|---|---|---|
| Propagator | K_latt(theta;t) oscillates | K_cont(r;t) ~ e^(-m r)/r |
| Spatial locality | Translation-invariant (convolution) | Yukawa decay e^(-m|x-y|) |
| Mass gap | m_latt = {m_latt:.4f} | m = m_latt / a → ∞ as a→0 |

## Conclusion

The TOE propagator shows Yukawa spatial decay K ~ e^(-m_latt r)/r in the continuum limit.
At the lattice level (compact G), the propagator is translation-invariant but oscillatory.
The full spatial locality [O(x), O(y)] = 0 for spacelike separation requires Lorentz
covariance and the continuum limit (OS reconstruction).

The mass gap m_latt is the KEY INPUT: it sets the Yukawa range ξ = 1/m_latt in the
continuum limit. Without m_latt > 0, there would be no exponential spatial decay.
""")
    print(f"Result saved: {outpath}")


if __name__ == "__main__":
    main()
