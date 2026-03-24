"""
Dark Matter from Spectral 4th Generation Sterile Neutrino (RESULT_061)

From RESULT_044 (n_generations = 3 section):
The spectral action predicts a "sterile neutrino" from the 4th spectral
generation (from the Clifford algebra structure: KO-dim 6 gives 4 generations
before J identification, with the 4th being sterile).

Mass of 4th generation sterile neutrino: m_4 = 5.4 eV (from Koide-like formula)

Dark matter candidates by mass:
- Hot DM: m < 100 eV (sterile neutrino 5.4 eV → HOT DM candidate)
- Warm DM: m ~ keV (e.g., 3.5 keV X-ray line candidate)
- Cold DM: m >> keV (WIMPs, axions, etc.)

A 5.4 eV sterile neutrino contributes to:
1. N_eff (Planck constraint: N_eff = 2.99 ± 0.17)
2. Hot dark matter component (structure formation bounds)
3. Neutrino mass sum (Planck: Sum m_nu < 120 meV)

KEY QUESTION: Is m_4 = 5.4 eV allowed by cosmological observations?
"""

import numpy as np

print("=" * 65)
print("DARK MATTER FROM SPECTRAL 4TH GENERATION STERILE NEUTRINO")
print("=" * 65)

# ============================================================
# PART 1: 4th spectral generation properties
# ============================================================

print("\n--- PART 1: 4th spectral generation (from RESULT_044) ---")

# From RESULT_044: The Koide formula applied to 4 generations gives:
# m_4 (4th lepton) = m_tau × (something from Koide)
# Specifically: m_4 = 5.4 eV (sterile neutrino mass)

m_4_eV = 5.4   # eV (predicted from spectral Koide extension)
m_4_GeV = m_4_eV * 1e-9  # convert to GeV

print(f"\n  m_4 (4th spectral generation) = {m_4_eV} eV")
print(f"  Classification: HOT dark matter (too light for warm DM)")

# Is it the 4th NEUTRINO or 4th CHARGED LEPTON?
# In the spectral action: the 4th generation sterile neutrino is the more natural DM candidate
# (the 4th charged lepton would be a sequential heavy lepton with m_4 >> m_tau → not 5.4 eV)
print(f"  Type: sterile neutrino (4th species, ν_4)")

# ============================================================
# PART 2: N_eff contribution
# ============================================================

print("\n--- PART 2: Contribution to N_eff ---")

# If ν_4 is in thermal equilibrium during BBN (T > m_4 = 5.4 eV):
# It contributes 1 full unit to N_eff (like a standard neutrino)
# N_eff (SM) = 3.044 (QED corrections)
# With ν_4: N_eff = 4.044 (if fully thermalized)

N_eff_SM = 3.044
N_eff_with_nu4 = 4.044  # if ν_4 fully thermalized

# Planck 2018 constraint:
N_eff_Planck = 2.99
sigma_N_eff = 0.17  # 1-sigma

print(f"\n  N_eff (SM) = {N_eff_SM:.3f}")
print(f"  N_eff (with thermalized ν_4) = {N_eff_with_nu4:.3f}")
print(f"  Planck 2018: N_eff = {N_eff_Planck:.2f} ± {sigma_N_eff:.2f}")
print(f"\n  Tension if ν_4 thermalized:")
tension = (N_eff_with_nu4 - N_eff_Planck) / sigma_N_eff
print(f"  (N_eff_pred - N_eff_obs) / sigma = {tension:.1f}σ")
print(f"  EXCLUDED at {tension:.0f}σ (if ν_4 fully thermalized)")

# Partial thermalization:
# If ν_4 decouples before neutrino decoupling (T_dec > few MeV):
# Its contribution to N_eff is diluted by (g_*s(T_dec)/g_*s(nu_dec))^{4/3}
# For T_dec >> MeV: g_*s(T_dec) ~ 106.75, g_*s(nu_dec) = 43/4 = 10.75
# Dilution factor = (10.75/106.75)^{4/3} ≈ 0.066
dilution = (10.75/106.75)**(4.0/3.0)
dN_eff_partial = dilution * 1.0
print(f"\n  If ν_4 decouples early (T_dec >> MeV):")
print(f"  Dilution factor = (10.75/106.75)^{{4/3}} = {dilution:.4f}")
print(f"  ΔN_eff = {dN_eff_partial:.4f}")
print(f"  N_eff (total) = {N_eff_SM + dN_eff_partial:.3f}")
print(f"  Tension: {abs(N_eff_SM + dN_eff_partial - N_eff_Planck)/sigma_N_eff:.1f}σ")
print(f"  Status: {'ALLOWED' if abs(N_eff_SM + dN_eff_partial - N_eff_Planck) < 2*sigma_N_eff else 'MARGINALLY ALLOWED/EXCLUDED'}")

# ============================================================
# PART 3: Hot dark matter bounds
# ============================================================

print("\n--- PART 3: Hot dark matter bounds ---")

# If ν_4 has mass 5.4 eV and contributes as hot DM:
# Omega_nu4 * h^2 = m_4 / (93.14 eV)  (for a single species in equilibrium)
h = 0.674  # Hubble parameter
Omega_nu4_h2 = m_4_eV / 93.14
Omega_nu4 = Omega_nu4_h2 / h**2

print(f"\n  If ν_4 contributes as hot DM:")
print(f"  Ω_ν4 × h² = m_4 / 93.14 = {m_4_eV}/{93.14} = {Omega_nu4_h2:.4f}")
print(f"  Ω_ν4 = {Omega_nu4:.4f}")
print(f"  Ω_CDM (observed) = 0.265")
print(f"  Ratio Ω_ν4 / Ω_CDM = {Omega_nu4/0.265:.3f}")

# Neutrino mass sum including m_4:
# Sum_active m_nu ≈ 58 meV (RESULT_048)
# + m_4 = 5400 meV
Sum_m_nu_with_nu4 = 58.0 + m_4_eV * 1000  # meV
print(f"\n  Sum m_nu (3 active) ≈ 58 meV (RESULT_048)")
print(f"  Sum m_nu + m_4 = {Sum_m_nu_with_nu4:.0f} meV = {Sum_m_nu_with_nu4/1000:.2f} eV")
print(f"  Planck 2018 bound: Sum m_nu < 120 meV")
print(f"  STATUS: EXCLUDED by Planck (Sum > 120 meV by factor {Sum_m_nu_with_nu4/120:.0f})")

# ============================================================
# PART 4: Sterile neutrino as hot DM — constraint
# ============================================================

print("\n--- PART 4: Cosmological bounds on ν_4 ---")

print(f"""
COSMOLOGICAL CONSTRAINTS ON m_4 = 5.4 eV STERILE NEUTRINO:

1. N_eff: If ν_4 thermalized → N_eff = 4.04 → EXCLUDED at 6σ
   If early decoupling → ΔN_eff ≈ 0.066 → ALLOWED (just marginally)

2. Structure formation:
   Hot DM free-streaming length: λ_FS ~ T_dec/m_4 ~ several Mpc
   For m_4 = 5.4 eV: λ_FS ~ 30 Mpc (suppresses large-scale structure!)

3. Neutrino mass sum:
   If ν_4 thermalizes: Sum m_nu → 5.46 eV >> 0.12 eV (Planck)
   EXCLUDED (factor 45 above Planck bound!)

VERDICT:
   A fully thermalized ν_4 with m_4 = 5.4 eV is STRONGLY EXCLUDED.

   The only way to avoid exclusion:
   a) ν_4 never thermalizes (mixing angle θ_4 too small)
   b) ν_4 decays before BBN (lifetime τ_4 < 1 s)
   c) ν_4 mass different from 5.4 eV (RESULT_044 prediction needs revision)
""")

# ============================================================
# PART 5: Lifetime and decay
# ============================================================

print("--- PART 5: ν_4 lifetime (if Majorana) ---")

# If ν_4 is a Majorana fermion, it can decay via:
# ν_4 → ν_i + γ  (radiative decay, if mixing angle exists)
# Lifetime: τ ~ 10^{24} (eV/m_4)^5 × (10^{-6}/θ^2) sec
# For m_4 = 5.4 eV and θ^2 ~ 10^{-6}: τ ~ (1/5.4)^5 × 10^{30} sec >> age of universe

# UNLESS: ν_4 is part of the seesaw and decays via ν_4 → ν_i + nu_bar + nu_bar (3-body)
# 3-body decay: τ ~ G_F^{-2} m_4^{-5} ~ (5.4 eV)^{-5}

# For 3-body Majorana decay:
G_F = 1.166e-5    # GeV^{-2}
m_4_GeV_val = 5.4e-9  # GeV

# Rate: Gamma ~ G_F^2 m_4^5 / (192 pi^3)
Gamma = G_F**2 * m_4_GeV_val**5 / (192 * np.pi**3)
tau_GeV = 1/Gamma  # in GeV^{-1}
hbar_c = 6.582e-25  # GeV · s (in units GeV^{-1} → seconds)
tau_sec = tau_GeV * hbar_c

print(f"\n  ν_4 3-body Majorana decay rate:")
print(f"  Γ ~ G_F² m_4^5 / 192π³")
print(f"  Γ = {Gamma:.3e} GeV")
print(f"  τ = 1/Γ = {tau_GeV:.3e} GeV^{{-1}} = {tau_sec:.3e} sec")
print(f"  Age of universe = 4.3×10^{17} sec")
print(f"  τ / age = {tau_sec/4.3e17:.3e}")
print(f"  Status: τ >> age → ν_4 is STABLE on cosmological timescales")

# ============================================================
# PART 6: Warm dark matter rescue?
# ============================================================

print("\n--- PART 6: Can ν_4 be warm dark matter if m_4 is different? ---")

print("""
For warm dark matter (WDM), the required mass is:
  m_WDM > 3.5 keV (Lyman-α forest constraint)
  Typical WDM candidate: m ~ 7 keV (3.5 keV X-ray line)

The spectral action prediction m_4 = 5.4 eV is 1000× too light for WDM.

HOWEVER: If the 4th generation sterile neutrino mass gets a seesaw-type
correction at M_R:
  m_4_physical ≈ m_4_Dirac² / m_4_Majorana

With m_4_Dirac ~ m_4_eV = 5.4 eV and m_4_Majorana set by some scale...

For m_4_physical = 7 keV (WDM):
  m_4_Majorana = m_4_Dirac² / m_4_physical = (5.4 eV)² / 7000 eV
              = 29.16 / 7000 eV = 4.2 meV

But a 4.2 meV Majorana mass is far below any natural scale.

CONCLUSION: The spectral 4th generation with m_4 = 5.4 eV is:
  - NOT viable as thermalized hot DM (Sum m_nu constraint)
  - NOT viable as warm DM (too light by factor 1000)
  - POSSIBLY VIABLE if: (a) decouples very early (reduces N_eff contribution)
                        (b) very small mixing angle (never thermalizes)

The 4th spectral generation prediction NEEDS REVISION or an escape mechanism.
""")

# ============================================================
# PART 7: Summary
# ============================================================

print("=" * 65)
print("SUMMARY: 4TH SPECTRAL GENERATION AS DARK MATTER")
print("=" * 65)
print(f"""
m_4 = 5.4 eV (sterile neutrino from RESULT_044)

CONSTRAINTS:
  1. N_eff: thermalized → 6σ excluded; early decoupling → marginally OK
  2. Sum m_nu: thermalized → 5.46 eV >> 0.12 eV (45× Planck bound) → EXCLUDED
  3. Structure: too light for WDM; hot DM suppresses structure
  4. Lifetime: τ >> age of universe (stable, makes problem worse)

VERDICT:
  The 4th spectral generation ν_4 with m_4 = 5.4 eV is COSMOLOGICALLY EXCLUDED
  if it has normal thermal interactions (active-sterile mixing θ > few × 10^{{-4}}).

  If mixing angle θ_4 << 10^{{-4}}: ν_4 NEVER thermalizes → cosmologically safe
  but also NOT a dark matter candidate (Ω_ν4 ~ 0).

IMPLICATION FOR TOE v3:
  The 4th spectral generation is a genuine prediction (Clifford algebra structure)
  but its cosmological role is unclear. The spectral prediction m_4 = 5.4 eV
  may need to be revised (uncertainty in the Koide formula for the 4th generation).

  NEW FALSIFIABLE PREDICTION: If θ_4 is large enough, ν_4 would show up as:
  - A peak in reactor neutrino experiments (Bugey, SAGE, GALLEX anomaly?)
  - ΔN_eff ~ 0.066 (early decoupling)

  STATUS: OPEN PROBLEM in TOE v3.
""")
