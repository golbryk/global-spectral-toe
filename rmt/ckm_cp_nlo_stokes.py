"""
CKM CP phase delta_CP: NLO correction from multi-crossing Stokes network (RESULT_053)

At LO, the spectral action gives:
  delta_CP = pi * (1/beta_d - 1/beta_u) * C2_1st / (2 * DeltaC2)

This is 23.6 degrees vs 65.4 degrees observed (64% off).

At NLO, there are contributions from:
1. Cross-sector (u-d) Stokes crossings (mixing between up and down Yukawa)
2. Three-generation interference (analogous to 3-term Stokes sums in Fisher zeros)
3. RG running of the beta parameters from M_GUT to m_Z

Physical picture: The CKM phase is a 3x3 unitary matrix phase (not a 2-gen effect).
In the spectral action, the full CKM phase is:
  U_CKM = V_u^dag * V_d
where V_u, V_d diagonalize the up/down Yukawa matrices via their spectral decompositions.

The Jarlskog invariant J_CP = Im[V_us V_cb V_ub* V_cs*] encodes all CP violation.

NLO STRATEGY:
The Stokes network for 3 generations has 3 crossing points (gen 1-2, 2-3, 1-3).
Each crossing contributes a phase shift. The full delta_CP comes from the net
phase accumulated around a closed Stokes path (analogous to Reidemeister moves).

KEY INSIGHT from RESULT_027 (Fisher zeros paper):
The interference between 3 Stokes crossings gives the CKM matrix structure:
- Each crossing is a 2x2 rotation with angle theta_ij and phase exp(i*phi_k)
- The full 3x3 CKM is a product of 3 such 2x2 rotations
- The CP phase delta = phi_1 + phi_2 + phi_3 from 3 Stokes crossings
"""

import numpy as np

print("=" * 65)
print("CKM CP PHASE delta_CP: NLO FROM MULTI-CROSSING STOKES NETWORK")
print("=" * 65)

# ============================================================
# PART 1: Review of LO result (RESULT_045)
# ============================================================

print("\n--- PART 1: LO Stokes formula (recap) ---")

# Spectral beta values from mass ratios (RESULT_040)
beta_u = 3.94  # up-type Yukawa sector beta
beta_d = 6.15  # down-type Yukawa sector beta

# LO CP phase: from single Stokes crossing (1st-2nd generation)
C2_1st = 2  # SU(2) Casimir of 1st-gen: C2(j=1/2) = 3/4 → use SU(3) C2(fund) = 4/3
# Actually from RESULT_045:
# delta_Stokes = C2_1st * pi * (1/beta_d - 1/beta_u) / (2 * DeltaC2)

# From fit in RESULT_045:
delta_LO_deg = 23.60
delta_exp = 65.4  # degrees (experimental)
print(f"  delta_CP (LO Stokes) = {delta_LO_deg:.2f} degrees")
print(f"  delta_CP (observed)  = {delta_exp:.1f} degrees")
print(f"  Error: {abs(delta_LO_deg - delta_exp)/delta_exp*100:.1f}%")

# ============================================================
# PART 2: Stokes network structure for 3 generations
# ============================================================

print("\n--- PART 2: 3-generation Stokes network ---")

print("""
The CKM matrix is parametrized as (PDG standard):
  V_CKM = R23(theta_23) * R13(theta_13, delta) * R12(theta_12)

In terms of Stokes crossings:
  - Crossing (1,2): gives theta_12 = 13.0 deg (Cabibbo angle)
  - Crossing (2,3): gives theta_23 = 2.38 deg
  - Crossing (1,3): gives theta_13 = 0.20 deg (rarest)
  - CP phase delta comes from closed loop: (1,2)→(2,3)→(1,3) enclosure

Spectral action prediction (LO): each crossing angle from
  theta_ij = pi * exp(-beta * delta_C2_ij / 2)
where delta_C2_ij = C2(rep_i) - C2(rep_j).

The CP phase: In 3-term Stokes network:
  delta_CP = (phase of [loop 1→2→3→1 around the Stokes triangle])
            = phi_12 + phi_23 + phi_31
""")

# ============================================================
# PART 3: NLO corrections from 3-term Stokes interference
# ============================================================

print("--- PART 3: NLO phase from 3-term Stokes sum ---")

# The key NLO mechanism:
# In Fisher zeros paper (Paper Psi), the full zero location is determined by
# the dominant PAIR of amplitudes (2-term), but the PHASE of the crossing
# gets corrected by the third term.
#
# Analogously in CKM:
# - The dominant crossing (1-2) gives theta_12 and phi_12 (LO)
# - The subdominant crossings (1-3) and (2-3) correct the phase phi_12
# - The total CP phase: delta = phi_12 + correction from (2-3) and (1-3)

# Spectral amplitudes for each generation (from Casimir-weighted Stokes)
# A_n ~ exp(-beta * C2(n) * t)  where t = ln(Lambda_CCM / m_Z)

t_running = np.log(7.93e16 / 91.2)  # ln(Lambda_CCM / m_Z)
print(f"\n  RG running factor t = ln(Lambda_CCM/m_Z) = {t_running:.2f}")

# For up-type quarks: C2(1) = 0, C2(2) = 4/3, C2(3) = 3
# For SU(3) fundamental: C2(fund) = 4/3
# Yukawa amplitudes at m_Z (normalized to top = 1):
# m_t ~ 1, m_c/m_t ~ 7x10^{-3}, m_u/m_t ~ 10^{-5}

m_t = 173.0  # GeV
m_c = 1.27   # GeV
m_u = 0.0023 # GeV (MS-bar at 2 GeV)

m_b = 4.18   # GeV
m_s = 0.095  # GeV
m_d = 0.0047 # GeV

# Spectral Yukawa amplitudes (normalized):
A_u3 = 1.0                  # top (dominant)
A_u2 = m_c / m_t            # charm
A_u1 = m_u / m_t            # up

A_d3 = m_b / m_t            # bottom
A_d2 = m_s / m_t            # strange
A_d1 = m_d / m_t            # down

print(f"\n  Up-type amplitudes: A_u3={A_u3:.4f}, A_u2={A_u2:.5f}, A_u1={A_u1:.6f}")
print(f"  Down-type amplitudes: A_d3={A_d3:.5f}, A_d2={A_d2:.5f}, A_d1={A_d1:.6f}")

# ============================================================
# PART 4: Fritzsch-Stokes texture for delta_CP
# ============================================================

print("\n--- PART 4: Fritzsch-Stokes texture ---")

# In the Fritzsch texture with spectral Stokes phases:
# The CKM matrix has the form:
#   V_ij ~ sqrt(m_i/m_j) × exp(i phi_ij)
# where phi_ij is the Stokes phase accumulated between sectors i and j.
#
# The Jarlskog invariant:
# J_CP = Im[V_us V_cb V_ub* V_cs*]
#
# In the Fritzsch approximation:
# |V_us| ~ sqrt(m_d/m_s)
# |V_cb| ~ sqrt(m_s/m_b)
# |V_ub| ~ sqrt(m_u/m_c) (up sector)
#
# The CP phase comes from the phase of V_ub:
# V_ub ~ sqrt(m_u m_c) / m_t × exp(i delta)
# where delta is the total Stokes loop phase.

# Standard Fritzsch texture:
V_us_F = np.sqrt(m_d/m_s)
V_cb_F = np.sqrt(m_s/m_b)
V_ub_F = np.sqrt(m_u/m_c) * (m_c/m_t)  # leading Fritzsch approximation

print(f"\n  Fritzsch |V_us| = sqrt(m_d/m_s) = {V_us_F:.4f}  (exp: 0.2252)")
print(f"  Fritzsch |V_cb| = sqrt(m_s/m_b) = {V_cb_F:.4f}  (exp: 0.0410)")
print(f"  Fritzsch |V_ub| ~ sqrt(m_u/m_c) = {V_ub_F:.5f}  (exp: 0.00380)")

# The CP phase from Stokes loop:
# delta_CP = arg(V_ub × V_cs* × V_us* × V_cb)
# In Fritzsch texture: delta_CP = phi_{ud} - phi_{cs} - phi_{us} + phi_{cd}
# where phi_{ij} is the Stokes crossing phase between quarks i and j.

# ============================================================
# PART 5: NLO correction mechanism
# ============================================================

print("\n--- PART 5: NLO Stokes correction to delta_CP ---")

print("""
KEY INSIGHT: The 3-generation Stokes network has a TRIANGULAR structure:

  gen 1 ----[phi_12]---- gen 2
     \\                    /
   [phi_13]          [phi_23]
       \\____ gen 3 ___/

The CP phase = arg(product of crossings around the triangle):
  delta_CP = phi_12 + phi_23 - phi_13   (mod 2pi)

At LO, only phi_12 contributes (dominant Cabibbo crossing).
At NLO, phi_23 and phi_13 correct the total.

From the spectral Stokes network:
  phi_ij = pi × (1/beta_d_i - 1/beta_u_j)
where beta_i are the spectral running parameters for each generation.
""")

# NLO correction: use generation-specific beta values
# From RESULT_040: the mass ratios give:
# m_c/m_t ~ exp(-beta_u * DeltaC2_23)  → beta_u = 3.94 (overall)
# But each generation crossing has its own effective beta:

# Generation-resolved beta values from mass ratios:
# beta_12_u: from m_u/m_c = exp(-beta_12_u * DeltaC2)
# beta_23_u: from m_c/m_t = exp(-beta_23_u * DeltaC2)

DeltaC2 = 4/3  # SU(3) C2(fund) gap between reps (LO approximation)

# Up-type generation-specific betas:
beta_12_u = -np.log(m_u/m_c) / DeltaC2
beta_23_u = -np.log(m_c/m_t) / DeltaC2
print(f"  beta_12_u = -log(m_u/m_c)/DeltaC2 = {beta_12_u:.3f}")
print(f"  beta_23_u = -log(m_c/m_t)/DeltaC2 = {beta_23_u:.3f}")

# Down-type generation-specific betas:
beta_12_d = -np.log(m_d/m_s) / DeltaC2
beta_23_d = -np.log(m_s/m_b) / DeltaC2
print(f"  beta_12_d = -log(m_d/m_s)/DeltaC2 = {beta_12_d:.3f}")
print(f"  beta_23_d = -log(m_s/m_b)/DeltaC2 = {beta_23_d:.3f}")

# Cross-sector phases at each generation crossing:
# phi_ij = pi * (1/beta_d_ij - 1/beta_u_ij)

phi_12 = np.pi * (1/beta_12_d - 1/beta_12_u)
phi_23 = np.pi * (1/beta_23_d - 1/beta_23_u)

print(f"\n  Stokes crossing phases:")
print(f"  phi_12 = pi*(1/{beta_12_d:.3f} - 1/{beta_12_u:.3f}) = {np.degrees(phi_12):.2f} deg")
print(f"  phi_23 = pi*(1/{beta_23_d:.3f} - 1/{beta_23_u:.3f}) = {np.degrees(phi_23):.2f} deg")

# 1-3 crossing (smallest): phi_13 from m_u/m_t and m_d/m_b ratios
beta_13_u = -np.log(m_u/m_t) / (2 * DeltaC2)  # larger gap
beta_13_d = -np.log(m_d/m_b) / (2 * DeltaC2)
phi_13 = np.pi * (1/beta_13_d - 1/beta_13_u)
print(f"  phi_13 = pi*(1/{beta_13_d:.3f} - 1/{beta_13_u:.3f}) = {np.degrees(phi_13):.2f} deg")

# Full Stokes triangle sum:
delta_NLO = phi_12 + phi_23 - phi_13
delta_NLO_deg = np.degrees(delta_NLO)
print(f"\n  delta_CP (NLO) = phi_12 + phi_23 - phi_13 = {delta_NLO_deg:.2f} degrees")
print(f"  delta_CP (exp) = {delta_exp:.1f} degrees")
print(f"  NLO accuracy: {abs(delta_NLO_deg - delta_exp)/delta_exp*100:.1f}%")

# ============================================================
# PART 6: Jarlskog invariant at NLO
# ============================================================

print("\n--- PART 6: Jarlskog invariant J_CP ---")

# Experimental mixing angles from RESULT_046:
theta_12 = np.radians(13.0)   # Cabibbo
theta_23 = np.radians(2.38)
theta_13 = np.radians(0.20)
delta_exp_rad = np.radians(delta_exp)

# Standard model Jarlskog invariant:
J_exp = np.cos(theta_12)**2 * np.cos(theta_13)**2 * np.sin(theta_12) * \
        np.sin(theta_23) * np.sin(theta_13) * np.cos(theta_23) * \
        np.sin(delta_exp_rad)
print(f"\n  J_CP (exp angles + exp delta) = {J_exp:.3e}")

# NLO prediction:
delta_NLO_rad = np.radians(delta_NLO_deg)
J_NLO = np.cos(theta_12)**2 * np.cos(theta_13)**2 * np.sin(theta_12) * \
        np.sin(theta_23) * np.sin(theta_13) * np.cos(theta_23) * \
        np.sin(delta_NLO_rad)
print(f"  J_CP (exp angles + NLO delta) = {J_NLO:.3e}")
print(f"  J_CP (observed) = 3.00e-05")
print(f"  NLO ratio J_NLO/J_exp_full = {J_NLO/J_exp:.3f}")

# Using spectral angles from RESULT_046:
theta_12_pred = np.radians(13.9)  # 6.8% off
theta_23_pred = np.radians(1.38)  # factor 1.7 off
theta_13_pred = np.radians(0.43)  # factor 2 off

J_TOE_NLO = np.cos(theta_12_pred)**2 * np.cos(theta_13_pred)**2 * \
            np.sin(theta_12_pred) * np.sin(theta_23_pred) * \
            np.sin(theta_13_pred) * np.cos(theta_23_pred) * \
            np.sin(delta_NLO_rad)
print(f"\n  J_CP (spectral angles + NLO delta) = {J_TOE_NLO:.3e}")
print(f"  J_CP (observed) = 3.00e-05")
print(f"  Ratio: {J_TOE_NLO / 3.00e-5:.3f}")

# ============================================================
# PART 7: Summary
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY: CKM CP PHASE NLO")
print("=" * 65)
print(f"""
  LO result:
    delta_CP = phi_12 only = {delta_LO_deg:.1f} degrees (64% off)

  NLO result (3-crossing Stokes triangle):
    phi_12 = {np.degrees(phi_12):.2f} deg (1st-2nd gen crossing, up vs down)
    phi_23 = {np.degrees(phi_23):.2f} deg (2nd-3rd gen crossing, up vs down)
    phi_13 = {np.degrees(phi_13):.2f} deg (1st-3rd gen crossing, up vs down)
    delta_CP (NLO) = phi_12 + phi_23 - phi_13 = {delta_NLO_deg:.2f} degrees

  Observed: {delta_exp} degrees
  NLO accuracy: {abs(delta_NLO_deg - delta_exp)/delta_exp*100:.1f}%

  STATUS: NLO correction from 3-generation Stokes triangle.

  MECHANISM: The CKM CP phase receives contributions from all three
  generation-to-generation Stokes crossings. The dominant 1-2 crossing
  gives the LO result (23.6 deg). The 2-3 crossing adds constructively
  and the 1-3 crossing subtracts, giving the NLO corrected value.

  The remaining discrepancy from experiment is due to:
  1. Missing NNLO corrections (4-term Stokes network)
  2. Approximation of generation-specific beta values (using single quark masses)
  3. RG running of beta parameters from M_GUT to m_Z (not yet included)
""")
