# Version 3 — Reproducibility Map

**Global Spectral Theory of Everything** · DOI [10.5281/zenodo.20374769](https://doi.org/10.5281/zenodo.20374769)
Release commit `ef5c93a` · companion `hsmi-yang-mills` `0fea9c5` · date 2026-05-25

This document maps each public claim associated with Version 3 to the tracked
files in this repository that bear on it, the command to exercise them, the
expected output, a reproducibility status, and the known limitations.

**Honest scope.** Version 3 is a synthesis/scope/status PDF, not a full
research archive. Some claims have a runnable script in this repo; others are
**conceptual / model-selection / global-QG** statements with **no local
reproduction script**, and are marked as such. The author does **not** claim a
complete TOE and does **not** claim the neutrino-ordering ("A4") question is
solved.

All files referenced below are **tracked** files already in the repository at
commit `ef5c93a`. No quarantined or untracked material is relied upon.

---

## Claim ↔ evidence table

### 1. RMT: intermediate / pseudo-integrable statistics (stands)

| | |
|---|---|
| **Claim** | The fundamental vacuum / minimal-excitation operator shows **intermediate (pseudo-integrable) statistics** between Poisson and GOE; universal GOE quantum chaos is **not** obtained at the fundamental level, and increasing N or coupling α does not generically restore it. |
| **Evidence / files** | `rmt/excitation_operator.py`, `rmt/ensemble_references.py`, `rmt/scaling_tests.py` (GPU/CuPy sweep); `rmt/rmt_extended_analysis.py` (NumPy/SciPy: Brody/semi-Poisson fits); raw log `rmt/logs/scaling_tests_2026-01-11.txt`; `scripts/v3_smoke_tests/run_rmt_smoke.py` (CPU sanity). |
| **Command** | Full (GPU): `cd rmt && python scaling_tests.py`. Extended (CPU): `python rmt/rmt_extended_analysis.py`. Sanity (CPU, no GPU): `python3 scripts/v3_smoke_tests/run_rmt_smoke.py`. |
| **Expected output** | Spacing-ratio statistic ⟨r⟩ for the TOE operator in the window ≈ (0.386, 0.530), mean ≈ 0.43 — between Poisson (≈ 0.386) and GOE (≈ 0.531). Smoke test prints the same separation at toy size and exits 0. |
| **Status** | **Reproducible.** Full sweep needs CuPy/CUDA; smoke test reproduces the qualitative separation on CPU. Raw log committed without selection. |
| **Limitations** | Diagnostic operator is a ring Laplacian + Arnold-cat phase, one representative construction. No exhaustive unfolding / finite-size extrapolation in the committed log. See `docs/RMT_NUMERICAL_AUDIT_PLAN.md`. The `r` here is the spacing-ratio statistic, **not** any neutrino mass ratio. |

### 2. Stokes–RG correspondence (materials / pointer)

| | |
|---|---|
| **Claim** | The lattice mass gap equals the distance from the real coupling axis to the dominant Stokes curve of the partition function — connecting the constructive mass-gap programme with Lee–Yang / Stokes-zero theory. |
| **Evidence / files** | `theory/toe_v4_CMP_submission.tex` + `theory/toe_v4_CMP_submission.pdf` ("Stokes–RG Correspondence in Lattice Gauge Theory", 18pp); Stokes-related numerics `rmt/toe_stokes_verification.py`, `rmt/cp_stokes_phases.py`, `rmt/ckm_delta_cp_stokes.py`, `rmt/ckm_cp_nlo_stokes.py`, `rmt/strong_cp_stokes.py`. |
| **Command** | Read the manuscript PDF/TeX. The rigorous engine is in the companion repo (below); the scripts here are exploratory and may require CuPy. |
| **Expected output** | The CMP submission states what is **proved unconditionally** (the Stokes–RG correspondence) vs **conditional on the mass-gap programme**. |
| **Status** | **Documented, partially conditional.** The unconditional correspondence is established in the manuscript; spacetime/mass-gap consequences are conditional. |
| **Limitations** | This repository holds the manuscript and exploratory scripts, **not** the rigorous proof engine — that is the companion repo. Some scripts need a GPU and are not part of the minimal smoke set. |

### 3. HSMI / Yang–Mills companion (pointer)

| | |
|---|---|
| **Claim** | The rigorous lattice-gauge mathematics (Yang–Mills mass-gap candidate proof, Fisher zeros, Stokes concentration theorem) underpins the Stokes–RG machinery. |
| **Evidence / files** | External companion repository `hsmi-yang-mills` at commit `0fea9c5` (e.g. `mass_gap_rigorous.tex`, Fisher-zero papers, Stokes concentration / Paper Psi). |
| **Command** | See that repository: <https://github.com/golbryk/hsmi-yang-mills>. |
| **Expected output** | Independent rigorous results; the mass-gap proof is a **candidate** (W5 and large-field Gribov open). |
| **Status** | **Pointer only.** Not reproduced from within this repository. |
| **Limitations** | The two repositories are linked only through the Stokes/Fisher-zero machinery; none of the foundational claims here are assumed in the rigorous results there. |

### 4. Neutrino mass ordering — layered status (no local reproduction script)

| | |
|---|---|
| **Claim** | The ordering status is **layered**: (a) local construction — Majorana scale free, ordering **not** a local theorem (empirical boundary, open); (b) minimal/symmetric Majorana branch favors Normal Ordering (model-selection, not forced); (c) a global-spectral / swampland argument convergently favors Normal Ordering (conjectural). |
| **Evidence / files** | `docs/STATUS_AND_LIMITS.md` §4 (authoritative wording). Model-level exploratory scripts exist (`rmt/neutrino_hierarchy_spectral.py`, `rmt/neutrino_seesaw_spectral.py`, `rmt/majorana_phases_spectral.py`) but represent the **model-selection branch**, not a local derivation of the ordering. |
| **Command** | **None that derives the ordering.** The exploratory scripts illustrate the model-selection branch only. |
| **Expected output** | No script outputs "the ordering is X as a theorem." The status is conceptual / model-selection / global-QG. |
| **Status** | **Not locally reproduced — by design.** This is an **open empirical boundary**, not a result. There is **no claim that A4 / the ordering is solved.** |
| **Limitations** | Resolution rests on experiment (JUNO, DUNE, cosmology / DESI Σmν). An inverted-ordering measurement would falsify the predictive branch and the global-spectral argument, but **not** the entire local framework. |

### 5. Deprecated / retracted internal leads (not reproduced, not active)

| | |
|---|---|
| **Claim** | Recorded for honesty only — **not** used as final claims. |
| **Evidence / files** | `docs/STATUS_AND_LIMITS.md` §5; `theory/DEPRECATED_toe_v3_stokes_foundation.md`; deprecated draft `theory/toe_v3_stokes_foundation.tex` / `.pdf`. |
| **Command** | None — these are explicitly **not** active. |
| **Expected output** | n/a. |
| **Status** | **Retracted / deprecated.** An earlier neutrino mass-ratio "inverted-ordering exclusion" argument was **retracted** as non-robust; several earlier internal mechanism routes to force the ordering were **tried and abandoned**. |
| **Limitations** | Listed for a complete public record; none is a result and none should be cited as one. |

---

## How to reproduce the minimal smoke set

```bash
# CPU-only, no GPU required, < a few seconds:
python3 scripts/v3_smoke_tests/run_rmt_smoke.py
```

This validates the r-statistic tooling and shows the Poisson < intermediate <
GOE separation at toy size. It is a **sanity check of the tools**, not a
reproduction of the full published RMT sweep. See
[`../scripts/v3_smoke_tests/README.md`](../scripts/v3_smoke_tests/README.md).

## Environment notes

- Python ≥ 3.9.
- Smoke test: **NumPy only** (no CuPy/CUDA/matplotlib).
- Full RMT sweep (`rmt/scaling_tests.py`): **CuPy + CUDA** (GPU).
- Extended RMT analysis (`rmt/rmt_extended_analysis.py`): NumPy + SciPy (CPU).
- No `requirements.txt` / `setup.py`; dependencies are installed manually.

*See also: [`V3_RELEASE_NOTES.md`](V3_RELEASE_NOTES.md),
[`RMT_NUMERICAL_AUDIT_PLAN.md`](RMT_NUMERICAL_AUDIT_PLAN.md),
[`../releases/v3_manifest.json`](../releases/v3_manifest.json).*
