# RMT Numerical Audit Plan

**Global Spectral Theory of Everything** · accompanies Zenodo Version 3
(DOI [10.5281/zenodo.20374769](https://doi.org/10.5281/zenodo.20374769)),
release commit `ef5c93a`.

## Purpose and status of this document

This is a **plan**, not a completed audit, and not a claim. It responds to a
reviewer-style critique (raised by an external LLM, "Gemini") that the RMT
intermediate-statistics result should be stress-tested for numerical
robustness before being treated as load-bearing.

The current public claim is deliberately modest: the fundamental excitation
operator shows **intermediate (pseudo-integrable) level-spacing statistics**
between Poisson and GOE (spacing-ratio ⟨r⟩ ≈ 0.43, window ≈ (0.386, 0.530)),
and increasing system size or coupling does not generically restore full GOE
chaos. The committed evidence is `rmt/scaling_tests.py` (GPU sweep),
`rmt/rmt_extended_analysis.py` (CPU Brody / semi-Poisson fits), and the raw log
`rmt/logs/scaling_tests_2026-01-11.txt`.

Where a check below has **not** been run to completion, that is stated. Nothing
here upgrades the claim; it only defines what a full audit would establish.

---

## Audit dimensions

### 1. Floating-point stability
- **Concern.** Does ⟨r⟩ depend on float32 vs float64, or on summation order?
- **Plan.** Recompute spectra and ⟨r⟩ in float64 and float32; report the
  difference. The committed operator builds in `float64`; confirm the GPU
  (CuPy) and CPU (NumPy) paths agree to within eigensolver tolerance.
- **Status.** Partially covered (float64 throughout); explicit float32 vs
  float64 delta table **not yet produced**.

### 2. Eigensolver residuals
- **Concern.** Are the eigenpairs accurate enough that small spacings are
  trustworthy (small spacings dominate ⟨r⟩)?
- **Plan.** For representative N, compute the residual ‖H v − λ v‖ per
  eigenpair and the symmetry residual ‖H − Hᵀ‖; confirm residuals are far
  below the typical nearest-neighbour spacing. Compare `eigvalsh` (used) with a
  full `eigh` and with an independent LAPACK driver.
- **Status.** **Not yet produced** as a committed residual report.

### 3. Unfolding sensitivity
- **Concern.** The ratio statistic ⟨r⟩ is unfolding-free **by construction**,
  which is its main virtue; but P(s) and Brody-β fits **do** depend on the
  unfolding procedure.
- **Plan.** Cross-check the ⟨r⟩ conclusion against P(s) computed under at least
  two unfolding schemes (polynomial fit of the integrated density vs Gaussian
  broadening). The ⟨r⟩ result should be invariant; only the P(s)/β fits should
  shift. Report both.
- **Status.** ⟨r⟩ is unfolding-independent (covered in principle);
  multi-scheme P(s) comparison **not yet produced**.

### 4. Finite-size scaling
- **Concern.** Is the intermediate value an artifact of finite N that would
  drift to GOE (or Poisson) as N → ∞?
- **Plan.** The committed log already scans N ∈ {200, 400, 800, 1200}; extend
  to larger N (GPU permitting) and fit ⟨r⟩(N) for a trend. The committed data
  show **no drift toward GOE** with N at fixed α — that is the basis of the
  claim — but a formal extrapolation with error bars is **not yet produced**.
- **Status.** Trend present in committed data; quantitative extrapolation
  **pending**.

### 5. Seed / ensemble dependence
- **Concern.** Is ⟨r⟩ ≈ 0.43 stable across RNG seeds and ensemble draws, or a
  lucky seed?
- **Plan.** Repeat over many seeds; report mean and standard error of ⟨r⟩ for
  the TOE operator and for the Poisson/GOE references. Note the Arnold-cat
  phase is **deterministic** (fixed x₀, y₀), so for the TOE operator the
  ensemble variation comes from N and α, not from the phase; document this and
  optionally vary (x₀, y₀).
- **Status.** Single committed seed (123). **Multi-seed error bars not yet
  produced.**

### 6. CPU / GPU comparison
- **Concern.** Do the CuPy (GPU) and NumPy (CPU) paths agree?
- **Plan.** Run the same N, α, seed on both backends and confirm ⟨r⟩ agrees to
  eigensolver tolerance. The smoke test `scripts/v3_smoke_tests/run_rmt_smoke.py`
  is the CPU reference implementation of the exact same r-statistic.
- **Status.** Smoke test confirms the CPU path reproduces the qualitative
  Poisson < intermediate < GOE separation; a numeric CPU-vs-GPU equality table
  at matched (N, α, seed) is **not yet produced**.

### 7. Operator-structure universality
- **Concern.** Is the intermediate regime specific to the ring Laplacian +
  Arnold-cat phase, or robust across constructions?
- **Plan.** `rmt/rmt_extended_analysis.py` already varies Laplacian structures
  and chaotic maps and fits Brody-β / semi-Poisson; consolidate its output into
  a committed summary table.
- **Status.** Script exists; a committed consolidated universality table is
  **pending**.

---

## Expected future audit (deliverable)

A complete audit would commit a single reproducible report containing, per
(N, α): ⟨r⟩ with multi-seed error bars, eigensolver residual bounds, float32
vs float64 deltas, CPU-vs-GPU agreement, P(s) under ≥ 2 unfolding schemes, and
a finite-size extrapolation of ⟨r⟩(N). Until that report exists, the RMT claim
should be read as **well-supported at the tested sizes/seeds but not yet
formally audited** for the robustness dimensions above.

This plan does **not** assert the audit's outcome. It does not bear on the
neutrino-ordering status (open empirical boundary) and makes no TOE-completeness
claim.

*See also: [`V3_REPRODUCIBILITY.md`](V3_REPRODUCIBILITY.md),
[`V3_RELEASE_NOTES.md`](V3_RELEASE_NOTES.md).*
