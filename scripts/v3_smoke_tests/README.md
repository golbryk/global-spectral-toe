# Version 3 smoke tests

Small, fast, self-contained sanity checks that accompany the Zenodo
**Version 3** synthesis (DOI [10.5281/zenodo.20374769](https://doi.org/10.5281/zenodo.20374769)).

These are *smoke tests* of the numerical tooling and statistics — they are
**not** a reproduction of the full published RMT result.

## `run_rmt_smoke.py`

### What it does
- Generates small synthetic **Poisson** and **GOE** spectra (pure NumPy).
- Computes the adjacent-gap-ratio statistic ⟨r⟩ using the exact same
  definition as `rmt/ensemble_references.py`.
- Verifies the references land near their textbook values and are clearly
  separated: Poisson ⟨r⟩ ≈ 0.386 **<** GOE ⟨r⟩ ≈ 0.531.
- Builds a tiny CPU copy of the TOE excitation operator (ring Laplacian +
  Arnold-cat diagonal phase, ported from `rmt/excitation_operator.py`) and
  reports its ⟨r⟩, which should fall in the **intermediate /
  pseudo-integrable** window between the two references.

### What it does NOT do
- It does **not** require a GPU (the full sweep `rmt/scaling_tests.py` uses
  CuPy).
- It does **not** run the size/coupling scan, unfolding, finite-size scaling,
  or Brody / semi-Poisson fits.
- It does **not**, on its own, establish the published intermediate-statistics
  claim. The full audit scope is in
  [`docs/RMT_NUMERICAL_AUDIT_PLAN.md`](../../docs/RMT_NUMERICAL_AUDIT_PLAN.md).
- It says **nothing** about neutrino mass ordering, the A4 question, or any
  "TOE complete" claim.

### How to run
```bash
python3 scripts/v3_smoke_tests/run_rmt_smoke.py
# options: --N 600  --alpha 0.30  --seed 123
```

Requirements: Python ≥ 3.9 and NumPy only. No CuPy / CUDA / matplotlib.

### Expected output
A small table; the run is considered OK when:
- Poisson ⟨r⟩ falls in ≈ (0.34, 0.44),
- GOE ⟨r⟩ falls in ≈ (0.47, 0.58),
- Poisson ⟨r⟩ < GOE ⟨r⟩.

Exit code `0` = sanity checks passed. Exit code `1` = a reference statistic
fell outside its generous sanity band (a tooling problem, not a physics
result). The toy-size TOE operator typically reports ⟨r⟩ ≈ 0.43–0.47, between
the two references, but this is an illustration of the tool, not a claim.
