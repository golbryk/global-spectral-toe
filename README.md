# Global Spectral Theory of Everything

**Author:** Grzegorz Olbryk · **Contact:** g.olbryk@gmail.com

Previous version (v2) on Zenodo: DOI [10.5281/zenodo.18213696](https://doi.org/10.5281/zenodo.18213696)

---

## Status notice — versioning (read first)

Two version numberings appear in this project; they are **not** the same.

### Zenodo release versions

The citable public records of the *Global Spectral Theory of Everything*
synthesis document:

- **Version 1** — DOI 10.5281/zenodo.17961030 (superseded)
- **Version 2** — DOI 10.5281/zenodo.18213696 (current published record)
- **Version 3 — forthcoming** — a new, single-file synthesis/scope PDF
  (`global_spectral_toe_v3.pdf`) continuing Version 2 with the corrected,
  layered scientific status (see [Current scientific status](#current-scientific-status)
  below). The Version 3 DOI is pending and will be added once Zenodo mints it.

### Internal programme drafts

Working LaTeX sources under `theory/`.

> **Deprecation notice.** The internal draft `theory/toe_v3_stokes_foundation.tex`
> is an **obsolete, superseded internal draft**. It is **not** the Zenodo
> Version 3 release document and must not be cited as the final Version 3. In
> particular, its flat statement that the framework "predicts the normal
> neutrino hierarchy and an inverted result would falsify the theory" is
> superseded by the layered status in the new Version 3 synthesis (and in
> `docs/STATUS_AND_LIMITS.md`). See `theory/DEPRECATED_toe_v3_stokes_foundation.md`.

The companion mathematical-physics paper `theory/toe_v4_CMP_submission.tex`
(**Stokes–RG Correspondence in Lattice Gauge Theory**, 18pp) does **not** claim
to be a Theory of Everything. It establishes only the Stokes–RG correspondence:
the lattice mass gap equals the distance from the real coupling axis to the
dominant Stokes curve of the partition function, connecting constructive QFT
with Lee–Yang zero theory. Its rigorous engine lives in the companion repository
`hsmi-yang-mills`.

---

## Current scientific status

*Authoritative summary. Full detail and falsifier map:* `docs/STATUS_AND_LIMITS.md`.

- The framework is a **background-independent, relational, global-spectral
  foundational construction**. It is **not** a complete Theory of Everything,
  **not** a phenomenological replacement for the Standard Model or QCD, and
  makes **no "TOE solved / 100%" claim**.

- **RMT result stands.** The fundamental operators show intermediate
  (pseudo-integrable) statistics between Poisson and GOE; universal quantum
  chaos is **not** obtained at the fundamental level. This falsifies a direct
  hadronic reading and is treated as a positive structural constraint.

### Neutrino mass ordering — layered status

None of the following constitutes a solution:

1. **Local derivation: open.** Within the framework's local construction the
   relevant Majorana scale is a free parameter; the ordering is **not derived
   as a local theorem** — it is an empirical boundary.
2. **Minimal Majorana-sector branch.** The minimal / symmetric choice favors
   **Normal Ordering**, but this is a model-selection branch — **not forced**.
3. **Global / swampland argument.** A global-spectral (quantum-gravity)
   consistency argument **convergently favors Normal Ordering**, but it is
   **conjectural** (it rests on unproven swampland-type conjectures).

An inverted-ordering measurement would falsify the *predictive (minimal
Majorana) branch* and the *global-spectral Normal-Ordering argument* — it would
**not** automatically falsify the entire local framework.

### Deprecated / retracted internal leads

Recorded for honesty, **not** used as final claims: an earlier neutrino
mass-ratio "inverted-ordering exclusion" argument was **retracted** as
non-robust; several earlier internal mechanism routes for forcing the ordering
were tried and **abandoned**. None of these are claimed as results.

> Note: the RMT level-spacing window `r ∈ (0.386, 0.530)` is a *different*
> quantity and still stands.

---

## Companion repository: Yang-Mills mass gap

The Yang-Mills mass gap **candidate proof** (79pp, non-perturbative RG via
free-energy convexity; W5 and large-field Gribov open) is maintained in a
separate repository:

- https://github.com/golbryk/hsmi-yang-mills

That repository contains:

- `mass_gap_rigorous.tex` — 4D SU(N) mass gap candidate proof, all N ≥ 2, all g² > 0
- Fisher zeros programme (Papers Pi, Rho, Chi, Psi, Omega, Sigma, Tau, Xi)
- Stokes concentration theorem (Paper Psi) — the mathematical engine underlying
  the Stokes–RG correspondence in this repository

The superseded Version 1 remains archived under DOI
[10.5281/zenodo.17961030](https://doi.org/10.5281/zenodo.17961030).

---

## Global Spectral Theory of Everything — Version 2

This repository accompanies the Zenodo record:

> Grzegorz Olbryk, *Global Spectral Theory of Everything — Version 2: Scope,
> Validation, and Limits*, Zenodo (2026),
> DOI [10.5281/zenodo.18213696](https://doi.org/10.5281/zenodo.18213696).

### Documentation structure

The documentation is intentionally split into three complementary parts:

1. **Core mathematical construction** — the complete axiomatic and mathematical
   formulation is provided in the LaTeX source `global_spectral_toe_core.tex`,
   which defines the foundational axioms, spectral functional, vacuum structure,
   emergent mass and time, no-go results, and example numerical fixed points.
2. **Validation and scope (Zenodo PDF)** — the Zenodo record contains an
   overview document describing scope and limits of applicability, falsification
   and validation tests, Random Matrix Theory (RMT) results, and interpretation
   of negative results.
3. **Reproducible numerical tests (this repository)** — all numerical
   experiments, raw logs, and figure-generation scripts used in the validation.

---

## Repository scope

The purpose of this repository is to:

- provide a transparent reference implementation of a global spectral–relational
  framework,
- document explicit falsification and validation tests,
- allow independent reproduction of Random Matrix Theory analyses,
- clearly delineate the limits of the construction.

This repository is **not**:

- a phenomenological model of hadrons,
- a replacement for QCD or effective field theory,
- a general-purpose simulation framework.

---

## Random Matrix Theory (RMT)

The directory `rmt/` contains all Random Matrix Theory analyses used to test the
spectral properties of the theory. These tests demonstrate that:

- the fundamental vacuum and minimal excitation operators exhibit
  pseudo-integrable / mixed statistics,
- universal GOE behavior is **not** obtained at the fundamental level,
- increasing system size or coupling does not generically restore full quantum
  chaos.

These results falsify interpretations of the theory as a direct effective
description of hadronic spectra, while remaining consistent with its intended
fundamental scope. All RMT tests are reproducible from raw logs without data
selection.

---

## Requirements

- Python ≥ 3.9
- NumPy
- CuPy with CUDA support (for GPU-based RMT tests)
- matplotlib (for figure generation)

## Running the RMT tests

From the repository root:

```bash
cd rmt
python scaling_tests.py
python generate_figures.py
```

The generated figures will appear in `theory/figures/`.

---

## Reproducibility

All numerical experiments are deterministic up to explicitly stated random seeds
and ensemble constructions. Reproducibility guarantees numerical consistency of
the implementation, not physical uniqueness or phenomenological completeness.
Raw outputs are provided to avoid selection bias.

## Use of computational assistance

Parts of the numerical experimentation, code generation, and manuscript
preparation were performed with the assistance of large language models (LLMs),
under the direct supervision of the author. All scientific decisions,
interpretations, and validations remain the responsibility of the author.

## Citation

If you use or reference this work, please cite:

> Grzegorz Olbryk, *Global Spectral Theory of Everything — Version 2: Scope,
> Validation, and Limits*, Zenodo (2026),
> DOI [10.5281/zenodo.18213696](https://doi.org/10.5281/zenodo.18213696).

A machine-readable citation entry is provided in `CITATION.cff`.

## License

MIT License (see `LICENSE` file).

## Contact

Grzegorz Olbryk · g.olbryk@gmail.com
