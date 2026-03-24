Previous version (v2) on Zenodo DOI:
https://doi.org/10.5281/zenodo.18213696

======================================================================
STATUS NOTICE
======================================================================

This repository contains **Versions 2, 3, and 4** of the Global Spectral
programme.

**Version 2** (Zenodo): Original spectral TOE with RMT validation.
**Version 3** (`toe_v3_stokes_foundation.tex`): Stokes foundation —
resolves the RMT paradox, derives mass mechanism from transfer matrix,
and connects to the Yang-Mills mass gap programme.
**Version 4** (`toe_v4_CMP_submission.tex`): **Stokes-RG Correspondence
in Lattice Gauge Theory** — focused mathematical physics paper (18pp),
submission-ready for Communications in Mathematical Physics. Score: 8/10
after 10 rounds of adversarial builder-critic review.

The v4 paper does NOT claim to be a Theory of Everything. It establishes
the Stokes-RG correspondence: the lattice mass gap equals the distance
from the real coupling axis to the dominant Stokes curve of the partition
function, connecting constructive QFT with Lee-Yang zero theory.

======================================================================
COMPANION REPOSITORY: YANG-MILLS MASS GAP
======================================================================

The rigorous Yang-Mills mass gap proof (79pp, non-perturbative RG via
free-energy convexity) is maintained in a separate repository:

https://github.com/golbryk/hsmi-yang-mills

That repository contains:
- `mass_gap_rigorous.tex` — 4D SU(N) mass gap proof, all N>=2, all g^2>0
- Fisher zeros programme (Papers Pi, Rho, Chi, Psi, Omega, Sigma, Tau, Xi)
- Stokes concentration theorem (Paper Psi) — the mathematical engine
  underlying the Stokes-RG correspondence in this repository

The superseded Version 1 remains archived under:
https://doi.org/10.5281/zenodo.17961030

======================================================================
GLOBAL SPECTRAL THEORY OF EVERYTHING — VERSION 2
======================================================================

Author: Grzegorz Olbryk  
Contact: g.olbryk@gmail.com  

This repository accompanies the Zenodo record:

Grzegorz Olbryk  
*Global Spectral Theory of Everything — Version 2: Scope, Validation,
and Limits*  
Zenodo (2026)  
DOI: https://doi.org/10.5281/zenodo.18213696

----------------------------------------------------------------------
DOCUMENTATION STRUCTURE
----------------------------------------------------------------------

The documentation is intentionally split into three complementary parts:

1. **Core mathematical construction**  
   The complete axiomatic and mathematical formulation of the theory
   is provided in the LaTeX source:

   `global_spectral_toe_core.tex`

   This file defines the foundational axioms, spectral functional,
   vacuum structure, emergent mass and time, no-go results, and
   example numerical fixed points.

2. **Validation and scope (Zenodo PDF)**  
   The Zenodo record contains an overview document describing:
   - scope and limits of applicability,
   - falsification and validation tests,
   - Random Matrix Theory (RMT) results,
   - interpretation of negative results.

3. **Reproducible numerical tests (this repository)**  
   All numerical experiments, raw logs, and figure-generation scripts
   used in the validation are included here.

======================================================================
REPOSITORY SCOPE
======================================================================

The purpose of this repository is to:

- provide a transparent reference implementation of a global
  spectral–relational framework,
- document explicit falsification and validation tests,
- allow independent reproduction of Random Matrix Theory analyses,
- clearly delineate the limits of the construction.

This repository is **not**:
- a phenomenological model of hadrons,
- a replacement for QCD or effective field theory,
- a general-purpose simulation framework.

======================================================================
RANDOM MATRIX THEORY (RMT)
======================================================================

The directory `rmt/` contains all Random Matrix Theory analyses used to
test the spectral properties of the theory.

These tests demonstrate that:
- the fundamental vacuum and minimal excitation operators exhibit
  pseudo-integrable / mixed statistics,
- universal GOE behavior is **not** obtained at the fundamental level,
- increasing system size or coupling does not generically restore
  full quantum chaos.

These results falsify interpretations of the theory as a direct
effective description of hadronic spectra, while remaining consistent
with its intended fundamental scope.

All RMT tests are reproducible from raw logs without data selection.

======================================================================
REQUIREMENTS
======================================================================

- Python >= 3.9
- NumPy
- CuPy with CUDA support (for GPU-based RMT tests)
- matplotlib (for figure generation)

======================================================================
RUNNING THE RMT TESTS
======================================================================

From the repository root:
```bash
cd rmt
python scaling_tests.py
python generate_figures.py
```
The generated figures will appear in theory/figures/.

======================================================================
REPRODUCIBILITY
All numerical experiments are deterministic up to explicitly stated
random seeds and ensemble constructions.

Reproducibility guarantees numerical consistency of the implementation,
not physical uniqueness or phenomenological completeness.

Raw outputs are provided to avoid selection bias.

======================================================================
USE OF COMPUTATIONAL ASSISTANCE
Parts of the numerical experimentation, code generation, and manuscript
preparation were performed with the assistance of large language models
(LLMs), under the direct supervision of the author.

All scientific decisions, interpretations, and validations remain the
responsibility of the author.

======================================================================
CITATION
If you use or reference this work, please cite:

Grzegorz Olbryk,
Global Spectral Theory of Everything — Version 2: Scope, Validation,
and Limits,
Zenodo (2026),
DOI: https://doi.org/10.5281/zenodo.18213696

======================================================================
LICENSE
MIT License (see LICENSE file).

======================================================================
CONTACT
Grzegorz Olbryk
g.olbryk@gmail.com
