This repository corresponds to Zenodo DOI:
https://doi.org/10.5281/zenodo.17961030

======================================================================
STATUS NOTICE
======================================================================

This repository contains **Version 2** of the Global Spectral Theory of
Everything (TOE).

An earlier exploratory version (Zenodo record v1) has been **superseded**
following explicit validation and falsification tests, most notably
Random Matrix Theory (RMT) analysis. The present repository documents
the corrected scope, limits, and core mathematical structure of the
theory in a transparent and reproducible manner.

This work is **not presented as a complete effective theory of hadrons
or strongly chaotic many-body systems**. Its claims are explicitly
limited to the fundamental, background-independent level.

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
DOI: https://doi.org/10.5281/zenodo.17961030

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
The generated figures will appear in `theory/figures/`.

======================================================================
REPRODUCIBILITY
======================================================================

All numerical experiments are deterministic up to explicitly stated
random seeds and ensemble constructions.

Reproducibility guarantees numerical consistency of the implementation,
**not** physical uniqueness or phenomenological completeness.

Raw outputs are provided to avoid selection bias.

======================================================================
USE OF COMPUTATIONAL ASSISTANCE
======================================================================

Parts of the numerical experimentation, code generation, and manuscript
preparation were performed with the assistance of large language models
(LLMs), under the direct supervision of the author.

All scientific decisions, interpretations, and validations remain the
responsibility of the author.

======================================================================
CITATION
======================================================================

If you use or reference this work, please cite:

Grzegorz Olbryk,  
*Global Spectral Theory of Everything — Version 2: Scope, Validation,
and Limits*,  
Zenodo (2026),  
DOI: https://doi.org/10.5281/zenodo.17961030

======================================================================
LICENSE
======================================================================

MIT License (see LICENSE file).

======================================================================
CONTACT
======================================================================

Grzegorz Olbryk  
g.olbryk@gmail.com