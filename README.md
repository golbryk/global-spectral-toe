This repository corresponds to Zenodo DOI: https://doi.org/10.5281/zenodo.17961030

⚠️ STATUS NOTICE

This repository documents a **superseded exploratory global spectral framework**.
It is retained for transparency, reproducibility, and methodological reference
and should **not** be interpreted as a final or complete Theory of Everything.

# Global Spectral Framework (Superseded)

This repository contains reference code and numerical experiments associated with:

**Grzegorz Olbryk**  
*Global Spectral Theory of Everything*  
Zenodo DOI: https://doi.org/10.5281/zenodo.17961030

Subsequent analysis indicates that global spectral constructions generically
suffer from random-matrix universality and are therefore unsuitable as a final
fundamental theory. This repository represents an early-stage investigation of
such approaches.

---

## Repository Scope

The code is intended to:

- reproduce numerical experiments reported in the associated Zenodo record,
- provide a transparent reference implementation of a global operator–based model,
- allow independent inspection of spectral stability and irreversibility indicators,
- serve as a controlled testbed for evaluating global spectral constructions.

This is **not** a general-purpose simulation framework and **not** an endorsed
physical theory.

---

## Structure

src/ Core global spectral transport code
tests/ Minimal reproducibility scripts (spectral gaps, irreversibility,
entropy-related observables, observer stability)
experiments/ Batch and ensemble execution helpers

yaml
Skopiuj kod

Only scripts directly related to claims and figures in the Zenodo record are included.

---

## Requirements

- Python >= 3.10
- NumPy

No GPU acceleration or external numerical libraries are required.

---

## Running the Reference Tests

From the repository root:

```bash
python tests/test_mass_gap_qcd.py
python tests/test_arrow_of_time.py
python tests/test_no_local_mass_time.py
These scripts reproduce representative qualitative behaviors discussed in the
associated Zenodo document within the global spectral framework.

Reproducibility
All experiments are deterministic up to ensemble averaging.
Random seeds, system sizes, and operator constructions are explicitly specified.

Reproducibility guarantees numerical consistency of the implementation,
not physical validity or uniqueness.

Citation
If you use this code, please cite:

Grzegorz Olbryk,
Global Spectral Theory of Everything,
Zenodo (2025), DOI: https://doi.org/10.5281/zenodo.17961030

Note: This citation refers to a superseded global spectral formulation.

License
MIT License (see LICENSE file).

Contact
Grzegorz Olbryk
g.olbryk@gmail.com
