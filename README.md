This repository corresponds to Zenodo DOI: 10.5281/zenodo.17961030

# Global Spectral Theory of Everything

This repository contains the reference implementation and numerical experiments
supporting the paper:

**Grzegorz Olbryk**  
*Global Spectral Theory of Everything*  
Zenodo DOI: https://doi.org/10.5281/zenodo.17961030

The framework demonstrates how mass gaps, arrow of time, symmetry breaking
(QCD-like confinement, electroweak sector), fermion chirality, and cosmological
behavior emerge from the spectrum of a single global transport operator, without
assuming local dynamics, action principles, or predefined spacetime structure.

---

## Repository Scope

The code in this repository is intended to:

- reproduce the key numerical results reported in the paper,
- provide a minimal, transparent reference implementation,
- allow independent verification of the spectral mass gap and time emergence,
- serve as a basis for further analytical or numerical investigation.

This is **not** a general-purpose simulation framework, but a research-grade
reference code.

---

## Structure

src/ Core spectral transport and SU(3) connection code
tests/ Minimal reproducibility scripts (mass gap, time arrow, entropy)
experiments/ Batch and ensemble execution helpers


Only scripts directly related to figures, tables, and claims in the paper are
included.

---

## Requirements

- Python >= 3.10
- NumPy

No GPU acceleration or external numerical libraries are required.

---
## Running the core tests

From the repository root:

```bash
python tests/test_mass_gap_qcd.py
python tests/test_arrow_of_time.py
python tests/test_no_local_mass_time.py
```

## Reproducibility

All numerical experiments are deterministic up to ensemble averaging.
Random seeds, lattice sizes, loop definitions, smearing parameters, and
operator constructions are explicitly specified in the scripts.

Typical runtime on a modern CPU:
- single test: seconds to minutes
- ensemble runs: minutes

---

## Citation

If you use this code or results derived from it, please cite:

Grzegorz Olbryk,  
*Global Spectral Theory of Everything*,  
Zenodo (2025), DOI: 10.5281/zenodo.17961030

---

## License

MIT License (see LICENSE file).

---

## Contact

For questions or correspondence related to this work:

Grzegorz Olbryk  
g.olbryk@gmail.com
