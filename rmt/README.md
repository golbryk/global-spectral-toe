# Random Matrix Theory (RMT) Tests

This directory contains all Random Matrix Theory tests used to assess
the spectral properties of the Global Spectral TOE.

The purpose of these tests is falsification, not confirmation.

## What is tested
- Fundamental vacuum spectra
- Minimal excitation operators
- Scaling with matrix size (N)
- Scaling with coupling strength (alpha)
- Comparison with GOE and Poisson ensembles

## What is NOT claimed
- No direct reproduction of hadronic spectra
- No universal GOE behavior at the fundamental level

## How to run
Python >= 3.9 and CuPy with CUDA support are required.

Example:
    python scaling_tests.py
