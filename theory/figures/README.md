# Figures: Random Matrix Theory (RMT) Analysis

This directory contains figures generated directly from raw Random Matrix Theory
(RMT) test outputs located in `rmt/logs/`.

The figures document both positive and negative results and are intended
to illustrate the spectral properties of the Global Spectral TOE without
selection or post-processing.

All figures were generated automatically using the script
`rmt/generate_figures.py`.

---

## Overview of Figures

### 1. `rmt_scaling_r_vs_N.png`

This figure shows the dependence of the r-statistic on the matrix size N
for several fixed values of the coupling parameter alpha.

The absence of a monotonic trend toward the GOE reference value demonstrates
that increasing system size alone does not lead to universal quantum chaos.
Local proximity to GOE values occurs only for specific parameter choices
and is not stable under further scaling.

This behavior supports the classification of the fundamental dynamics
as pseudo-integrable rather than fully ergodic.

---

### 2. `rmt_scaling_r_vs_alpha.png`

This figure displays the r-statistic as a function of the coupling strength
alpha for several fixed matrix sizes N.

The results show that increasing the coupling does not systematically drive
the spectrum toward GOE statistics. In many cases, stronger coupling moves
the system back toward Poisson-like behavior.

This indicates that chaotic phase mixing at the fundamental level is
insufficient to produce universal RMT behavior.

---

### 3. `rmt_all_points.png`

This figure presents all RMT results as individual points without any
selection, averaging, or filtering.

Most data points lie between the Poisson and GOE reference lines, with no
stable accumulation near the GOE value.

This provides a compact visual summary of the mixed or pseudo-integrable
regime characterizing the fundamental spectral dynamics.

---

## Interpretation and Scope

The figures in this directory demonstrate that the fundamental vacuum and
minimal excitation operators of the Global Spectral TOE do not exhibit
universal GOE statistics.

This result falsifies interpretations of the theory as a direct effective
description of hadronic or strongly chaotic many-body spectra.

At the same time, the absence of full ergodicity at the fundamental level
is consistent with the intended scope of the theory as a background-independent
foundational framework. Stronger chaos, if present, is expected to emerge only
at the level of effective many-body excitations, which are not constructed here.

---

## Reproducibility

All figures can be reproduced exactly by running:

cd rmt
python generate_figures.py

powershell
Skopiuj kod

using the raw log files provided in `rmt/logs/`.

No manual data selection, smoothing, or parameter tuning was performed.