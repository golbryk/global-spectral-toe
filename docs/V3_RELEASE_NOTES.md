# Version 3 — Release Notes

**Global Spectral Theory of Everything**
Author: Grzegorz Olbryk · g.olbryk@gmail.com

These notes describe the **Zenodo Version 3** public record and pin the exact
state of the source repositories at release time. They are a post-release
reproducibility supplement; they do **not** change the Zenodo PDF.

---

## Release identity

| Field | Value |
|-------|-------|
| Title | *Global Spectral Theory of Everything — Version 3: Scope, Structural Status, and Operational Falsifiers* |
| Zenodo DOI (v3) | [10.5281/zenodo.20374769](https://doi.org/10.5281/zenodo.20374769) |
| Publication date | 2026-05-25 |
| Public artifact | single-file synthesis/scope PDF, `global_spectral_toe_v3.pdf` (hosted on Zenodo) |
| `global-spectral-toe` release commit | `ef5c93a` |
| `hsmi-yang-mills` companion commit | `0fea9c5` |
| Previous version (v2) | [10.5281/zenodo.18213696](https://doi.org/10.5281/zenodo.18213696) |
| Superseded version (v1) | [10.5281/zenodo.17961030](https://doi.org/10.5281/zenodo.17961030) |

> The public Version 3 PDF (`global_spectral_toe_v3.pdf`) lives on the Zenodo
> record, not in this repository. The repository provides the source code,
> raw logs, and the status/reproducibility documentation that accompany it.

---

## What Version 3 is

Version 3 is a **synthesis / scope / status document**. It continues
Version 2 with the corrected, *layered* scientific status. It is **not** a
full research archive, **not** a complete derivation, and **not** a record of
every internal exploration.

- It states scope, structural status, and operational falsifiers.
- It is the authoritative public wording; where any older file disagrees,
  Version 3 and [`docs/STATUS_AND_LIMITS.md`](STATUS_AND_LIMITS.md) take
  precedence.

## What Version 3 is **not** (scope boundary)

- It is **not** a claim of a complete or "100%" Theory of Everything.
- It does **not** claim the neutrino mass ordering is solved, and it does
  **not** claim the "A4" ordering question is solved at the local level — that
  status is explicitly **layered and open** (see below).
- It is **not** a phenomenological replacement for the Standard Model or QCD.

## Obsolete internal draft

The internal LaTeX draft `theory/toe_v3_stokes_foundation.tex` (and its built
PDF `theory/toe_v3_stokes_foundation.pdf`) is an **obsolete, superseded
internal draft**. It is **not** the Zenodo Version 3 release document and must
not be cited as the final Version 3. In particular, its flat statement that
the framework "predicts the normal neutrino hierarchy and an inverted result
would falsify the theory" is superseded by the layered status in the
Version 3 synthesis and in `docs/STATUS_AND_LIMITS.md`. See
`theory/DEPRECATED_toe_v3_stokes_foundation.md`.

## Neutrino mass ordering — layered status (summary)

None of these layers is a solution; see `docs/STATUS_AND_LIMITS.md` §4 for the
full statement and falsifier logic.

1. **Local construction — open.** The relevant Majorana scale is a free
   parameter; the ordering is **not derived as a local theorem** (empirical
   boundary).
2. **Minimal Majorana branch — model-selection, not forced.** The minimal /
   symmetric choice favors Normal Ordering.
3. **Global / swampland argument — conjectural.** A global-spectral
   (quantum-gravity) consistency argument convergently favors Normal Ordering,
   resting on unproven conjectures.

An inverted-ordering measurement would falsify the predictive (minimal
Majorana) branch and the global-spectral argument, but would **not**
automatically falsify the entire local framework.

## Companion repository

The rigorous lattice-gauge mathematics (Yang–Mills mass-gap candidate proof,
Fisher zeros, Stokes concentration, and the Stokes–RG correspondence engine)
lives in the **independent** companion repository
[`hsmi-yang-mills`](https://github.com/golbryk/hsmi-yang-mills), pinned at
commit `0fea9c5`. None of the foundational claims here are required by, or
assumed in, the rigorous results there.

---

*Reproducibility map: [`V3_REPRODUCIBILITY.md`](V3_REPRODUCIBILITY.md).
RMT audit scope: [`RMT_NUMERICAL_AUDIT_PLAN.md`](RMT_NUMERICAL_AUDIT_PLAN.md).
Machine-readable manifest: [`../releases/v3_manifest.json`](../releases/v3_manifest.json).*
