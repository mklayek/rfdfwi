<!--
================================================================================
RFDFWI — Full-Waveform Inversion (FWI) of GPR Data  |  By Mrinal

This documentation is part of a Python implementation for Full-Waveform
Inversion (FWI) of Ground Penetrating Radar (GPR) data. FWI is a geophysical
imaging technique used to reconstruct subsurface properties (electromagnetic
permittivity and conductivity) by iteratively comparing modelled and observed
data.

References:
  Lavoué et al. (2014); Layek & Sengupta (2019, 2021, & 2024)
  Köhn, D., De Nil, D. and Rabbel, W. (2017) Tutorial: Introduction to
  frequency domain modelling and FWI of georadar data with GERMAINE.
  DOI: 10.13140/RG.2.2.29354.03523
  ____________________________
  Layek, M. K., & Sengupta, P. (2024). Multi-parameter imaging by finite
  difference frequency domain full waveform inversion of GPR data: A guide
  for sedimentary architecture modeling. Pure and Applied Geophysics, 181,
  2107–2130. https://doi.org/10.1007/s00024-024-03520-1

Copyright © Mrinal Kanti Layek
Original MATLAB written during PhD @ 2018–19:
  Mrinal Kanti Layek, Senior Research Fellow (Geophysics)
  Department of Geology and Geophysics, IIT Kharagpur – 721302, INDIA
  layek.mk@gmail.com | https://www.researchgate.net/profile/Mrinal_Layek

Python code written during Postdoc @ March 2026:
  Dr. Mrinal Kanti Layek — Postdoctoral Researcher | 박사후 연구원
  Geophysics & AI Lab, Department of Energy & Resources Engineering
  Chonnam National University, Gwangju, Republic of Korea [61186]
  지구물리 및 인공지능 연구실, 에너지자원공학과, 전남대학교, 광주광역시 [61186]
  Email: layek.mk@gmail.com
================================================================================
-->
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python port of a MATLAB RFDFWI codebase: 2D frequency-domain finite-difference (FDFD) forward modelling and full-waveform inversion (FWI) for ground-penetrating radar (GPR). The physics is TE-mode (field component is **Ey**, not Ez — label as `Re(Ey)` everywhere).

## Environment and commands

All commands run from `D:\rfdfwi` with the conda environment active:

```bash
conda activate rfdfwimkl
cd D:\rfdfwi
```

**Standard workflow (MATLAB-matched defaults):**
```bash
python examples/run_build_model.py
python examples/run_forward_wavefield.py --stag2 --ncpus 15
python examples/run_forward_bscan.py     --stag2 --ncpus 15
python examples/run_forward_cmp.py       --stag2 --ncpus 15
python examples/run_inversion_example.py --stag2 --ncpus 15
```

**Validation script:** `python examples/validate_python.py`

There are no unit tests or linter configs. The `rfdfwimkl/` directory is the local conda environment — do not modify it.

## Architecture

### Data flow

```
input/*.yaml  ──►  scripts/config_loader.py  ──►  scripts/forward_fdfd.py
                                                          │
                   create_models/build_models.py  ────────┘
                                                          │
                                              scripts/inversion_fwi.py
                                                          │
                                              results/  (npz, png, tiff)
```

### Core modules

**`scripts/forward_fdfd.py`** — the heart of the codebase.
- `build_helmholtz_2d()` assembles the sparse complex 9-point Helmholtz impedance matrix `A` using CFS-PML. Two stencil variants: `stag1` (Hustedt 2004) and `stag2` (Layek & Sengupta 2024, **recommended**).
- `solve_forward()` solves `A @ Ez = b` via `scipy.sparse.linalg.spsolve` (direct LU).
- `run_forward_single_source()` is the main entry point used by all `examples/run_forward_*.py` scripts. It returns `(trace_1d, field_2d, info)`.
- The matrix is assembled once and reused across sources — `run_forward()` handles multi-source parallelism via `ThreadPoolExecutor`.

**`scripts/inversion_fwi.py`** — adjoint-state FWI.
- Gradient via adjoint method: solve `A^H λ = P^T conj(residual)`, then accumulate `Re(conj(λ) * dk²/dparam * u)` over sources.
- Steepest descent with optional linesearch; Tikhonov (Laplacian) regularization; hard box bounds.

**`create_models/build_models.py`** — model generators.
- `mkl_two_cross_model()` is the exact MATLAB replica (use for benchmarking). All other types (`homogeneous`, `layered`, `two_cross`, `file`) are for experiments.
- `build_4sided_acquisition()` generates the MATLAB-style 82-source / 162-receiver boundary geometry.

**`examples/run_forward_cmp.py`** — most complex example.
- Runs `nf × n_offsets` FDFD solves (e.g. 50 × 25 = 1250). Use `--ncpus` to parallelize.
- FD→TD conversion uses **Hermitian IFFT** (MATLAB GPRFM style): construct two-sided spectrum `[conj(E[nf-1:0:-1]), E[0:nf]]`, optionally zero-pad, then `ifftshift` + `ifft`. This is the only correct approach — one-sided IFFT gives Fourier artifacts (horizontal stripes).

### Configuration

YAML files in `input/` are the primary configuration. Key config helpers in `scripts/config_loader.py`:
- `get_freq_sweep()` — returns `{fc_low, fc_high, nf, df, clip, clip1, tmax_td}`
- `get_acquisition_sources/receivers()` — supports `mode: 4sided` (auto-generates MATLAB geometry) or explicit lists

### Output conventions (established and must be consistent)

All `examples/run_forward_*.py` scripts follow these conventions — do not change them:
- **Colormap**: `seismic` (not `seismic_r`) — this is the "flipped" version the user wants
- **Field label**: `Re(Ey)` everywhere (TE mode — MATLAB convention)
- **DPI**: 600 for all saved figures
- **Wavefield**: plots the full 200×200 extended grid (includes PML), at the **highest** frequency (`results[-1]` = 200 MHz) for thin rings
- **B-scan / CMP**: per-trace AGC normalisation (divide each column by its max), then 95th-percentile colour clip, `interpolation="none"`

## Critical implementation details

### MATLAB index mapping
MATLAB uses 1-indexed interior cells; Python uses 0-indexed over the full extended grid:
```python
python_ix = npml + matlab_i - 1    # e.g. MATLAB i=11 → Python ix=20
```

### CFS-PML stretch factor
```python
sx = kappax + sigma_x * 1j / (alpha_x + omega)   # eps01 = 1, NOT eps0
```
The MATLAB code normalises by `eps01=1`, not the vacuum permittivity `EPS0`. This is a critical difference from naive implementations.

### Source amplitude (matches MATLAB `RHS_TE1.m`)
```python
src_amp = -(omega * MU0 * 1j) / dh**2
```

### CMP Hermitian IFFT time axis
```
ns = (nf + pad) * 2 - 1
dt = 1 / (ns * df)        where df = (fc_high - fc_low) / (nf - 1)
Tmax ≈ 1 / df  (independent of zero-padding)
```

## Key parameters (MATLAB-matched defaults)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Grid | 200×200 cells | 180 interior + 10 PML per side |
| `dh` | 0.05 m | Grid spacing (dx = dz) |
| `a0_cfs` | 9.0e8 | CFS-PML sigma_max |
| `freq_hz` | 50 MHz | Single-frequency forward |
| `fc_low / fc_high` | 50 / 200 MHz | Frequency sweep |
| `nf` | 50 | → df ≈ 3.061 MHz |
| Background model | εᵣ=4.0, σ=3e-3 S/m | `mkl_two_cross` default |
| Source/receiver depth | `iz=20` | = npml + 10 cells from PML |

## Reference documents

- `docs/MATLAB_to_Python_Mapping.md` — definitive MATLAB↔Python parameter table, index formulas, cross positions
- `CLI_REFERENCE.md` — all CLI flags for every script
- `docs/MANUAL.md` — complete YAML schema
