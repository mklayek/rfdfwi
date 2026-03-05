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
# RFDFWI — Full Reference Manual

> **Python port of the MATLAB RFDFWI toolbox.**
> 2-D Frequency-Domain Finite-Difference (FDFD) forward modelling and
> Full-Waveform Inversion (FWI) for Ground-Penetrating Radar (GPR).

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Installation & Environment](#2-installation--environment)
3. [Quick Start](#3-quick-start)
4. [Project Structure](#4-project-structure)
5. [Configuration Reference (YAML)](#5-configuration-reference-yaml)
6. [Model Types](#6-model-types)
7. [Example Scripts Reference](#7-example-scripts-reference)
8. [Core Algorithms](#8-core-algorithms)
9. [Output Files Reference](#9-output-files-reference)
10. [MATLAB Correspondence](#10-matlab-correspondence)
11. [Parallelism & Performance](#11-parallelism--performance)
12. [Troubleshooting](#12-troubleshooting)
13. [API Quick Reference](#13-api-quick-reference)

---

## 1. Overview & Architecture

### What the project does

RFDFWI solves the 2-D TE-mode Helmholtz equation in the frequency domain using a
9-point staggered-grid finite-difference stencil with Complex-Frequency-Shifted
Perfectly Matched Layers (CFS-PML) absorbing boundaries. It produces:

- Multi-frequency forward wavefields (Ez component)
- Zero-offset B-scan radargrams
- Common Mid-Point (CMP) gathers
- Shot gathers (single-source, all receivers)
- Full-Waveform Inversion recovery of relative permittivity (εr) and electrical
  conductivity (σ)

The code is a direct Python translation of the MATLAB RFDFWI toolbox and is designed
to reproduce its numerical output exactly.

### Key capabilities

- Two 9-point CFS-PML stencil variants: **stag1** (Hustedt et al. 2004) and
  **stag2** (Layek & Sengupta 2024)
- Sparse direct LU solver via `scipy.sparse.linalg.spsolve`
- Hermitian IFFT with Blackman-Harris window for time-domain conversion
- AGC2 automatic gain control for display normalisation
- Adjoint-state gradient computation (εr and σ simultaneously)
- Pseudo-Hessian diagonal pre-conditioner
- Tikhonov (Laplacian) regularisation on σ
- Armijo backtracking line search
- 4-sided borehole-style acquisition geometry (82 sources, 162 receivers)
- Parallel multi-source/multi-frequency solving via `ThreadPoolExecutor`
- YAML-driven configuration with CLI overrides
- Optional GPU flag (`--use-gpu`) reserved for future CuPy integration

### Data flow diagram (ASCII)

```
  YAML config
       |
       v
  build_models.py ─────────────────────────────────────────────
       |                                                        |
       v                                                        v
  model_epsr.npy                                       model_sigma.npy
  model_sigma.npy                                   (inputmodel/ directory)
       |
       v
  forward_fdfd.py ──── CFS-PML coefficients ──── stag1 / stag2 stencil
  build_helmholtz_2d()                                    |
       |                                                  v
       |                                   Sparse Helmholtz matrix A
       |                                   scipy.sparse.linalg.spsolve
       |                                          A u = b
       v
  Ez wavefield (complex, per frequency)
       |
       +──── wavefield plot ───────────────────> results/forward/wavefield/
       |
       +──── B-scan stack ─────────────────────> results/forward/bscan/
       |
       +──── CMP IFFT ─────────────────────────> results/forward/cmp/
       |
       +──── Shot-gather IFFT ─────────────────> results/forward/shotgather/
       |
       v
  inversion_fwi.py
  compute_gradient()
  compute_hessian()
  tikhonov_regularise()
  armijo_line_search()
       |
       v
  results/inversion/{models, gradient, hessian,
                     search_direction, tikhonov,
                     misfit, logs}/
```

### Module dependency map

```
examples/
  run_build_model.py          -> create_models/build_models.py
  run_forward_wavefield.py    -> scripts/forward_fdfd.py
                                 scripts/config_loader.py
                                 scripts/plot_utils.py
                                 scripts/_cli.py
  run_forward_bscan.py        -> scripts/forward_fdfd.py
                                 scripts/plot_bscan.py
                                 scripts/_cli.py
  run_forward_cmp.py          -> scripts/forward_fdfd.py
                                 scripts/plot_cmp.py
                                 scripts/_cli.py
  run_forward_shotgather.py   -> scripts/forward_fdfd.py
                                 scripts/plot_cmp.py     (reused)
                                 scripts/_cli.py
  run_forward_shotgather_center.py  -> (same as shotgather)
  run_inversion_example.py    -> scripts/inversion_fwi.py
                                 scripts/forward_fdfd.py
                                 scripts/_cli.py
  validate_python.py          -> scripts/forward_fdfd.py
                                 scripts/inversion_fwi.py
```

---

## 2. Installation & Environment

### Requirements

| Package    | Minimum | Upper bound | Notes                     |
|------------|---------|-------------|---------------------------|
| Python     | 3.11    | 3.12        | 3.11 recommended          |
| NumPy      | 1.20    | <3.0        |                           |
| SciPy      | 1.7     | <2.0        | sparse LU solver          |
| Matplotlib | 3.4     | <4.0        | seismic colormap, TIFF    |
| PyYAML     | 5.4     | <7.0        | YAML config loading       |

### Conda setup (rfdfwimkl environment)

```bash
# 1 — clone the repository
git clone <repository-url>
cd rfdfwi

# 2 — create the conda environment
conda create -n rfdfwimkl python=3.11 -y

# 3 — activate
conda activate rfdfwimkl

# 4 — install Python dependencies
pip install -r requirements.txt

# 5 — verify
python -c "import numpy, scipy, matplotlib, yaml; print('All dependencies OK')"

# 6 — validate the solver
python examples/validate_python.py
```

### Optional: GPU acceleration

CuPy (CUDA) is listed as an optional dependency. The `--use-gpu` flag is accepted
by all scripts, but the sparse direct solver (`scipy.sparse.linalg.spsolve`) does
not currently use the GPU. Install CuPy for future compatibility:

```bash
# CUDA 11.x
conda install -c conda-forge cupy cuda-version=11.8

# CUDA 12.x
conda install -c conda-forge cupy cuda-version=12.0
```

When CuPy is absent the code falls back to the CPU solver automatically and no
functionality is lost.

---

## 3. Quick Start

Five commands reproduce the full MATLAB RFDFWI workflow:

```bash
conda activate rfdfwimkl

# Step 1 — Build the mkl_two_cross model (exact MATLAB replica, 200x200 grid)
python examples/run_build_model.py

# Step 2 — Multi-frequency forward wavefield (nf=50, 50-200 MHz)
python examples/run_forward_wavefield.py --stag2 --ncpus 15

# Step 3 — Zero-offset B-scan radargram
python examples/run_forward_bscan.py --stag2 --ncpus 15

# Step 4 — CMP gather
python examples/run_forward_cmp.py --stag2 --ncpus 10

# Step 5 — Full-Waveform Inversion (4-sided acquisition, 82 sources)
python examples/run_inversion_example.py --stag2 --ncpus 15
```

All outputs are written under `results/`.

---

## 4. Project Structure

### Full directory tree

```
rfdfwi/
├── create_models/
│   └── build_models.py            Model constructors (homogeneous, layered,
│                                  two_cross, mkl_two_cross, file)
├── scripts/
│   ├── forward_fdfd.py            2-D FDFD Helmholtz solver (PML, stencils,
│                                  source injection, multi-source/freq driver)
│   ├── inversion_fwi.py           FWI engine: gradient, Hessian, Tikhonov,
│                                  Armijo line search, iteration loop
│   ├── config_loader.py           YAML load + validation helpers
│   ├── plot_bscan.py              B-scan figure generation (AGC2, wiggle)
│   ├── plot_cmp.py                CMP / shot-gather figure generation
│   ├── plot_utils.py              Shared utilities: seismic colormap, PML
│                                  boundary overlay, draw_pml_boundary()
│   └── _cli.py                    add_common_args() shared argparse factory
├── examples/
│   ├── run_build_model.py         Step 1 : build and save 2-D GPR model
│   ├── run_forward_wavefield.py   Step 2a: single-source wavefield (PNG+TIFF)
│   ├── run_forward_bscan.py       Step 2b: zero-offset B-scan radargram
│   ├── run_forward_cmp.py         Step 2c: CMP gather (time-domain)
│   ├── run_forward_shotgather.py  Step 2d: single-source shot gather
│   ├── run_forward_shotgather_center.py  Step 2e: shot gather, centred source
│   ├── run_inversion_example.py   Step 3 : Full-Waveform Inversion
│   └── validate_python.py         Automated solver correctness checks
├── input/
│   ├── input_forward.yaml         Forward modelling configuration
│   ├── input_forward_with_model.yaml  Forward config loading saved model
│   └── input_inversion.yaml       FWI inversion configuration
├── inputmodel/                    Auto-created; holds generated model arrays
│   ├── model_epsr.npy             Relative permittivity grid
│   ├── model_sigma.npy            Conductivity grid [S/m]
│   └── model_eps_sig.png          Preview figure
├── results/
│   ├── forward/
│   │   ├── wavefield/             wavefield_real.png, wavefield_real.tiff
│   │   ├── bscan/                 bscan.npz, bscan.png
│   │   ├── cmp/                   cmp_<tags>.npz, cmp_<tags>.png,
│   │   │                          cmp_wiggle_<tags>.png
│   │   └── shotgather/            sg_<tags>.npz, sg_<tags>.png,
│   │                              sg_wiggle_<tags>.png
│   └── inversion/
│       ├── obs/                   d_obs.npz (synthetic observed data)
│       ├── models/                iter_<N>_epsr.png, iter_<N>_sigma.png,
│       │                          final_result.npz
│       ├── gradient/              iter_<N>_grad_epsr.png, grad_sigma.png
│       ├── hessian/               iter_<N>_hess_epsr.png, hess_sigma.png
│       ├── search_direction/      iter_<N>_dir_epsr.png, dir_sigma.png
│       ├── tikhonov/              iter_<N>_tikh_epsr.png, tikh_sigma.png
│       ├── misfit/                misfit_curve.png
│       └── logs/                  run_log.txt
├── docs/
│   ├── MANUAL.md                  This file — full reference manual
│   └── MATLAB_to_Python_Mapping.md  Function-level MATLAB->Python table
├── requirements.txt
├── README.md
├── INSTALLATION.md
├── CLI_REFERENCE.md
└── CLAUDE.md                      Developer notes
```

### Key input files

| File | Description |
|------|-------------|
| `input/input_forward.yaml` | Main forward config: grid, PML, frequencies, model, acquisition |
| `input/input_forward_with_model.yaml` | Same but `model.type: file` to load saved arrays |
| `input/input_inversion.yaml` | Inversion config: all `inversion.*` keys + same forward sections |
| `inputmodel/model_epsr.npy` | 200x200 float64 relative permittivity array |
| `inputmodel/model_sigma.npy` | 200x200 float64 conductivity array [S/m] |

### Key output files

| File | Description |
|------|-------------|
| `results/forward/wavefield/wavefield_real.png` | Real(Ez) at FC_low, seismic colormap |
| `results/forward/bscan/bscan.npz` | B-scan traces array |
| `results/inversion/obs/d_obs.npz` | Observed data [n_src, n_freq, n_rec] complex |
| `results/inversion/models/final_result.npz` | Final, initial, and true model arrays |
| `results/inversion/logs/run_log.txt` | Full run metadata and per-iteration misfit log |

---

## 5. Configuration Reference (YAML)

### 5.1 Forward configuration (`input/input_forward.yaml`)

This is the primary configuration file used by all forward-modelling scripts.

```yaml
# -- Domain ------------------------------------------------------------------
domain:
  nx: 200          # Total grid points in x (including PML both sides)
  nz: 200          # Total grid points in z (including PML both sides)
  dh: 0.05         # Cell size [m] -- same in x and z (isotropic grid)
                   # Physical domain: (nx - 2*npx)*dh = 180*0.05 = 9.0 m

# -- Perfectly Matched Layer -------------------------------------------------
pml:
  npx: 10          # PML cells each side in x  (total x PML = 20 cells)
  npz: 10          # PML cells each side in z  (total z PML = 20 cells)
  a0_cfs: 9.0e8    # CFS-PML sigma_max parameter (Chen et al. 2013)
                   # MATLAB: sig_max = 9e8

# -- Source ------------------------------------------------------------------
source:
  ix: 99           # Source x-index (0-based); MATLAB ix=100 (1-based) -> 99
  iz: 20           # Source z-index (0-based); top acquisition row

# -- Receivers ---------------------------------------------------------------
receivers:
  mode: line       # 'line' = horizontal surface array
  iz: 20           # Receiver depth index (same row as source for surface acq.)
  ix_start: 20     # First receiver x-index (= npx, inside PML boundary)
  ix_end: 179      # Last receiver x-index  (= nx - npx - 1)

# -- Model -------------------------------------------------------------------
model:
  type: mkl_two_cross   # See Section 6 for all model types
  # For type: homogeneous
  # epsr: 4.0
  # sigma: 3e-3
  # For type: file -- loads inputmodel/model_epsr.npy + model_sigma.npy

# -- Frequency sweep ---------------------------------------------------------
freq_sweep:
  fc_low:  50e6    # Start frequency [Hz]  -- MATLAB: FC_low  = 50 MHz
  fc_high: 200e6   # End frequency   [Hz]  -- MATLAB: FC_high = 200 MHz
  nf:      50      # Number of frequencies; df = (fc_high - fc_low)/(nf-1)
                   # = 150e6/49 approx 3.061 MHz
  clip:    2.5e-3  # Blackman-Harris amplitude clip  (MATLAB: clip)
  clip1:   1.0e-2  # Secondary amplitude clip        (MATLAB: clip1)
  tmax_td: 150e-9  # Max time-domain window          (MATLAB: TmaxTD = 150 ns)

# -- Acquisition (used by FWI scripts) ---------------------------------------
acquisition:
  mode: 4sided          # 4-sided borehole-style geometry
  npml: 10              # PML thickness (must match pml.npx/npz)
  nrec_per_side: 40     # Receivers per side -> 162 total (41+41+40+40)
  nsrc_per_side: 20     # Sources per side   ->  82 total (21+21+20+20)
```

**Key derived quantities:**

| Quantity | Value | Formula |
|----------|-------|---------|
| Interior cells x | 180 | `nx - 2*npx = 200 - 20` |
| Interior cells z | 180 | `nz - 2*npz = 200 - 20` |
| Physical domain | 9 m x 9 m | `180 x 0.05 m` |
| Frequency step df | ~3.061 MHz | `(200e6 - 50e6) / (50-1)` |
| Min acquisition index | 20 | `npx` (0-based) |
| Max acquisition index | 179 | `nx - npx - 1` (0-based) |

### 5.2 Model-with-file configuration (`input/input_forward_with_model.yaml`)

Identical to `input_forward.yaml` except:

```yaml
model:
  type: file       # Load pre-built arrays from inputmodel/
                   # model_epsr.npy and model_sigma.npy must exist
```

Use this after running `run_build_model.py` to avoid rebuilding the model
on every forward run.

### 5.3 Inversion configuration (`input/input_inversion.yaml`)

The inversion config extends the forward config with an `inversion:` block.

```yaml
# -- Forward section (same keys as input_forward.yaml) ----------------------
domain:
  nx: 200
  nz: 200
  dh: 0.05
pml:
  npx: 10
  npz: 10
  a0_cfs: 9.0e8
model:
  type: mkl_two_cross    # True model used to generate synthetic d_obs
freq_sweep:
  fc_low:  50e6
  fc_high: 200e6
  nf:      50
  clip:    2.5e-3
  clip1:   1.0e-2
  tmax_td: 150e-9
acquisition:
  mode: 4sided
  npml: 10
  nrec_per_side: 40
  nsrc_per_side: 20

# -- Inversion block ---------------------------------------------------------
inversion:
  max_iter: 20          # Maximum FWI iterations
  conv_tol: 5.0e-5      # Stop when L2/L2[0] <= conv_tol (MATLAB default)

  # Initial model
  initial_model:
    type: smooth         # 'smooth' = Gaussian-blur of true model (MATLAB default)
    smooth_px: 6.0       # Gaussian sigma [pixels] (MATLAB default)
    # Alternative: type: homogeneous
    # epsr:  4.0
    # sigma: 3e-3

  # Parameter bounds (applied after each update)
  bounds:
    epsr_min:  1.0       # Minimum relative permittivity
    epsr_max:  25.0      # Maximum relative permittivity
    sigma_min: 0.0       # Minimum conductivity [S/m]
    sigma_max: 1.0       # Maximum conductivity [S/m]

  # Tikhonov regularisation
  regularization:
    type: tikhonov       # Laplacian smoothing on sigma
    LAMBDA_1: 2.0e-4     # Regularisation weight on sigma  (MATLAB default)
    LAMBDA_2: 0.0        # Regularisation weight on epsr   (MATLAB: 0)

  # Armijo backtracking line search
  line_search:
    STEPMAX:  3          # Max backtracking steps
    SCALEFAC: 2          # Step reduction factor per backtrack
    C1:       1.0e-4     # Armijo sufficient-decrease constant
    step_init: auto      # 'auto' = L2 / ||gradient||^2 (MATLAB default)
                         # Or set a float to fix the initial step

  # Output control
  output:
    dir: results/inversion
    save_every_iter: true    # Save PNG of model at each iteration
    save_gradient:   true    # Save gradient images
    save_hessian:    true    # Save Hessian diagonal images
    save_search_dir: true    # Save search-direction images
    save_tikhonov:   true    # Save regularisation-term images
```

**All inversion keys with types and defaults:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `inversion.max_iter` | int | 20 | Maximum iterations |
| `inversion.conv_tol` | float | 5e-5 | Convergence: stop when L2/L2[0] <= tol |
| `inversion.regularization.LAMBDA_1` | float | 2e-4 | Tikhonov weight on sigma |
| `inversion.regularization.LAMBDA_2` | float | 0.0 | Tikhonov weight on epsr |
| `inversion.line_search.STEPMAX` | int | 3 | Max Armijo backtrack steps |
| `inversion.line_search.SCALEFAC` | float | 2.0 | Step reduction per backtrack |
| `inversion.line_search.C1` | float | 1e-4 | Armijo sufficient-decrease constant |
| `inversion.bounds.epsr_min` | float | 1.0 | Lower bound on epsr |
| `inversion.bounds.epsr_max` | float | 25.0 | Upper bound on epsr |
| `inversion.bounds.sigma_min` | float | 0.0 | Lower bound on sigma [S/m] |
| `inversion.bounds.sigma_max` | float | 1.0 | Upper bound on sigma [S/m] |

---

## 6. Model Types

The `model.type` key in the YAML selects the model constructor. All models
produce two 200x200 float64 arrays: `model_epsr` and `model_sigma`.

### homogeneous

Uniform background with constant epsr and sigma everywhere.

```yaml
model:
  type: homogeneous
  epsr:  4.0     # Relative permittivity (dry sand background)
  sigma: 3.0e-3  # Conductivity [S/m]
```

### two_cross

Two rectangular anomalies (crosses) with parametric geometry:

```yaml
model:
  type: two_cross
  background_epsr:  4.0
  background_sigma: 3.0e-3
  cross1:
    center_x: 3.0    # [m] from left physical edge
    center_z: 3.0    # [m] from top physical edge
    half_len:  1.0   # [m] half-length of each arm
    epsr:  1.0       # Dry sand (air-like low permittivity)
    sigma: 0.0
  cross2:
    center_x: 6.0
    center_z: 6.0
    half_len:  1.0
    epsr:  8.0       # Dry clay (higher permittivity)
    sigma: 0.0
```

### mkl_two_cross (MATLAB replica)

**Exact replica of `create_models_mkl.m`.** Hard-coded geometry matching the
MATLAB coordinate system. Use this for MATLAB comparison runs.

```yaml
model:
  type: mkl_two_cross
```

Physical geometry (dh=0.05 m, 200x200 grid, npml=10):

| Feature | Grid indices (0-based) | Physical position [m] |
|---------|------------------------|------------------------|
| Background | entire grid | epsr=4, sigma=3e-3 S/m |
| Cross 1 (dry sand) | ix: 60-79, iz: 60-139 + ix: 40-99, iz: 80-119 | epsr=1, sigma=0 |
| Cross 2 (dry clay) | ix: 120-139, iz: 60-139 + ix: 100-159, iz: 80-119 | epsr=8, sigma=0 |

### layered

Horizontal layer stack:

```yaml
model:
  type: layered
  layers:
    - { z_bottom: 3.0, epsr: 4.0, sigma: 1.0e-3 }   # top layer
    - { z_bottom: 6.0, epsr: 9.0, sigma: 5.0e-3 }   # middle layer
    - { z_bottom: 9.0, epsr: 6.0, sigma: 2.0e-3 }   # bottom layer
```

`z_bottom` is measured from the top of the physical domain (PML excluded).

### file

Load pre-built NumPy arrays:

```yaml
model:
  type: file
  epsr_file:  inputmodel/model_epsr.npy
  sigma_file: inputmodel/model_sigma.npy
```

Arrays must be float64 with shape `(nz, nx)` matching the domain config.

### 4-sided acquisition geometry (`build_4sided_acquisition`)

The 4-sided acquisition places sources and receivers around all four sides of the
interior domain:

```
 ix=20                           ix=179
  |------------------------------|
  |  Top receivers + sources    |  iz=20
  |                             |
  |  Left sources/recvrs        |  Right sources/recvrs
  |                             |
  |  Bottom receivers + sources |  iz=179
  |------------------------------|
```

With `nsrc_per_side=20`, `nrec_per_side=40`, `npml=10`:

- **82 sources** total: 21 top + 21 bottom + 20 left + 20 right
- **162 receivers** total: 41 top + 41 bottom + 40 left + 40 right
- All indices in range [20, 179] (0-based, exclusive of PML)

---

## 7. Example Scripts Reference

All scripts are run from the project root with the `rfdfwimkl` environment active.

### 7.1 `run_build_model.py`

**Purpose:** Build and save the 2-D GPR model (epsr + sigma) from a YAML config.
Produces NumPy `.npy` arrays and a MATLAB-style preview PNG.

**CLI arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config FILE` | str | `input/input_forward.yaml` | YAML configuration file |

**Usage:**

```bash
# Default -- builds mkl_two_cross model
python examples/run_build_model.py

# Load a different config
python examples/run_build_model.py --config input/input_forward_with_model.yaml
```

**Output files:**

| File | Description |
|------|-------------|
| `inputmodel/model_epsr.npy` | Relative permittivity grid (float64, shape nz x nx) |
| `inputmodel/model_sigma.npy` | Conductivity grid [S/m] (float64, shape nz x nx) |
| `inputmodel/model_eps_sig.png` | Side-by-side preview of epsr and sigma |

---

### 7.2 `run_forward_wavefield.py`

**Purpose:** Solve the FDFD problem for a single source across `nf` frequencies,
apply Hermitian IFFT + Blackman-Harris windowing, and save the real part of Ez
at `FC_low` as a MATLAB-style seismic-colormap figure (PNG + TIFF).

**CLI arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| *(all common args)* | -- | -- | See CLI_REFERENCE.md common table |
| `--source-ix IX` | int | 99 (from config) | Source x grid index (0-based) |
| `--source-iz IZ` | int | 20 (from config) | Source z grid index (0-based) |
| `--freq-min HZ` | float | 50e6 | FC_low -- lowest frequency [Hz] |
| `--freq-max HZ` | float | 200e6 | FC_high -- highest frequency [Hz] |
| `--nf N` | int | 50 | Number of frequencies in linspace sweep |
| `--freq-step HZ` | float | None | Explicit frequency step -- overrides `--nf` |
| `--clip V` | float | 2.5e-3 | Blackman-Harris clip value |
| `--clip1 V` | float | 1.0e-2 | Secondary clip value |
| `--caxis V` | float | 10.0 | Symmetric colour limits +-V [V/m]; 0 = auto |
| `--no-tiff` | flag | off | Save PNG only, skip TIFF |

**Usage:**

```bash
# MATLAB defaults (nf=50, 50-200 MHz)
python examples/run_forward_wavefield.py --stag2 --ncpus 15

# Explicit 1 MHz step (151 frequencies)
python examples/run_forward_wavefield.py --stag2 --ncpus 15 --freq-step 1e6

# Custom source, no TIFF
python examples/run_forward_wavefield.py --stag2 --source-ix 99 --source-iz 20 --no-tiff
```

**Output files:**

| File | Description |
|------|-------------|
| `results/forward/wavefield/wavefield_real.png` | Seismic colormap, 300 DPI |
| `results/forward/wavefield/wavefield_real.tiff` | Same, TIFF format (MATLAB comparison) |

Figure style: seismic colormap, symmetric +-caxis clim, white dashed PML boundary,
white star at source position, axes in metres.

**Note on field labelling:** The plotted quantity is the Ez component (vertical
electric field in TE mode). Some plot titles display "Ey" — this is a historical
naming convention from the original 3-D-to-2-D reduction; both labels refer to the
same physical quantity.

---

### 7.3 `run_forward_bscan.py`

**Purpose:** Zero-offset GPR B-scan (radargram). Steps a single source along the
surface, extracts the vertical column at each source position, stacks into a
depth-vs-position image. Each source position requires `nf` FDFD solves.

**CLI arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| *(all common args)* | -- | -- | See CLI_REFERENCE.md common table |
| `--src-step N` | int | 2 | Grid-cell step between source positions |
| `--src-iz IZ` | int | 20 | Source depth index (absolute, includes PML) |

**Usage:**

```bash
# Default
python examples/run_forward_bscan.py

# Recommended: stag2, parallel
python examples/run_forward_bscan.py --stag2 --ncpus 15

# Fine source spacing
python examples/run_forward_bscan.py --stag2 --ncpus 15 --src-step 1
```

**Output files:**

| File | Description |
|------|-------------|
| `results/forward/bscan/bscan.npz` | Stacked traces array (keys: `data`, `x`, `t`) |
| `results/forward/bscan/bscan.png` | Radargram with AGC2 normalisation |

---

### 7.4 `run_forward_cmp.py`

**Purpose:** Common Mid-Point (CMP) gather. For each half-offset, solves FDFD at
all frequencies, extracts the receiver response Ez[src_iz, rec_ix], and applies the
Hermitian IFFT to produce a time-vs-offset display. Uses `nf x n_offsets` FDFD
solves.

Default frequencies: GPRFM 10 discrete [50, 60, 70, 80, 90, 100, 125, 150, 175,
200] MHz. Override with `--fc-low`/`--fc-high`/`--nf` for a linspace sweep.

**CLI arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| *(all common args)* | -- | -- | See CLI_REFERENCE.md common table |
| `--n-offsets N` | int | 25 | Number of half-offset positions |
| `--offset-min M` | float | 0.1 | Minimum half-offset [m] |
| `--offset-max M` | float | 2.0 | Maximum half-offset [m] |
| `--mid-ix IX` | int | nx//2 | Mid-point x grid index |
| `--src-iz IZ` | int | 20 | Source/receiver depth index |
| `--fc-low HZ` | float | GPRFM | Override start frequency |
| `--fc-high HZ` | float | GPRFM | Override end frequency |
| `--nf N` | int | GPRFM | Override frequency count |
| `--tmax-ns NS` | float | 150.0 | Maximum display time [ns] |
| `--pad N` | int | 0 | Zero-samples per side of Hermitian spectrum |
| `--wiggle-gain G` | float | 1.5 | Wiggle amplitude scale |

**Usage:**

```bash
# Default: GPRFM 10 discrete freqs
python examples/run_forward_cmp.py --stag2 --ncpus 10

# Custom offset range
python examples/run_forward_cmp.py --stag2 --ncpus 10 --n-offsets 30 --offset-max 3.0

# Linspace sweep
python examples/run_forward_cmp.py --stag2 --ncpus 15 --fc-low 50e6 --fc-high 200e6 --nf 50
```

**Output files:**

| File | Description |
|------|-------------|
| `results/forward/cmp/cmp_<tags>.npz` | CMP trace array |
| `results/forward/cmp/cmp_<tags>.png` | Wiggle + density display |
| `results/forward/cmp/cmp_wiggle_<tags>.png` | Wiggle-only display |

---

### 7.5 `run_forward_shotgather.py`

**Purpose:** Time-domain shot gather for a single source position. Solves FDFD
at all frequencies, extracts all surface receivers, and applies the Hermitian IFFT.
Requires only `nf` FDFD solves -- much faster than the CMP script.

X-axis: signed offset from source [m] (negative = left, positive = right).
Y-axis: two-way travel time [ns].

**CLI arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| *(all common args)* | -- | -- | See CLI_REFERENCE.md common table |
| `--src-ix IX` | int | nx//2 | Source x grid index |
| `--src-iz IZ` | int | 20 | Source z grid index |
| `--rec-iz IZ` | int | =src-iz | Receiver z grid index |
| `--rec-step N` | int | 1 | Step between receivers [cells] |
| `--rec-ix-start IX` | int | npx+1 | First receiver x index |
| `--rec-ix-end IX` | int | nx-npx-1 | Last receiver x index |
| `--fc-low HZ` | float | GPRFM | Override start frequency |
| `--fc-high HZ` | float | GPRFM | Override end frequency |
| `--nf N` | int | GPRFM | Override frequency count |
| `--tmax-ns NS` | float | 150.0 | Maximum display time [ns] |
| `--pad N` | int | 0 | Zero-samples per Hermitian side |
| `--wiggle-gain G` | float | 1.5 | Wiggle amplitude scale |

**Usage:**

```bash
# Default: source at nx//2, GPRFM 10 freqs
python examples/run_forward_shotgather.py --stag2 --ncpus 10

# Custom source position
python examples/run_forward_shotgather.py --stag2 --ncpus 10 --src-ix 60

# Sparser receivers, deeper source
python examples/run_forward_shotgather.py --stag2 --ncpus 10 --src-ix 99 --rec-step 2

# Linspace frequency sweep
python examples/run_forward_shotgather.py --stag2 --ncpus 15 --fc-low 50e6 --fc-high 200e6 --nf 50
```

**Output files:**

| File | Description |
|------|-------------|
| `results/forward/shotgather/sg_<tags>.npz` | Shot gather trace array |
| `results/forward/shotgather/sg_<tags>.png` | Density + wiggle display |
| `results/forward/shotgather/sg_wiggle_<tags>.png` | Wiggle-only display |

---

### 7.6 `run_forward_shotgather_center.py`

**Purpose:** Convenience wrapper around `run_forward_shotgather.py` with the
source fixed at the grid centre (`src-ix = nx//2`). Accepts the same CLI arguments.

```bash
python examples/run_forward_shotgather_center.py --stag2 --ncpus 10
```

Output directory: `results/forward/shotgather/` (same as `run_forward_shotgather.py`).

---

### 7.7 `run_inversion_example.py`

**Purpose:** Full-Waveform Inversion following the MATLAB RFDFWI.m workflow.
Generates synthetic observed data at GPRFM 10 discrete frequencies for all 82
sources in the 4-sided acquisition, then runs the adjoint-state FWI loop with
Tikhonov regularisation and Armijo backtracking line search.

Observed data shape: `d_obs[n_src, n_freq, n_rec]` complex128.

**CLI arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| *(all common args)* | -- | -- | See CLI_REFERENCE.md common table |
| `--true-epsr V` | float | from config model | Homogeneous true epsr (overrides YAML model) |
| `--true-sigma V` | float | from config model | Homogeneous true sigma [S/m] |
| `--init-smooth PX` | float | 6.0 | Gaussian smooth [pixels], true->initial model |
| `--init-epsr V` | float | None | Homogeneous initial epsr (overrides --init-smooth) |
| `--init-sigma V` | float | None | Homogeneous initial sigma [S/m] |
| `--max-iter N` | int | from config | Override inversion max iterations |
| `--lambda-sigma V` | float | from config | Override Tikhonov LAMBDA_1 (MATLAB: 2e-4) |
| `--step-init V` | float | auto | Override initial step (<=0 = auto-scale) |
| `--fc-low HZ` | float | None | Switch to linspace sweep: start frequency |
| `--fc-high HZ` | float | None | Linspace sweep: end frequency |
| `--nf N` | int | None | Linspace sweep: number of frequencies |

**Usage:**

```bash
# Default: mkl_two_cross true model, GPRFM 10 freqs, 4-sided acquisition
python examples/run_inversion_example.py --stag2 --ncpus 15

# Quick test with homogeneous true model, 10 iterations
python examples/run_inversion_example.py --stag2 --ncpus 4 \
    --true-epsr 9.0 --true-sigma 0.01 --init-epsr 4.0 --init-sigma 3e-3 --max-iter 10

# Custom regularisation
python examples/run_inversion_example.py --stag2 --ncpus 15 --lambda-sigma 2e-4 --max-iter 50
```

**Output files:**

| File | Description |
|------|-------------|
| `results/inversion/obs/d_obs.npz` | Synthetic observed data [n_src, n_freq, n_rec] complex |
| `results/inversion/models/iter_<N>_epsr.png` | epsr model at iteration N |
| `results/inversion/models/iter_<N>_sigma.png` | sigma model at iteration N |
| `results/inversion/models/final_result.npz` | final + initial + true model arrays |
| `results/inversion/misfit/misfit_curve.png` | L2 convergence plot (log scale) |
| `results/inversion/logs/run_log.txt` | Run metadata + per-iteration misfit table |

The iteration callback `extras` dict contains:
`L2`, `grad_epsr`, `grad_sigma`, `hess_epsr`, `hess_sigma`, `tikh_epsr`,
`tikh_sigma`, `reg_grad_epsr`, `reg_grad_sigma`, `dir_epsr`, `dir_sigma`,
`step`, `delta_epsr`, `delta_sigma`.

---

### 7.8 `validate_python.py`

**Purpose:** Automated solver correctness checks. Runs two tests that must both
pass before any forward or inversion results should be trusted.

**CLI arguments:** None (runs automatically).

```bash
python examples/validate_python.py
```

**Tests performed:**

| Test | Description | Pass criterion |
|------|-------------|----------------|
| Forward residual | Solves A u = b, checks ||A u - b|| | Residual < 1e-10 |
| Misfit decrease | Runs 3 FWI iterations on a simple model | L2[iter1] < L2[iter0] |

Both tests print PASS / FAIL and exit with code 0 on success, 1 on failure.

---

## 8. Core Algorithms

### 8.1 CFS-PML construction

Complex-Frequency-Shifted PML is applied on all four sides with thickness `npx`
(x) and `npz` (z) cells. The stretched coordinate in direction d is:

```
s_d(d) = kappa_d + sigma_d / (alpha_d + j*omega)
```

where `sigma_d` is a polynomial profile ramping from 0 at the inner boundary to
`a0_cfs` at the outer boundary, `alpha_d` is a constant shift, and `kappa_d = 1`
inside the PML. The PML modifies the effective wavenumber in the Helmholtz equation.

The `a0_cfs` parameter (MATLAB `sig_max`) is set in the YAML:
```yaml
pml:
  a0_cfs: 9.0e8    # Chen et al. 2013 value
```

### 8.2 9-point Helmholtz stencil (stag1 vs stag2)

The 2-D TE-mode Helmholtz equation for the Ez component is:

```
d/dx(1/mu * dEz/dx) + d/dz(1/mu * dEz/dz) + omega^2 * eps* * Ez = -j*omega*mu0*Jz
```

where `eps* = eps0*epsr - j*sigma/omega` is the complex permittivity.

**stag1** (Hustedt et al. 2004): The parallel staggered-grid 9-point stencil.
Coefficients are computed by averaging the medium properties on shifted half-cell
grids in the MATLAB style.

**stag2** (Layek & Sengupta 2024): A new 9-point staggered-grid formulation with
improved dispersion properties at higher frequencies. Recommended for most runs.

Both stencils assemble a sparse `(nx*nz) x (nx*nz)` complex matrix A.

### 8.3 Forward solve (sparse LU)

```python
u = scipy.sparse.linalg.spsolve(A, b)
```

- A is assembled once per frequency (shape: N x N where N = nx x nz)
- b is the right-hand side source vector (point source injection)
- u is reshaped to (nz, nx) to give the Ez wavefield
- For multiple sources at the same frequency, A is factored once and each RHS
  is solved by forward/back substitution (UMFPACK under the hood)

### 8.4 Source amplitude

The source amplitude matches MATLAB `RHS_TE1.m`:

```
src_amp = -(omega * mu0 * j) / dh^2
```

where `j` is the imaginary unit, omega = 2*pi*f, mu0 = 4*pi*1e-7 H/m, and dh
is the cell size [m]. The factor `1/dh^2` converts the point source from volume
density to grid units.

### 8.5 Time-domain conversion (Hermitian IFFT + Blackman-Harris)

The frequency-domain data at `nf` positive frequencies is converted to time domain
using the GPRFM Hermitian IFFT method:

1. Zero-pad the spectrum symmetrically (optional `--pad` argument)
2. Apply Blackman-Harris window: clip amplitudes at `clip` then at `clip1`
3. Construct one-sided + conjugate-mirrored Hermitian spectrum
4. Apply `numpy.fft.ifft` along the frequency axis
5. Take the real part; scale by the number of frequency points

The time axis is: `t = numpy.arange(N_fft) / (df * N_fft)` where
`N_fft = 2 * (nf + pad)`.

### 8.6 AGC2 normalisation

For B-scan display, each time trace is normalised by its own RMS energy computed
in a running window (Automatic Gain Control, order 2). This suppresses geometrical
spreading and enhances late-arriving reflections.

### 8.7 Adjoint-state FWI gradient

Given observed data `d_obs` and predicted data `d_pre`, the residual is:

```
delta_d = d_pre - d_obs
```

The adjoint source is `-delta_d` injected at receiver positions. The gradient with
respect to the medium parameters at each grid point (i, k) is:

```
grad_epsr[i,k]  += Re( omega^2 * conj(u[i,k]) * lambda[i,k] )  summed over freq, src
grad_sigma[i,k] += Re( j*omega * conj(u[i,k]) * lambda[i,k] )  summed over freq, src
```

where `u` is the forward wavefield and `lambda` is the adjoint (back-propagated)
wavefield, both solutions to the same Helmholtz system (with different RHS).

### 8.8 Pseudo-Hessian diagonal

The diagonal of the approximate Hessian is used as a pre-conditioner to scale
the gradient:

```
hess_epsr[i,k]  += omega^4 * |u[i,k]|^2   summed over freq, src
hess_sigma[i,k] += omega^2 * |u[i,k]|^2   summed over freq, src
```

The search direction is: `dir = -grad / (hess + eps)` where `eps` is a small
stabilisation constant.

### 8.9 Tikhonov regularisation

Laplacian smoothing is applied to the conductivity gradient:

```
reg_grad_sigma = grad_sigma + LAMBDA_1 * Laplacian(sigma_current)
```

where `Laplacian(sigma)` is the discrete 2-D Laplacian (5-point stencil). With
the default `LAMBDA_1 = 2e-4` this matches the MATLAB RFDFWI.m regularisation.
The epsr regularisation weight `LAMBDA_2` is 0 by default (no epsr smoothing).

### 8.10 Armijo backtracking line search

Starting from `step_init` (auto: `L2 / ||gradient||^2`), the step size is halved
up to `STEPMAX` times until the Armijo sufficient-decrease condition is met:

```
L2(m + step * dir) <= L2(m) - C1 * step * ||gradient||^2
```

Default parameters: `STEPMAX=3`, `SCALEFAC=2`, `C1=1e-4`.

---

## 9. Output Files Reference

### Forward wavefield

| File | Description |
|------|-------------|
| `wavefield_real.png` | PNG, 300 DPI, seismic colormap |
| `wavefield_real.tiff` | Same, TIFF for MATLAB import |

### B-scan

| File | Keys | Shape | Dtype |
|------|------|-------|-------|
| `bscan.npz` | `data` | (nz_interior, n_sources) | float64 |
| | `x` | (n_sources,) | float64 |
| | `t` | (nz_interior,) | float64 |
| `bscan.png` | -- | AGC2 normalised radargram | -- |

### CMP gather

| File | Keys | Shape | Dtype |
|------|------|-------|-------|
| `cmp_<tags>.npz` | `data` | (n_time, n_offsets) | float64 |
| | `offsets` | (n_offsets,) | float64 |
| | `t` | (n_time,) | float64 |
| `cmp_<tags>.png` | -- | Density + wiggle display | -- |
| `cmp_wiggle_<tags>.png` | -- | Wiggle-only display | -- |

### Shot gather

| File | Keys | Shape | Dtype |
|------|------|-------|-------|
| `sg_<tags>.npz` | `data` | (n_time, n_receivers) | float64 |
| | `offsets` | (n_receivers,) | float64 |
| | `t` | (n_time,) | float64 |
| `sg_<tags>.png` | -- | Density + wiggle display | -- |
| `sg_wiggle_<tags>.png` | -- | Wiggle-only display | -- |

### Inversion outputs

| File | Keys | Shape | Notes |
|------|------|-------|-------|
| `d_obs.npz` | `data` | (n_src, n_freq, n_rec) complex128 | Synthetic observed data |
| `final_result.npz` | `epsr_final` | (nz, nx) float64 | Recovered epsr |
| | `sigma_final` | (nz, nx) float64 | Recovered sigma |
| | `epsr_init` | (nz, nx) float64 | Initial model epsr |
| | `sigma_init` | (nz, nx) float64 | Initial model sigma |
| | `epsr_true` | (nz, nx) float64 | True model epsr |
| | `sigma_true` | (nz, nx) float64 | True model sigma |
| `iter_<N>_epsr.png` | -- | -- | epsr at iteration N, jet colormap |
| `iter_<N>_sigma.png` | -- | -- | sigma at iteration N, jet colormap |
| `misfit_curve.png` | -- | -- | L2/L2[0] vs iteration, log scale |

### run_log.txt format

```
RFDFWI Run Log
==============
Date       : YYYY-MM-DD HH:MM:SS
Stencil    : stag2
ncpus      : 15
n_src      : 82
n_freq     : 10
n_rec      : 162
LAMBDA_1   : 2.0e-04
conv_tol   : 5.0e-05

Iteration  L2            L2/L2[0]      Step
---------  ------------  ------------  --------
0          1.2345e+00    1.0000e+00    --
1          9.8765e-01    8.0000e-01    3.2e-05
...
CONVERGED at iteration N: L2/L2[0] = 4.8e-05
```

---

## 10. MATLAB Correspondence

### Complete file mapping

| MATLAB file | Python equivalent | Notes |
|-------------|-------------------|-------|
| `inp_GPRmodel1.m` | `input/input_forward.yaml` | Parameters -> YAML keys |
| `create_models_mkl.m` | `create_models/build_models.py` -> `mkl_two_cross` | Index 1->0 offset |
| `RHS_TE1.m` | `scripts/forward_fdfd.py` -> source injection | Same amplitude formula |
| `Helmholtz_9pCFSPML_stag.m` | `scripts/forward_fdfd.py` -> `build_helmholtz_2d()` stag1 | 9-point coefficients |
| `Helmholtz_9pCFSPML_stag2.m` | `scripts/forward_fdfd.py` -> `build_helmholtz_2d()` stag2 | 9-point coefficients |
| `GPRFM_freq2time.m` | `scripts/forward_fdfd.py` -> `freq_to_timedomain()` | Hermitian IFFT |
| `blackman_harris.m` | `scripts/forward_fdfd.py` -> `blackman_harris_spectrum()` | 4-term window |
| `RFDFWI.m` | `scripts/inversion_fwi.py` -> `run_inversion()` | Main FWI loop |
| `gradient_TE.m` | `scripts/inversion_fwi.py` -> `compute_gradient()` | Adjoint state |
| `hessian_diag_TE.m` | `scripts/inversion_fwi.py` -> `compute_hessian()` | Pseudo-Hessian |
| `tikhonov_reg.m` | `scripts/inversion_fwi.py` -> regularisation block | Laplacian smoothing |
| `armijo_linesearch.m` | `scripts/inversion_fwi.py` -> line search block | Backtracking |
| `build_4sided_acquisition.m` | `scripts/inversion_fwi.py` -> `build_4sided_acquisition()` | Geometry |

### Index translation

MATLAB uses 1-based indexing; Python uses 0-based indexing:

| MATLAB | Python | Physical position |
|--------|--------|-------------------|
| `ix = 100` | `ix = 99` | Source x (grid centre) |
| `iz = 21` | `iz = 20` | Top acquisition row |
| `ix = 180` | `ix = 179` | Max acquisition x |
| `ix_range = 21:180` | `ix_range = 20:180` | Acquisition range |

### CFS-PML parameter mapping

| MATLAB variable | Python / YAML key | Value |
|-----------------|-------------------|-------|
| `sig_max` | `pml.a0_cfs` | 9.0e8 |
| `npx`, `npz` | `pml.npx`, `pml.npz` | 10, 10 |
| `FC_low` | `freq_sweep.fc_low` | 50e6 |
| `FC_high` | `freq_sweep.fc_high` | 200e6 |
| `nf` | `freq_sweep.nf` | 50 |
| `TmaxTD` | `freq_sweep.tmax_td` | 150e-9 |
| `clip` | `freq_sweep.clip` | 2.5e-3 |
| `clip1` | `freq_sweep.clip1` | 1.0e-2 |
| `LAMBDA_1` | `inversion.regularization.LAMBDA_1` | 2e-4 |
| `LAMBDA_2` | `inversion.regularization.LAMBDA_2` | 0.0 |

### Source amplitude formula

Both MATLAB and Python compute:

```
b[iz_src, ix_src] = -(omega * mu0 * j) / dh^2
```

Python implementation in `forward_fdfd.py`:

```python
src_amp = -(omega * mu0 * 1j) / dh**2
```

### Field component note

The vertical electric field is the Ez component in TE mode. The MATLAB code and
some plot titles label this as "Ey" (a historical naming convention from the
original 3-D-to-2-D reduction). Both refer to the same vertical E-field component.

---

## 11. Parallelism & Performance

### ThreadPoolExecutor usage

Multi-source and multi-frequency solves are parallelised using
`concurrent.futures.ThreadPoolExecutor`. The number of worker threads is set with:

```bash
python examples/run_forward_bscan.py --ncpus 15
```

The solver calls `scipy.sparse.linalg.spsolve`, which releases the Python GIL
(via UMFPACK/SuperLU), so thread-level parallelism is effective.

### --ncpus flag guidance

| Script | Parallelised over | Recommended ncpus |
|--------|------------------|-------------------|
| `run_forward_wavefield.py` | Frequencies | 8-15 |
| `run_forward_bscan.py` | Source positions | 8-15 |
| `run_forward_cmp.py` | Offset x frequency combinations | 8-15 |
| `run_forward_shotgather.py` | Frequencies | 8-10 |
| `run_inversion_example.py` | Sources per iteration | 8-15 |

Setting `--ncpus` higher than the number of cores provides diminishing returns.
Setting it higher than the number of tasks (sources, frequencies) wastes threads.

### GPU status and limitations

The `--use-gpu` flag is accepted by all scripts. When set, the code checks for CuPy
and, if available, may use it for dense array operations. However,
`scipy.sparse.linalg.spsolve` (the main computational bottleneck) **does not
currently use the GPU**. The flag is reserved for future integration of a
CuPy-based sparse solver. There is no loss of accuracy or correctness when
`--use-gpu` is used with only a CPU -- it simply falls back to the CPU solver.

---

## 12. Troubleshooting

### Common issues and solutions

| Problem | Likely cause | Solution |
|---------|-------------|----------|
| `ModuleNotFoundError: yaml` | PyYAML not installed | `pip install pyyaml` |
| `ModuleNotFoundError: scipy` | SciPy not installed | `pip install scipy` |
| `Config not found` | Running from wrong directory | Run from project root `D:\rfdfwi` |
| Wavefield all zeros | Wrong source index (outside domain) | Check `--source-ix` is in [npx, nx-npx-1] |
| Misfit not decreasing | Step size too large or regularisation too weak | Reduce `step_init` or increase `LAMBDA_1` |
| Slow forward runs | ncpus=1 default | Increase `--ncpus` (up to number of sources) |
| Memory error on large grids | N x N sparse matrix too large | Reduce `nx`/`nz` or `nf` |
| TIFF not opening in MATLAB | Matplotlib format mismatch | Ensure `format="tiff"` (default behaviour) |
| Results written to wrong directory | Relative vs absolute path confusion | Run from project root or use `--results-dir` |
| `impedance_matrix.npz` not created | Flag not set | Add `--impedance-matrix` to the command |
| Validation test fails: residual | Stencil assembly bug | Check stag1/stag2 flag and PML parameters |
| FWI diverges after first iteration | Initial model too far from true | Use `--init-smooth 6.0` (smooth-then-start) |
| conda env not activating | Wrong environment name | Use `conda activate rfdfwimkl` (not rfdfwi-env) |
| B-scan has artefacts at edges | PML too thin | Increase `pml.npx`/`pml.npz` (try 15-20) |

### Environment activation reminder

```bash
# Correct environment name (not rfdfwi-env)
conda activate rfdfwimkl

# Verify you are in the right environment
python -c "import sys; print(sys.prefix)"
# Should show a path containing 'rfdfwimkl'
```

### Checking the sparse solve residual manually

```python
import numpy as np
import scipy.sparse as sp

A = sp.load_npz("results/forward/wavefield/impedance_matrix.npz")
# Load b and u from the run, then:
residual = np.linalg.norm(A @ u.ravel() - b.ravel())
print(f"Residual: {residual:.2e}")   # Should be < 1e-10
```

---

## 13. API Quick Reference

The following function signatures are the primary API entry points. All are in the
`scripts/` directory.

### `build_helmholtz_2d` (`scripts/forward_fdfd.py`)

```python
def build_helmholtz_2d(
    epsr: np.ndarray,       # (nz, nx) relative permittivity
    sigma: np.ndarray,      # (nz, nx) conductivity [S/m]
    freq: float,            # frequency [Hz]
    dh: float,              # cell size [m]
    npx: int,               # PML cells in x
    npz: int,               # PML cells in z
    a0_cfs: float,          # CFS-PML sigma_max
    stencil: str = 'stag1', # 'stag1' or 'stag2'
) -> scipy.sparse.csc_matrix:
    """Return the sparse Helmholtz system matrix A."""
```

### `run_forward_single_source` (`scripts/forward_fdfd.py`)

```python
def run_forward_single_source(
    epsr: np.ndarray,       # (nz, nx)
    sigma: np.ndarray,      # (nz, nx)
    freq: float,            # [Hz]
    src_ix: int,            # source x index (0-based)
    src_iz: int,            # source z index (0-based)
    dh: float,
    npx: int,
    npz: int,
    a0_cfs: float,
    stencil: str = 'stag1',
    use_gpu: bool = False,
) -> np.ndarray:
    """Solve A u = b for one source; return Ez (nz, nx) complex128."""
```

### `compute_forward_data` (`scripts/forward_fdfd.py`)

```python
def compute_forward_data(
    epsr: np.ndarray,
    sigma: np.ndarray,
    freqs: list,            # list of frequencies [Hz]
    sources: list,          # [(src_ix, src_iz), ...]
    receivers: list,        # [(rec_ix, rec_iz), ...]
    dh: float,
    npx: int,
    npz: int,
    a0_cfs: float,
    stencil: str = 'stag1',
    ncpus: int = 1,
    use_gpu: bool = False,
) -> np.ndarray:
    """Return d[n_src, n_freq, n_rec] complex128."""
```

### `freq_to_timedomain` (`scripts/forward_fdfd.py`)

```python
def freq_to_timedomain(
    spec: np.ndarray,       # (..., nf) complex -- one-sided spectrum
    df: float,              # frequency step [Hz]
    clip: float = 2.5e-3,   # Blackman-Harris amplitude clip
    clip1: float = 1.0e-2,  # secondary clip
    pad: int = 0,           # zero-samples per side
) -> tuple:
    """Return (time_traces, t_axis) where t_axis is in seconds."""
```

### `blackman_harris_spectrum` (`scripts/forward_fdfd.py`)

```python
def blackman_harris_spectrum(
    spec: np.ndarray,       # (..., nf) complex -- one-sided spectrum
    clip: float,
    clip1: float,
) -> np.ndarray:
    """Apply 4-term Blackman-Harris window in frequency domain."""
```

### `compute_gradient` (`scripts/inversion_fwi.py`)

```python
def compute_gradient(
    epsr: np.ndarray,
    sigma: np.ndarray,
    d_obs: np.ndarray,      # [n_src, n_freq, n_rec] complex
    freqs: list,
    sources: list,
    receivers: list,
    dh: float,
    npx: int,
    npz: int,
    a0_cfs: float,
    stencil: str = 'stag1',
    ncpus: int = 1,
) -> tuple:
    """Return (grad_epsr, grad_sigma, L2_misfit)."""
```

### `run_inversion` (`scripts/inversion_fwi.py`)

```python
def run_inversion(
    epsr_init: np.ndarray,
    sigma_init: np.ndarray,
    d_obs: np.ndarray,
    freqs: list,
    sources: list,
    receivers: list,
    dh: float,
    npx: int,
    npz: int,
    a0_cfs: float,
    stencil: str = 'stag1',
    ncpus: int = 1,
    max_iter: int = 20,
    LAMBDA_1: float = 2e-4,
    LAMBDA_2: float = 0.0,
    conv_tol: float = 5e-5,
    bounds: dict = None,
    callback: callable = None,
) -> tuple:
    """Run FWI loop; return (epsr_final, sigma_final)."""
```

---

## References

- Layek, M. K., & Sengupta, P. (2024). Multi-parameter imaging by finite difference
  frequency domain full waveform inversion of GPR data: A guide for sedimentary
  architecture modeling. *Pure and Applied Geophysics*, 181, 2107–2130.
  https://doi.org/10.1007/s00024-024-03520-1
- Hustedt, B., Operto, S., & Virieux, J. (2004). Mixed-grid and staggered-grid
  finite-difference methods for frequency-domain acoustic wave modelling.
  *Geophysical Journal International*, 157(3), 1269-1296.
- Chen, J.-B. (2013). A generalized optimal 9-point scheme for frequency-domain
  scalar wave equation. *Journal of Applied Geophysics*, 92, 1-7.

---

*This manual covers RFDFWI as of March 2026. For developer notes, see `CLAUDE.md`.*
