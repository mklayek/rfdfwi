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
# RFDFWI – Python Implementation  
**2D Frequency-Domain FDFD + Full Waveform Inversion for GPR**

Last major update: early 2025  
Status: stable single-frequency & frequency-sweep forward modeling, adjoint-state FWI (L2 + Tikhonov), synthetic inversion workflows  
Primary goal: faithful Python port & modernization of the MATLAB RFDFWI codebase (Layek & Sengupta 2024)

## 1. What is this project?

Modern, clean Python implementation of **2D TMz finite-difference frequency-domain (FDFD)** modeling and **full-waveform inversion (FWI)** tailored for ground-penetrating radar (GPR).

Key features matching / improving the original MATLAB version:

- Complex-frequency stretched (CFS) PML — both classic (Hustedt 2004) and improved staggered (Layek & Sengupta 2024 = stag2)
- 9-point stencil (more accurate than 5-point)
- Multi-source forward modeling (parallelized over sources / frequencies)
- Adjoint-state gradient + steepest descent with linesearch or fixed step
- Tikhonov regularization + hard bounds on εᵣ and σ
- Synthetic B-scan, CMP gathers, wavefield snapshots
- Exact parameter & geometry matching option for the famous “two-cross” benchmark model

## 2. Quick Start – Full MATLAB-matched Workflow

```bash
# ────────────────────────────────────────────────
#  Windows / PowerShell / Anaconda Prompt
# ────────────────────────────────────────────────

# 0. Activate environment (do this every new terminal)
conda activate rfdfwimkl
cd D:\rfdfwi

# 1. Build exact "two-cross" model (epsr & sigma) — matches create_models_mkl.m
python examples/run_build_model.py

# 2. Multi-frequency wavefield snapshot (50–200 MHz, stag2 stencil, 15 threads)
python examples/run_forward_wavefield.py --stag2 --ncpus 15

# 3. Zero-offset B-scan (radargram)
python examples/run_forward_bscan.py --stag2 --ncpus 15

# 4. Full-waveform inversion (4-sided acquisition, 82 sources)
python examples/run_inversion_example.py --stag2 --ncpus 15
```

Typical output folders (2025 layout):

```
results/
├── forward/
│   ├── wavefield/       wavefield_real.png + .tiff
│   ├── bscan/           bscan_traces.npz   bscan.png
│   └── cmp/             cmp_traces.npz     cmp.png  cmp_wiggle.png
└── inversion/
    ├── models/          inversion_epsr_000.png … inversion_epsr_final.png
    ├── misfit/          inversion_misfit.png   inversion_history.npz
    └── logs/            run_log.txt
```

## 3. Most Important Folders & Files

| Path                                 | Purpose                                                                 |
|:-------------------------------------|:-----------------------------------------------------------------------|
| `examples/run_*.py`                  | Main runnable workflows (recommended entry points)                     |
| `scripts/forward_fdfd.py`            | Core single-frequency FDFD solver (builds & solves complex Helmholtz)  |
| `scripts/inversion_fwi.py`           | Gradient computation, linesearch, update step, Tikhonov term           |
| `create_models/build_models.py`      | All analytical & parametric model generators                           |
| `input/*.yaml`                       | Configuration files (forward + inversion)                              |
| `inputmodel/`                        | Saved εᵣ, σ arrays (.npy) + visualization PNGs                         |
| `docs/MATLAB_to_Python_Mapping.md`   | **Most valuable reference** — parameter & index translation table      |
| `docs/MANUAL.md`                     | YAML schema documentation                                              |
| `CLI_REFERENCE.md`                   | Every command-line flag explained                                      |

## 4. Model Types You Can Use Right Now

```yaml
model:
  type: one of
    • mkl_two_cross        # exact replica of MATLAB create_models_mkl.m
    • two_cross            # parametric cross (center_x, center_z, half_len_m)
    • homogeneous
    • layered
    • file                 # load existing inputmodel/model_{epsr,sigma}.npy
```

## 5. Acquisition Geometries (acquisition / receivers)

```yaml
# Classic 4-sided surface acquisition (MATLAB ACQMY=1)
acquisition:
  mode: 4sided
  nrec_per_side: 40
  nsrc_per_side: 20

# Surface reflection line (most common for B-scan)
receivers:
  mode: line
  iz: 20               # usually npml + some small number
  ix_start: 20
  ix_end: 179
```

## 6. Recommended Frequency Sweep (matches inp_GPRmodel1.m)

```yaml
freq_sweep:
  fc_low:   50e6
  fc_high:  200e6
  nf:       50           # → df ≈ 3.061 MHz
  clip:     2.5e-3       # Blackman-Harris main lobe clip
  clip1:    1.0e-2
  tmax_td:  150e-9       # for IFFT / plotting
```

## 7. Inversion – What actually works well (2025)

- Single-frequency FWI (fast for debugging)
- Multi-frequency sequential FWI (recommended)
- Tikhonov regularization (`regularization.alpha`)
- Hard box constraints (`bounds.epsr_min/max`, `sigma_min/max`)
- Linesearch (`step_type: linesearch`) usually more robust than fixed step

Known good starting point (inversion section):

```yaml
inversion:
  max_iter: 30
  step_type: linesearch
  step_init: 0.8
  regularization:
    alpha: 1e-4           # tune between 5e-5 – 5e-3
  bounds:
    epsr_min:  2.0
    epsr_max:  20.0
    sigma_min: 0.0001
    sigma_max: 0.05
```

## 8. Quick Troubleshooting Table

| Symptom                                | Most likely fix / check                                                     |
|:---------------------------------------|:----------------------------------------------------------------------------|
| Misfit explodes / NaN                  | Decrease `step_init`, increase `regularization.alpha`, check source amplitude |
| Very slow forward modeling             | Use `--ncpus 12–20`, consider `--use-gpu` if CuPy installed                 |
| PML reflections visible                | Try `--stag2`, increase `pml.npx/npz` to 12–15, check `a0_cfs` ~1e8–1e9    |
| Inversion stuck / doesn't update       | Wrong gradient sign → check source RHS sign convention                      |
| Index mismatch with MATLAB             | Read `docs/MATLAB_to_Python_Mapping.md` — npml offset is critical           |
| TIFF figures not opening               | Use PNG instead (`--no-tiff`) or install better TIFF viewer                 |

## 9. Final Recommended “Production” Command Set (2025)

```bash
# Build model once
python examples/run_build_model.py

# High-quality wavefield for paper / presentation (stag2)
python examples/run_forward_wavefield.py --stag2 --ncpus 16 --caxis 8.0

# Clean B-scan
python examples/run_forward_bscan.py --stag2 --ncpus 16 --src-step 1

# Full inversion – start here when developing new experiments
python examples/run_inversion_example.py --stag2 --ncpus 16 --config input/my_inversion.yaml
```

Good luck with your GPR FWI experiments!  
— the RFDFWI Python port (2024–2025)

```

Feel free to copy this text into a file called `CLAUDE.md` (or `PROJECT_SUMMARY.md`) at the root of the repository.

If you want a shorter executive version, a version focused only on inversion, or want to add sections (e.g. performance benchmarks, GPU notes, real-data tips), just tell me.