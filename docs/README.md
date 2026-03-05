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
# RFDFWI – Python

2D Frequency-Domain Finite-Difference (FDFD) forward modeling and Full Waveform Inversion (FWI)
for ground-penetrating radar (GPR), ported from the MATLAB RFDFWI codebase.

## Requirements

- Python 3.11
- NumPy, SciPy, Matplotlib, PyYAML
- Optional: CuPy (GPU acceleration)

```bash
conda activate rfdfwimkl
```

## Project layout

```
rfdfwi/
├── scripts/          FDFD solver, config loader, plot utilities
├── create_models/    Model builders (mkl_two_cross, homogeneous, layered, file)
├── examples/         Runnable scripts (build_model, wavefield, bscan, cmp, inversion)
├── input/            YAML configs (matched to inp_GPRmodel1.m)
├── inputmodel/       Built model arrays (model_epsr.npy, model_sigma.npy)
├── results/          Output figures and traces
├── obs/              Observed data for inversion
├── docs/             This documentation
├── CLI_REFERENCE.md  All commands and arguments
└── README.md         Quick start
```

## MATLAB parameter correspondence

| MATLAB (`inp_GPRmodel1.m`) | Python config | Value |
|---|---|---|
| `nx=180, ny=180, dh=0.05` | `domain: nx:200, nz:200, dx:0.05` | 180 interior + 10 PML each side |
| `npml=10, a0_cfs=9e8` | `pml: npx:10, npz:10, a0_cfs:9e8` | CFS-PML |
| `f=50e6` | `freq_hz: 50e6` | Centre frequency |
| `FC_low=50e6, FC_high=200e6, nf=50` | `freq_sweep: fc_low/fc_high/nf` | df ≈ 3.06 MHz |
| `clip=2.5e-3, clip1=1e-2` | `freq_sweep: clip/clip1` | Blackman-Harris clip |
| `TmaxTD=150e-9` | `freq_sweep: tmax_td` | Max recording time |
| `sig_max=9e8` | `pml: a0_cfs` | PML sigma_max |
| `STAG=2` | `--stag2` CLI flag | New staggered grid |
| `epsr=4.0, sigma=3e-3` | `model: epsr_bg/sigma_bg` | Background |
| `ACQMY=1` (4-sided) | `acquisition: mode: 4sided` | 82 src, 162 rec |

## Workflow

```bash
# From project root with conda env active:
python examples/run_build_model.py                              # Build true model
python examples/run_forward_wavefield.py --stag2 --ncpus 15    # Wavefield (nf=50)
python examples/run_forward_bscan.py     --stag2 --ncpus 15    # B-scan radargram
python examples/run_inversion_example.py --stag2 --ncpus 15    # FWI inversion
```

See **[CLI_REFERENCE.md](../CLI_REFERENCE.md)** for the full command reference
and **[MANUAL.md](MANUAL.md)** for detailed configuration documentation.
