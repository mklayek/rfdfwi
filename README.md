# RFDFWI (Python)

2-D Frequency-Domain Finite-Difference (FDFD) forward modelling and Full-Waveform
Inversion (FWI) for Ground-Penetrating Radar (GPR) — Python port of the MATLAB
RFDFWI toolbox (Layek & Sengupta 2024).

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform: Windows/Linux](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

---

## Key features

- **Two 9-point CFS-PML stencils**: stag1 (Hustedt et al. 2004) and stag2
  (Layek & Sengupta 2023, recommended)
- **Exact MATLAB replica**: 200x200 grid (180 interior + 10-cell PML each side),
  dh=0.05 m, physical domain 9 m x 9 m, frequencies 50-200 MHz (nf=50)
- **TE-mode forward solver**: sparse LU via `scipy.sparse.linalg.spsolve`;
  Ez component (plot titles may show "Ey" — same physical quantity)
- **Full-Waveform Inversion**: adjoint-state gradient, pseudo-Hessian
  pre-conditioner, Tikhonov (Laplacian) regularisation, Armijo backtracking
- **4-sided acquisition geometry**: 82 sources + 162 receivers around all four
  sides of the interior domain
- **Time-domain displays**: B-scan, CMP gather, shot gather via Hermitian IFFT
  + Blackman-Harris windowing
- **Parallel solving**: `--ncpus N` via `ThreadPoolExecutor` (GIL-releasing spsolve)
- **YAML-driven configuration** with CLI overrides for every parameter
- **GPU flag** (`--use-gpu`) accepted; sparse solver GPU support reserved for future

---

## Quick start

```bash
conda activate rfdfwimkl

# Step 1 -- Build the mkl_two_cross model (exact MATLAB replica, 200x200 grid)
python examples/run_build_model.py

# Step 2 -- Multi-frequency forward wavefield (nf=50, 50-200 MHz, stag2)
python examples/run_forward_wavefield.py --stag2 --ncpus 15

# Step 3 -- Zero-offset B-scan radargram
python examples/run_forward_bscan.py --stag2 --ncpus 15

# Step 4 -- CMP gather
python examples/run_forward_cmp.py --stag2 --ncpus 15

# Step 5 -- Full-Waveform Inversion (4-sided acq., 82 sources, GPRFM 10 freqs)
python examples/run_inversion_example.py --stag2 --ncpus 15
```

All outputs are written under `results/`.

---

## Validation

```bash
python examples/validate_python.py
```

Runs two automated checks:

1. **Forward residual** -- verifies `||A u - b|| < 1e-10`
2. **Misfit decrease** -- confirms misfit decreases over 3 FWI iterations

Both must print PASS before trusting any results.

---

## Documentation

| Document | Description |
|----------|-------------|
| [INSTALLATION.md](INSTALLATION.md) | Environment setup (conda), step-by-step workflow, output directory guide, troubleshooting |
| [CLI_REFERENCE.md](CLI_REFERENCE.md) | Complete CLI flags for all example scripts, stencil selection, model types, acquisition geometry |
| [docs/MANUAL.md](docs/MANUAL.md) | Full reference manual: algorithms, YAML config, output file formats, MATLAB correspondence, API |
| [docs/MATLAB_to_Python_Mapping.md](docs/MATLAB_to_Python_Mapping.md) | Function-level MATLAB -> Python mapping table |
| [CLAUDE.md](CLAUDE.md) | Developer notes and internal architecture |

---

## Project summary

| Parameter | Value |
|-----------|-------|
| Grid size | 200 x 200 (180 interior + 10 PML each side) |
| Cell size dh | 0.05 m |
| Physical domain | 9 m x 9 m |
| Frequencies (wavefield) | 50 to 200 MHz, nf=50, df ~ 3.06 MHz |
| Frequencies (FWI/CMP) | GPRFM 10 discrete: 50,60,70,80,90,100,125,150,175,200 MHz |
| Acquisition | 4-sided: 82 sources, 162 receivers |
| Default stencil | stag2 (recommended) |
| Tikhonov LAMBDA_1 | 2e-4 (MATLAB default) |
| Convergence criterion | L2/L2[0] <= 5e-5 |

---

## References

- Layek, M. K., & Sengupta, P. (2024). Multi-parameter imaging by finite difference frequency domain full waveform inversion of GPR data: A guide for sedimentary architecture modeling. *Pure and Applied Geophysics*, 181, 2107–2130. https://doi.org/10.1007/s00024-024-03520-1
- Hustedt, B., Operto, S., & Virieux, J. (2004). Mixed-grid and staggered-grid
  finite-difference methods for frequency-domain acoustic wave modelling.
  *Geophysical Journal International*, 157(3), 1269-1296.
