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
# RFDFWI — Command-Line Reference

> **Repository:** https://github.com/mklayek/rfdfwi

All example scripts are run from the **project root** with the conda
environment activated:

```bash
conda activate rfdfwimkl
cd D:\rfdfwi
```

---

## Common arguments (all forward/inversion scripts)

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config FILE` | str | *(script default)* | Path to YAML configuration file |
| `--results-dir DIR` | str | *(from config)* | Override the output directory |
| `--impedance-matrix` | flag | off | Save Helmholtz matrix as `impedance_matrix.npz` |
| `--ncpus N` | int | `1` | Parallel CPU workers for multi-source/frequency solves |
| `--use-gpu` | flag | off | GPU acceleration via CuPy (experimental) |
| `--stag1` | flag | **default** | stag1 9-point CFS-PML (Hustedt et al. 2004) |
| `--stag2` | flag | off | stag2 9-point CFS-PML (Layek & Sengupta 2024) |
| `-v / --verbose` | flag | off | Print extra diagnostic information |
| `--patience N` | int | *(from config)* | Early-stop after N consecutive non-decreasing misfit iterations |
| `--warmup N` | int | *(from config)* | Skip first N iterations before applying early-stop check |
| `--step-epsr S` | float | *(from config)* | Override `step_init_epsr` — initial step size for εᵣ update |
| `--step-sigma S` | float | *(from config)* | Override `step_init_sigma` — initial step size for σ update |

> `--stag1` and `--stag2` are mutually exclusive; `--stag1` is the default.

---

## 1. `run_build_model.py`

Build and save the 2-D GPR model (permittivity + conductivity) from a YAML config.
Produces NumPy arrays and MATLAB-style PNG figures.

**Arguments**

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config FILE` | str | `input/input_forward.yaml` | Path to YAML config file |

**Examples**

```bash
# Default — builds mkl_two_cross model (exact MATLAB replica)
python examples/run_build_model.py

# With model-file config (loads inputmodel/model_epsr.npy if already built)
python examples/run_build_model.py --config input/input_forward_with_model.yaml
```

**Output:** `inputmodel/model_epsr.npy`, `model_sigma.npy`, `model_eps_sig.png`

---

## 2. `run_forward_wavefield.py`

FDFD multi-frequency wavefield — MATLAB-style figure (seismic colormap,
no source/receiver markers, PNG + TIFF output).

Sweeps `nf` frequencies from `FC_low` to `FC_high` using
`df = (FC_high - FC_low) / (nf - 1)` (MATLAB convention).
The wavefield at the **lowest frequency** (`FC_low = 50 MHz`) is plotted.

**Arguments**

| Argument | Type | Default | Description |
|---|---|---|---|
| *(all common args)* | — | — | See common arguments table |
| `--source-ix IX` | int | *(from config, 99)* | Source x grid index |
| `--source-iz IZ` | int | *(from config, 20)* | Source z grid index |
| `--freq-min HZ` | float | `50e6` | FC_low — start frequency [Hz] |
| `--freq-max HZ` | float | `200e6` | FC_high — end frequency [Hz] |
| `--nf N` | int | `50` | Number of frequencies; df=(freq_max-freq_min)/(nf-1) |
| `--freq-step HZ` | float | *(None)* | Explicit step [Hz] — overrides `--nf` when set |
| `--clip V` | float | `2.5e-3` | Blackman-Harris amplitude clip (MATLAB: clip) |
| `--clip1 V` | float | `1.0e-2` | Secondary amplitude clip (MATLAB: clip1) |
| `--caxis V` | float | `10.0` | Fixed colour limits +-V [V/m] (MATLAB: `caxis([-10 10])`); set 0 for auto |
| `--no-tiff` | flag | off | Save PNG only (skip TIFF output) |

**MATLAB-equivalent command (default parameters match inp_GPRmodel1.m exactly)**

```bash
# Recommended — MATLAB defaults: nf=50, FC_low=50MHz, FC_high=200MHz, df~3.06MHz
python examples/run_forward_wavefield.py --stag2 --ncpus 15

# Explicit parameters (same result)
python examples/run_forward_wavefield.py --stag2 --ncpus 15 \
    --freq-min 50e6 --freq-max 200e6 --nf 50

# Override frequency step (MATLAB-style 1 MHz step, 151 frequencies)
python examples/run_forward_wavefield.py --stag2 --ncpus 15 \
    --freq-min 50e6 --freq-max 200e6 --freq-step 1e6

# PNG only, no TIFF
python examples/run_forward_wavefield.py --stag2 --no-tiff

# Override source position
python examples/run_forward_wavefield.py --stag2 --source-ix 99 --source-iz 20
```

**Output:**
- `results/forward/wavefield/wavefield_real.png`
- `results/forward/wavefield/wavefield_real.tiff` (unless `--no-tiff`)

---

## 3. `run_forward_bscan.py`

FDFD forward modelling — zero-offset GPR B-scan (radargram).
Steps a single source across the surface, extracts the vertical
column at each source position, stacks into a depth-vs-position image.

**Arguments**

| Argument | Type | Default | Description |
|---|---|---|---|
| *(all common args)* | — | — | See common arguments table |
| `--src-step N` | int | `2` | Grid-cell step between source positions |
| `--src-iz IZ` | int | `20` | Source depth index (absolute, includes PML) |

**Examples**

```bash
# Default — stag1, 1 CPU, src-iz=20 (top acquisition row)
python examples/run_forward_bscan.py

# Recommended: stag2, parallel
python examples/run_forward_bscan.py --stag2 --ncpus 15

# Finer source spacing
python examples/run_forward_bscan.py --stag2 --ncpus 15 --src-step 1

# Custom config
python examples/run_forward_bscan.py --config input/input_forward.yaml --stag2

# Save impedance matrix for first source
python examples/run_forward_bscan.py --impedance-matrix
```

**Output:** `results/forward/bscan/bscan.npz`, `bscan.png`

---

## 4. `run_forward_cmp.py`

FDFD forward modelling — Common Mid-Point (CMP) gather (time-domain).

For each half-offset, solves FDFD at all frequencies, extracts the single
receiver response `Ez[src_iz, rec_ix]`, and IFFTs to produce a
**time vs offset** CMP gather.  Requires `nf × n_offsets` solves.

Default frequencies: GPRFM 10 discrete (50–200 MHz).
Override with `--fc-low`/`--fc-high`/`--nf` for linspace sweep.

**Arguments**

| Argument | Type | Default | Description |
|---|---|---|---|
| *(all common args)* | — | — | See common arguments table |
| `--n-offsets N` | int | `25` | Number of half-offset positions |
| `--offset-min M` | float | `0.1` | Minimum half-offset [m] |
| `--offset-max M` | float | `2.0` | Maximum half-offset [m] |
| `--mid-ix IX` | int | `nx//2` | Mid-point x grid index |
| `--src-iz IZ` | int | `20` | Source/receiver depth index |
| `--fc-low HZ` | float | *(GPRFM)* | Override start frequency → linspace sweep |
| `--fc-high HZ` | float | *(GPRFM)* | Override end frequency → linspace sweep |
| `--nf N` | int | *(GPRFM)* | Override frequency count → linspace sweep |
| `--tmax-ns NS` | float | `150.0` | Max display time [ns] |
| `--pad N` | int | `0` | Zero-samples per side of Hermitian spectrum |
| `--wiggle-gain G` | float | `1.5` | Wiggle amplitude scale |

**Examples**

```bash
# Default: GPRFM 10 discrete freqs (50,60,...,200 MHz)
python examples/run_forward_cmp.py --stag2 --ncpus 10

# Custom offset range
python examples/run_forward_cmp.py --stag2 --ncpus 10 \
    --n-offsets 30 --offset-min 0.05 --offset-max 2.0

# Override to linspace sweep (50 freqs)
python examples/run_forward_cmp.py --stag2 --ncpus 15 \
    --fc-low 50e6 --fc-high 200e6 --nf 50

# Show more time
python examples/run_forward_cmp.py --stag2 --ncpus 10 --tmax-ns 200
```

**Output:** `results/forward/cmp/cmp_<tags>.npz`, `cmp_<tags>.png`, `cmp_wiggle_<tags>.png`

---

## 5. `run_forward_shotgather.py`

FDFD forward modelling — time-domain shot gather.

For a **single source**, solves FDFD at all frequencies (default: GPRFM 10
discrete), extracts the full surface receiver line, and applies the GPRFM
Hermitian IFFT to produce a shot gather.

Requires only `nf` FDFD solves (e.g. 10 for GPRFM defaults) — much faster
than CMP.

X-axis: signed offset from source [m] (negative = left, positive = right).
Y-axis: two-way travel time [ns].

**Arguments**

| Argument | Type | Default | Description |
|---|---|---|---|
| *(all common args)* | — | — | See common arguments table |
| `--src-ix IX` | int | `nx//2` | Source x grid index |
| `--src-iz IZ` | int | `20` | Source depth grid index |
| `--rec-iz IZ` | int | *(= src-iz)* | Receiver depth grid index |
| `--rec-step N` | int | `1` | Step between receivers [cells] |
| `--rec-ix-start IX` | int | `npx+1` | First receiver x index |
| `--rec-ix-end IX` | int | `nx-npx-1` | Last receiver x index |
| `--fc-low HZ` | float | *(GPRFM)* | Override start frequency → linspace |
| `--fc-high HZ` | float | *(GPRFM)* | Override end frequency → linspace |
| `--nf N` | int | *(GPRFM)* | Override frequency count → linspace |
| `--tmax-ns NS` | float | `150.0` | Max display time [ns] |
| `--pad N` | int | `0` | Zero-samples per side of Hermitian spectrum |
| `--wiggle-gain G` | float | `1.5` | Wiggle amplitude scale |

**Examples**

```bash
# Default: GPRFM 10 freqs, source at nx//2, all surface receivers
python examples/run_forward_shotgather.py --stag2 --ncpus 10

# Custom source position
python examples/run_forward_shotgather.py --stag2 --ncpus 10 --src-ix 60

# Sparser receivers (every 2nd cell) and deeper source
python examples/run_forward_shotgather.py --stag2 --ncpus 10 \
    --src-ix 99 --src-iz 20 --rec-step 2

# Linspace frequency override
python examples/run_forward_shotgather.py --stag2 --ncpus 15 \
    --fc-low 50e6 --fc-high 200e6 --nf 50
```

**Output:** `results/forward/shotgather/sg_<tags>.npz`, `sg_<tags>.png`, `sg_wiggle_<tags>.png`

---

## 6. `run_forward_shotgather_center.py`

FDFD forward modelling — time-domain shot gather with source fixed at grid centre.

Convenience wrapper around `run_forward_shotgather.py` that hard-codes
`src-ix = nx//2`. Accepts all the same CLI arguments.

**Arguments**

| Argument | Type | Default | Description |
|---|---|---|---|
| *(all common args)* | — | — | See common arguments table |
| *(all shotgather args)* | — | — | See section 5 (`run_forward_shotgather.py`) |

**Examples**

```bash
# Default: centred source, GPRFM 10 freqs
python examples/run_forward_shotgather_center.py --stag2 --ncpus 10

# Linspace frequency override
python examples/run_forward_shotgather_center.py --stag2 --ncpus 15 \
    --fc-low 50e6 --fc-high 200e6 --nf 50
```

**Output:** `results/forward/shotgather/sg_<tags>.npz`, `sg_<tags>.png`, `sg_wiggle_<tags>.png`

---

## 7. `run_inversion_example.py`

Full-Waveform Inversion (FWI) — MATLAB RFDFWI.m style.

Generates synthetic observed data at **GPRFM 10 discrete frequencies** for all
sources in the 4-sided acquisition, then runs the adjoint-state FWI loop with
Tikhonov regularisation and Armijo line search.

**Data shape:** `d_obs[n_src, n_freq, n_rec]` complex.

**Arguments**

| Argument | Type | Default | Description |
|---|---|---|---|
| *(all common args)* | — | — | See common arguments table |
| `--true-epsr V` | float | *(from config model)* | Homogeneous true εᵣ (overrides YAML model) |
| `--true-sigma V` | float | *(from config model)* | Homogeneous true σ [S/m] |
| `--init-smooth PX` | float | `6.0` | Gaussian smooth (pixels) true→initial model |
| `--init-epsr V` | float | *(None)* | Homogeneous initial εᵣ (overrides --init-smooth) |
| `--init-sigma V` | float | *(None)* | Homogeneous initial σ [S/m] |
| `--max-iter N` | int | *(from config)* | Override inversion max iterations |
| `--lambda-sigma V` | float | *(from config)* | Override Tikhonov LAMBDA_1 (MATLAB: 2e-4) |
| `--step-init V` | float | *(auto)* | Override initial step size (≤0 = auto-scale) |
| `--step-epsr S` | float | *(from config)* | Override `step_init_epsr` — initial step for εᵣ (e.g. 0.5) |
| `--step-sigma S` | float | *(from config)* | Override `step_init_sigma` — initial step for σ (e.g. 5e-4) |
| `--fc-low HZ` | float | *(None)* | Switch to linspace sweep: start frequency |
| `--fc-high HZ` | float | *(None)* | Linspace sweep: end frequency |
| `--nf N` | int | *(None)* | Linspace sweep: number of frequencies |

**Examples**

```bash
# Default: mkl_two_cross true model, GPRFM 10 freqs, 4-sided acq, stag2
python examples/run_inversion_example.py --stag2 --ncpus 15

# Homogeneous true model (quick test)
python examples/run_inversion_example.py --stag2 --ncpus 4 \
    --true-epsr 9.0 --true-sigma 0.01 \
    --init-epsr 4.0 --init-sigma 3e-3 --max-iter 10

# Custom lambda_sigma (50-iteration run uses config max_iter default)
python examples/run_inversion_example.py --stag2 --ncpus 15 \
    --lambda-sigma 2e-4

# Override per-parameter initial step sizes
python examples/run_inversion_example.py --stag2 --ncpus 15 \
    --step-epsr 0.5 --step-sigma 5e-4

# Linspace frequency sweep instead of GPRFM 10
python examples/run_inversion_example.py --stag2 --ncpus 15 \
    --fc-low 50e6 --fc-high 200e6 --nf 10
```

**Output:**
- `results/inversion/obs/d_obs.npz` — observed data array
- `results/inversion/models/true_model_epsr.png` — true εᵣ model
- `results/inversion/models/true_model_sigma.png` — true σ model
- `results/inversion/models/#0initial_model_epsr.png` — initial εᵣ model
- `results/inversion/models/#0initial_model_sigma.png` — initial σ model
- `results/inversion/models/000Output_model_at_iteration=<NNNN>_epsr.png` — per-iteration εᵣ
- `results/inversion/models/000Output_model_at_iteration=<NNNN>_sigma.png` — per-iteration σ
- `results/inversion/gradient/02grad_iteration_<nw>_iter=<NNNN>_epsr.png` — per-iteration εᵣ gradient
- `results/inversion/gradient/02grad_iteration_<nw>_iter=<NNNN>_sigma.png` — per-iteration σ gradient
- `results/inversion/hessian/03HESS_iteration_<nw>_<NNNN>_epsr.png` — per-iteration εᵣ pseudo-Hessian
- `results/inversion/hessian/03HESS_iteration_<nw>_<NNNN>_sigma.png` — per-iteration σ pseudo-Hessian
- `results/inversion/search_direction/04Hgrad_iteration_<NNNN>_epsr.png` — per-iteration εᵣ search direction
- `results/inversion/search_direction/04Hgrad_iteration_<NNNN>_sigma.png` — per-iteration σ search direction
- `results/inversion/tikhonov/05Tikhonov_iter=<NNNN>_sigma.png` — per-iteration Tikhonov term
- `results/inversion/models/final_result.npz` — final + initial + true arrays
- `results/inversion/models/#Output_FINAL_Converged_Models_at_iteration=<NNNN>_epsr.png` — final εᵣ
- `results/inversion/models/#Output_FINAL_Converged_Models_at_iteration=<NNNN>_sigma.png` — final σ
- `results/inversion/misfit/#Output_L2_ratio_curve.png` — L2 convergence plot
- `results/inversion/logs/run_log.txt` — full run metadata

Where `<nw>` = number of GPRFM frequencies (10 by default) and `<NNNN>` = 4-digit zero-padded iteration number.

---

## Stencil selection summary

| Flag | Scheme | Reference |
|---|---|---|
| `--stag1` *(default)* | Parallel staggered grid | Hustedt et al. (2004), *GJI* |
| `--stag2` | New staggered grid (recommended) | Layek & Sengupta (2024) |

Both use a **9-point CFS-PML** formulation.
PML parameters are set in the `pml` section of the YAML config:

```yaml
pml:
  npx: 10
  npz: 10
  a0_cfs: 9.0e8    # MATLAB sig_max (Chen et al. 2013)
```

---

## Model types (`model.type` in YAML)

| Type | Description |
|---|---|
| `mkl_two_cross` | **Exact MATLAB replica** (create_models_mkl.m): bg epsr=4, sigma=3mS/m; cross1 dry sand epsr=1; cross2 dry clay epsr=8 |
| `two_cross` | Parametric cross model (center_x/z, half_len in metres) |
| `homogeneous` | Uniform epsr + sigma |
| `layered` | Horizontal layer stack |
| `file` | Load from `inputmodel/model_epsr.npy` + `model_sigma.npy` |

---

## Frequency sweep parameters (`freq_sweep` in YAML)

Matches `inp_GPRmodel1.m` exactly:

```yaml
freq_sweep:
  fc_low:  50e6      # MATLAB: FC_low  = 50 MHz
  fc_high: 200e6     # MATLAB: FC_high = 200 MHz
  nf:      50        # MATLAB: nf=50  ->  df = 150/49 ~ 3.061 MHz
  clip:    2.5e-3    # MATLAB: clip  (Blackman-Harris window)
  clip1:   1.0e-2    # MATLAB: clip1
  tmax_td: 150e-9    # MATLAB: TmaxTD = 150 ns
```

---

## Acquisition geometry (`acquisition.mode` in YAML)

### 4-sided (MATLAB default)

```yaml
acquisition:
  mode: 4sided
  npml: 10
  nrec_per_side: 40   # -> 162 receivers total (41+41+40+40)
  nsrc_per_side: 20   # ->  82 sources  total (21+21+20+20)
```

Boundary (with npml=10, dh=0.05m):
- `ix/iz = 20` (= `min(x) + 10*dh` in MATLAB = 0.55 m)
- `ix/iz = 179` (= `max(x) - 10*dh` in MATLAB = 8.50 m)

### Surface line

```yaml
receivers:
  mode: line
  iz: 20
  ix_start: 20
  ix_end: 179
```

---

## Complete MATLAB-matched workflow

```bash
conda activate rfdfwimkl
cd D:\rfdfwi

# Step 1 — build exact MATLAB model (mkl_two_cross, 200x200, dh=0.05m)
python examples/run_build_model.py

# Step 2 — multi-frequency wavefield (nf=50, 50-200 MHz, stag2)
python examples/run_forward_wavefield.py --stag2 --ncpus 15

# Step 3 — zero-offset B-scan
python examples/run_forward_bscan.py --stag2 --ncpus 15

# Step 4 — FWI inversion (4-sided acquisition, 82 sources)
python examples/run_inversion_example.py --stag2 --ncpus 15
```

---

## Quick-reference one-liners

```bash
# Build model (MATLAB mkl_two_cross replica)
python examples/run_build_model.py

# Wavefield: MATLAB defaults (nf=50, 50-200 MHz, stag2)
python examples/run_forward_wavefield.py --stag2 --ncpus 15

# Wavefield: explicit 1 MHz step (151 frequencies)
python examples/run_forward_wavefield.py --stag2 --ncpus 15 --freq-step 1e6

# B-scan: stag2, 15 CPUs, step=1
python examples/run_forward_bscan.py --stag2 --ncpus 15 --src-step 1

# CMP gather, stag2
python examples/run_forward_cmp.py --stag2 --n-offsets 30

# Shot gather, centred source
python examples/run_forward_shotgather_center.py --stag2 --ncpus 10

# FWI inversion, stag2
python examples/run_inversion_example.py --stag2 --ncpus 15
```

---

## 8. `validate_python.py`

Automated solver correctness checks. Runs two tests to verify that the forward
solver and FWI engine are working correctly. Run this after installation and after
any code changes.

**Arguments:** None (no CLI arguments; configuration is hard-coded internally).

```bash
python examples/validate_python.py
```

**Tests performed:**

| Test | What it checks | Pass criterion |
|------|---------------|----------------|
| Forward residual | Assembles A and solves A u = b; checks ‖A u − b‖ | Residual < 1e-10 |
| Misfit decrease | Runs 3 FWI iterations on a small homogeneous model | L2[iter 1] < L2[iter 0] |

**Exit codes:**

| Code | Meaning |
|------|---------|
| `0` | All tests passed |
| `1` | One or more tests failed |

**Example output (passing):**

```
Test 1: Forward residual ... PASS (residual=3.41e-13)
Test 2: Misfit decrease  ... PASS (L2[0]=1.23e+00, L2[1]=9.87e-01)
All tests passed.
```

**Example output (failing):**

```
Test 1: Forward residual ... FAIL (residual=2.45e-06 >= 1e-10)
FAILED: 1 test(s) failed.
```

---

## Appendix A: Parameter Combinations & Validity

### Mutually exclusive flags

| Flag pair | Rule | Effect if both set |
|-----------|------|-------------------|
| `--stag1` and `--stag2` | Mutually exclusive | Error: only one stencil allowed |
| `--nf N` and `--freq-step HZ` | Mutually exclusive | `--freq-step` takes precedence over `--nf` |
| `--init-smooth PX` and `--init-epsr V` | Mutually exclusive for FWI | `--init-epsr` overrides smooth |
| `--step-epsr S` and config `step_init_epsr` | CLI takes precedence | Overrides YAML value |
| `--step-sigma S` and config `step_init_sigma` | CLI takes precedence | Overrides YAML value |

### Flag dependencies

| Flag | Requires | Notes |
|------|----------|-------|
| `--freq-step HZ` | `--freq-min` and `--freq-max` also set | Computes nf = (max-min)/step + 1 |
| `--fc-high HZ` (inversion) | `--fc-low HZ` | Activates linspace sweep |
| `--nf N` (inversion) | `--fc-low` and `--fc-high` | All three required for linspace |
| `--init-epsr V` | `--init-sigma V` recommended | Sets homogeneous initial model |
| `--true-epsr V` | `--true-sigma V` recommended | Sets homogeneous true model |
| `--use-gpu` | CuPy installed | Falls back to CPU if CuPy absent |

### Valid ranges for key parameters

| Parameter | Valid range | Notes |
|-----------|-------------|-------|
| `--ncpus N` | 1 to num_cpu_cores | Higher values give diminishing returns |
| `--source-ix IX` | npx to nx-npx-1 (default: 10 to 189) | Must be inside PML boundary |
| `--source-iz IZ` | npz to nz-npz-1 (default: 10 to 189) | Must be inside PML boundary |
| `--freq-min HZ` | > 0 | Typically 50e6 to 200e6 for GPR |
| `--freq-max HZ` | > freq-min | Typically up to 200e6 for GPR |
| `--nf N` | >= 2 | nf=1 gives no frequency sweep |
| `--n-offsets N` | >= 1 | CMP: number of source-receiver offset pairs |
| `--lambda-sigma V` | >= 0 | 0 disables Tikhonov regularisation |
| `--step-init V` | any float; <= 0 for auto | Auto = L2 / ‖gradient‖² |
| `--step-epsr S` | > 0 | Max Δεᵣ per iteration; config default 0.5 (7% of εᵣ range) |
| `--step-sigma S` | > 0 | Max Δσ per iteration; config default 5e-4 (2.5% of σ range) |
| `--max-iter N` | >= 1 | Convergence may trigger early stop |
| `--patience N` | >= 1 | Early-stop window; config default 8 |
| `--warmup N` | >= 0 | Iterations before early-stop activates; config default 5 |

### Frequency mode selection

| Scripts | Default frequency mode | Override to linspace |
|---------|----------------------|---------------------|
| `run_forward_cmp.py` | GPRFM 10 discrete | `--fc-low --fc-high --nf` |
| `run_forward_shotgather.py` | GPRFM 10 discrete | `--fc-low --fc-high --nf` |
| `run_forward_shotgather_center.py` | GPRFM 10 discrete | `--fc-low --fc-high --nf` |
| `run_inversion_example.py` | GPRFM 10 discrete | `--fc-low --fc-high --nf` |
| `run_forward_wavefield.py` | Linspace (nf=50) | `--freq-step` overrides nf |
| `run_forward_bscan.py` | From YAML config | `--freq-min --freq-max --nf` |

GPRFM 10 discrete frequencies: 50, 60, 70, 80, 90, 100, 125, 150, 175, 200 MHz.
