# Installation & Usage Guide

## Requirements

| Requirement | Minimum version |
|-------------|----------------|
| Python      | 3.11            |
| NumPy       | 1.20            |
| SciPy       | 1.7             |
| Matplotlib  | 3.4             |
| PyYAML      | 5.4             |

Optional (GPU acceleration, future):

| Package | Purpose |
|---------|---------|
| CuPy    | GPU-accelerated sparse solver (falls back to CPU if absent) |

---

## Installation (Anaconda Prompt)

### 1 — Clone the repository

```
git clone <repository-url>
cd rfdfwi
```

### 2 — Create the conda environment

```
conda create -n rfdfwimkl python=3.11 -y
```

### 3 — Activate the environment

```
conda activate rfdfwimkl
```

Your prompt will change to `(rfdfwimkl) ...` confirming activation.

### 4 — Install dependencies

```
pip install -r requirements.txt
```

### 5 — Verify installation

```
python -c "import numpy, scipy, matplotlib, yaml; print('All dependencies OK')"
```

---

### Optional: GPU Acceleration

CuPy (CUDA) is listed as an optional dependency. The `--use-gpu` flag is accepted
by all scripts but the sparse direct solver (`scipy.sparse.linalg.spsolve`) does not
currently use GPU. Install for future compatibility:

```bash
conda install -c conda-forge cupy cuda-version=11.8   # CUDA 11.x
conda install -c conda-forge cupy cuda-version=12.0   # CUDA 12.x
```

When CuPy is absent the code falls back to the CPU solver automatically.

---

### Deactivate when done

```
conda deactivate
```

### Re-activate in a new session

```
conda activate rfdfwimkl
cd D:\rfdfwi
```

---

## Project Layout

```
rfdfwi/
├── create_models/
│   └── build_models.py         Model builders (homogeneous, layered, two-cross, file)
├── scripts/
│   ├── forward_fdfd.py         2D FDFD Helmholtz forward solver (PML, multi-source)
│   ├── inversion_fwi.py        Full Waveform Inversion (adjoint-state, Tikhonov)
│   ├── config_loader.py        YAML configuration helpers
│   ├── plot_bscan.py           B-scan visualisation
│   ├── plot_cmp.py             CMP gather visualisation
│   ├── plot_utils.py           Shared figure utilities + draw_pml_boundary
│   └── _cli.py                 Shared argparse helpers (add_common_args)
├── examples/
│   ├── run_build_model.py            Step 1: build and save 2-D model
│   ├── run_forward_bscan.py          Step 2a: B-scan forward modelling
│   ├── run_forward_cmp.py            Step 2b: CMP gather forward modelling
│   ├── run_forward_wavefield.py      Step 2c: single-source wavefield plot
│   ├── run_forward_shotgather.py     Step 2d: single-source shot gather
│   ├── run_forward_shotgather_center.py  Step 2e: shot gather, centred source
│   ├── run_inversion_example.py      Step 3: Full Waveform Inversion
│   └── validate_python.py            Automated solver checks
├── input/                      YAML configuration files
├── inputmodel/                 Generated model arrays and figures (auto-created)
├── results/
│   ├── forward/
│   │   ├── bscan/              bscan_traces.npz, bscan.png
│   │   ├── cmp/                cmp_traces.npz, cmp.png
│   │   └── wavefield/          wavefield_real.png, wavefield_real.tiff
│   └── inversion/
│       ├── obs/                d_obs.npz (synthetic observed data)
│       ├── models/             iter_N_epsr.png, iter_N_sigma.png, final_result.npz
│       ├── gradient/           iter_N_grad_epsr.png, grad_sigma.png
│       ├── hessian/            iter_N_hess_epsr.png, hess_sigma.png
│       ├── search_direction/   iter_N_dir_epsr.png, dir_sigma.png
│       ├── tikhonov/           iter_N_tikh_epsr.png, tikh_sigma.png
│       ├── misfit/             misfit_curve.png
│       └── logs/               run_log.txt
├── obs/                        Observed field data (optional, real-data inversion)
├── docs/                       Extended documentation
├── requirements.txt
└── INSTALLATION.md             This file
```

---

## Workflow

### Step 1 — Build the model

```
python examples/run_build_model.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config FILE` | `input/input_forward.yaml` | YAML config file |

```
python examples/run_build_model.py --config input/input_forward.yaml
```

**Output:** `inputmodel/model_epsr.npy`, `model_sigma.npy`, `model_eps_sig.png`

---

### Step 2a — B-scan forward modelling

```
python examples/run_forward_bscan.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config FILE` | auto-detected | YAML config file |
| `--results-dir DIR` | `results/forward/bscan` | Override output directory |
| `--ncpus N` | `1` | Parallel CPU workers |
| `--impedance-matrix` | off | Save Helmholtz matrix to output dir |
| `--use-gpu` | off | GPU acceleration (requires CuPy) |
| `--grid-style STYLE` | from config | `stag1` or `stag2` |
| `-v / --verbose` | off | Extra diagnostic output |

```
# Default
python examples/run_forward_bscan.py

# 4 CPUs, save impedance matrix
python examples/run_forward_bscan.py --ncpus 4 --impedance-matrix

# Custom config, stag2 discretisation
python examples/run_forward_bscan.py --config input/input_forward.yaml --grid-style stag2 -v
```

**Output:** `results/forward/bscan/bscan.npz`, `results/forward/bscan/bscan.png`

---

### Step 2b — CMP gather forward modelling

```
python examples/run_forward_cmp.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config FILE` | `input/input_forward.yaml` | YAML config file |
| `--results-dir DIR` | `results/forward/cmp` | Override output directory |
| `--ncpus N` | `1` | Parallel CPU workers |
| `--impedance-matrix` | off | Save Helmholtz matrix |
| `--n-offsets N` | `25` | Number of offset positions |
| `--offset-min M` | `0.1` | Minimum half-offset [m] |
| `--offset-max M` | `1.0` | Maximum half-offset [m] |
| `--grid-style STYLE` | from config | `stag1` or `stag2` |
| `-v / --verbose` | off | Extra diagnostic output |

```
# Default
python examples/run_forward_cmp.py

# Custom offset range, 4 CPUs
python examples/run_forward_cmp.py --ncpus 4 --n-offsets 50 --offset-min 0.05 --offset-max 2.0

# With impedance matrix and verbose
python examples/run_forward_cmp.py --impedance-matrix --ncpus 2 -v
```

**Output:** `results/forward/cmp/cmp_<tags>.npz`, `results/forward/cmp/cmp_<tags>.png`, `results/forward/cmp/cmp_wiggle_<tags>.png`

---

### Step 2c — Wavefield plot (MATLAB-style)

```
python examples/run_forward_wavefield.py
```

Solves the forward problem for a single source and saves the real part of Ez
as a high-resolution seismic-colourmap figure matching the MATLAB output style
(`Wavefield_9pCFSPML_para_stag_FM.tiff`).

| Flag | Default | Description |
|------|---------|-------------|
| `--config FILE` | auto-detected | YAML config file |
| `--results-dir DIR` | `results/forward/wavefield` | Override output directory |
| `--source-ix IX` | from config | Source x grid index |
| `--source-iz IZ` | from config | Source z grid index |
| `--impedance-matrix` | off | Save Helmholtz matrix |
| `--no-tiff` | off | Skip TIFF output (PNG only) |
| `--use-gpu` | off | GPU acceleration |
| `--grid-style STYLE` | from config | `stag1` or `stag2` |
| `-v / --verbose` | off | Print Ez amplitude info |

```
# Default (PNG + TIFF)
python examples/run_forward_wavefield.py

# PNG only, stag1, verbose
python examples/run_forward_wavefield.py --no-tiff --grid-style stag1 -v

# Custom source position
python examples/run_forward_wavefield.py --source-ix 45 --source-iz 2

# Full options
python examples/run_forward_wavefield.py \
    --config input/input_forward.yaml \
    --grid-style stag1 \
    --source-ix 45 \
    --source-iz 2 \
    --impedance-matrix \
    -v
```

**Output:**
```
results/forward/wavefield/wavefield_real.png
results/forward/wavefield/wavefield_real.tiff
```

Figure style:
- seismic colourmap, symmetric clim (+/- max|Re(Ez)|)
- White dashed PML inner-boundary with 'x' markers
- White star at source position
- Axes: Distance [m] (x), Depth [m] (y, increasing downward)
- Colourbar: Re(Ez) [V/m]

---

### Step 3 — Full Waveform Inversion

```
python examples/run_inversion_example.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config FILE` | `input/input_inversion.yaml` | YAML inversion config |
| `--results-dir DIR` | `results/inversion` | Override output base directory |
| `--ncpus N` | `1` | Parallel CPU workers |
| `--impedance-matrix` | off | Save observed-data impedance matrix |
| `--true-epsr V` | `9.0` | True permittivity for synthetic data |
| `--true-sigma V` | `0.01` | True conductivity [S/m] |
| `--use-gpu` | off | GPU acceleration |
| `--grid-style STYLE` | from config | `stag1` or `stag2` |
| `-v / --verbose` | off | Extra diagnostic output |

```
# Default
python examples/run_inversion_example.py

# Custom true model, 4 CPUs, verbose
python examples/run_inversion_example.py --true-epsr 7.5 --true-sigma 0.008 --ncpus 4 -v

# Full options
python examples/run_inversion_example.py \
    --config input/input_inversion.yaml \
    --ncpus 4 \
    --impedance-matrix \
    --true-epsr 9.0 \
    --true-sigma 0.01 \
    --grid-style stag1 \
    -v
```

**Output:**
```
results/inversion/obs/d_obs.npz                  Synthetic observed data [n_src, n_freq, n_rec]
results/inversion/models/iter_N_epsr.png         Recovered epsr at iteration N
results/inversion/models/iter_N_sigma.png        Recovered sigma at iteration N
results/inversion/models/final_result.npz        Final + initial + true model arrays
results/inversion/misfit/misfit_curve.png        L2 convergence curve (log scale)
results/inversion/logs/run_log.txt               Run metadata and full misfit history
```

---

### Validation

```
python examples/validate_python.py
```

Runs two automated checks:
1. **Forward residual** — verifies `||A u - b|| < 1e-10`.
2. **Misfit decrease** — confirms misfit does not increase over 3 FWI iterations.

---

## YAML Configuration Reference

### Forward (`input/input_forward.yaml`)

```yaml
domain:
  nx: 91        # grid points in x
  nz: 91        # grid points in z
  dx: 0.1       # cell size [m]
  dz: 0.1       # cell size [m]
pml:
  npx: 10       # PML cells in x
  npz: 10       # PML cells in z
freq_hz: 900e6
source:
  ix: 45
  iz: 2
receivers:
  mode: line
  iz: 2
  ix_start: 1
  ix_end: 90
model:
  type: two_cross   # homogeneous | file | layered | two_cross
output:
  dir: results/forward
  save_fields: false
  save_traces: true
```

### Inversion (`input/input_inversion.yaml`)

```yaml
forward:
  domain: { nx: 101, nz: 81, dx: 0.01, dz: 0.01 }
  pml:    { npx: 10, npz: 10 }
  freq_hz: 900e6
acquisition:
  sources:
    - { ix: 25, iz: 2 }
    - { ix: 50, iz: 2 }
    - { ix: 75, iz: 2 }
  receivers:
    mode: line
    iz: 2
    ix_start: 1
    ix_end: 100
initial_model:
  type: homogeneous
  epsr: 6.0
  sigma: 0.005
inversion:
  max_iter: 20
  step_type: linesearch   # linesearch | fixed
  step_init: 1.0
  regularization:
    type: tikhonov
    alpha: 1e-6
  bounds:
    epsr_min: 1.0
    epsr_max: 25.0
    sigma_min: 0.0
    sigma_max: 1.0
output:
  dir: results/inversion
  save_every_iter: true
```

---

## Impedance Matrix

The `--impedance-matrix` flag saves the sparse Helmholtz system matrix **A**
in SciPy sparse NPZ format.

```python
import scipy.sparse as sp
A = sp.load_npz("results/forward/bscan/impedance_matrix.npz")
print(A.shape, A.dtype)
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: yaml` | `pip install pyyaml` |
| `ModuleNotFoundError: scipy` | `pip install scipy` |
| `Config not found` | Run from the `rfdfwi/` root directory, or supply `--config` |
| Slow forward runs | Increase `--ncpus` (up to number of sources) |
| Misfit not decreasing | Reduce `step_init` or increase `alpha` in inversion config |
| Memory error on large grids | Reduce `nx`/`nz` or number of sources |
| TIFF not opening in MATLAB | Ensure Matplotlib saved with `format="tiff"` (default behaviour) |
