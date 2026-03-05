# MATLAB to Python Mapping (RFDFWI)

Maps the MATLAB RFDFWI codebase (`inp_GPRmodel*.m`, `create_models_mkl.m`,
`PML_9p_CFS_stag_para.m`, `GPRFM.m`, `RFDFWI.m`) to the Python implementation.

---

## Entry points

| MATLAB | Python | Notes |
|--------|--------|--------|
| `RFDFWI.m` | `examples/run_inversion_example.py` | Main FWI workflow |
| `GPRFM.m` | `scripts/forward_fdfd.py` + `examples/run_forward_*.py` | Forward solver entry |
| `inp_GPRmodel.m`, `inp_GPRmodel1.m` | `input/*.yaml` | Config → YAML |
| `create_models/create_models_mkl.m` | `create_models/build_models.py` | Model + acquisition geometry |
| Forward wavefield plot | `examples/run_forward_wavefield.py` | Multi-freq, seismic colourmap |
| B-scan plot | `examples/run_forward_bscan.py` | Zero-offset radargram |

---

## Grid and domain

| MATLAB | Python | Value |
|--------|--------|-------|
| `model1.nx = 180` | `domain.nx = 200` | 180 interior + 10 PML each side |
| `model1.ny = 180` | `domain.nz = 200` | Same (y→z depth axis) |
| `model1.dh = 0.05` | `domain.dx = domain.dz = 0.05` | Grid spacing [m] |
| `model1.x = dh*(1:nx)` | Python ix: `x = ix * dh` | Python 0-indexed, MATLAB 1-indexed |
| `npml = 10` | `pml.npx = pml.npz = 10` | PML width [cells] |
| Index mapping | `ix_python = npml + i_matlab - 1` | e.g. MATLAB i=11 → Python ix=20 |

---

## CFS-PML parameters

| MATLAB | Python | Value |
|--------|--------|-------|
| `model1.a0_cfs = 9e8` | `pml.a0_cfs = 9.0e8` | sigma_max (Chen et al. 2013) |
| `model1.kappax_max = 5.0` | hardcoded in `forward_fdfd.py` | kappa scaling |
| `model1.alphax_max = 0.2` | hardcoded | alpha scaling |
| `model1.alphay_max = pi/16 * 1e8` | hardcoded | |
| `model1.eps01 = 1` | `omega` (not `omega*EPS0`) in stretch factor | Critical: MATLAB normalises by 1, not eps0 |
| `model1.free = 0` | no free surface | Always 0 |

**Key formula (PML stretch factor):**
```
# MATLAB: sx = kappax + sigma_x * 1i / (alpha_x + omega * eps01)
# Python:  sx = kappax + sigma_x * 1j / (alpha_x + omega)   [eps01=1]
```

---

## Frequency parameters

| MATLAB (`inp_GPRmodel1.m`) | Python | Value |
|---|---|---|
| `model1.f = 50e6` | `freq_hz: 50e6` | Single-freq forward |
| `model1.FC_low = 50e6` | `freq_sweep.fc_low: 50e6` | Sweep start |
| `model1.FC_high = 200e6` | `freq_sweep.fc_high: 200e6` | Sweep end |
| `model1.nf = 50` | `freq_sweep.nf: 50` | Number of frequencies |
| `model1.df = (FC_high-FC_low)/(nf-1)` | derived: `df = 150e6/49 ≈ 3.061 MHz` | Step (linspace) |
| `model1.clip = 2.5e-3` | `freq_sweep.clip: 2.5e-3` | Blackman-Harris clip |
| `model1.clip1 = 1e-2` | `freq_sweep.clip1: 1.0e-2` | |
| `model1.TmaxTD = 150e-9` | `freq_sweep.tmax_td: 150e-9` | Max recording time [s] |
| `model1.sig_max = 9e8` | `pml.a0_cfs: 9.0e8` | Same parameter |

**Frequency generation:**
```python
# MATLAB: freqs = FC_low : df : FC_high  (linspace equivalent)
freqs = np.linspace(fc_low, fc_high, nf)   # 50 frequencies
```

---

## Model (create_models_mkl.m)

| MATLAB | Python | Value |
|--------|--------|-------|
| `epsilonr = 4.0*eps0*ones(ny,nx)` | `epsr_bg: 4.0` | Background relative permittivity |
| `sigma = 3.0e-3*ones(ny,nx)` | `sigma_bg: 3.0e-3` | Background conductivity [S/m] |
| `model1.sig0 = 5.6e-3` | inversion normalisation only | NOT background sigma |
| `epsilonr(50:80,60:70) = 1.0*eps0` | `iz[59:90], ix[69:80]` | Cross 1 vert bar (dry sand) |
| `epsilonr(60:70,50:80) = 1.0*eps0` | `iz[69:80], ix[59:90]` | Cross 1 horiz bar |
| `epsilonr(100:130,110:120) = 8.0*eps0` | `iz[109:140], ix[119:130]` | Cross 2 vert bar (dry clay) |
| `epsilonr(110:120,100:130) = 8.0*eps0` | `iz[119:130], ix[109:140]` | Cross 2 horiz bar |
| `sigma(50:80,60:70) = 0.10e-3` | `cross1_sigma: 0.1e-3` | |
| `sigma(100:130,110:120) = 10e-3` | `cross2_sigma: 10.0e-3` | |

Python model type: `mkl_two_cross` — use in YAML as `model.type: mkl_two_cross`.

Index mapping (MATLAB 1-indexed interior → Python 0-indexed extended):
```
python_iz = npml + matlab_row - 1   (npml=10: matlab 50 → python 59)
python_ix = npml + matlab_col - 1
```

---

## Acquisition geometry (create_models_mkl.m)

| MATLAB | Python | Value |
|--------|--------|-------|
| `nrec = 40` (per side) | `nrec_per_side: 40` | Receivers per side |
| `nsrc = 20` (per side) | `nsrc_per_side: 20` | Sources per side |
| `xrec_1 = min(x)+10*dh = 0.55m` | `ix_lo = 20` | Acquisition boundary left/top |
| `xrec_2 = max(x)-10*dh = 8.50m` | `ix_hi = 179` | Acquisition boundary right/bottom |
| Top receivers: 41 pts (nrec+1) | `linspace(20,179,41)` | |
| Left/right receivers: 40 pts (nrec) | `linspace(20,179,40)` | |
| Top/bottom sources: 21 pts (nsrc+1) | `linspace(20,179,21)` | |
| Left/right sources: 20 pts (nsrc) | `linspace(20,179,20)` | |
| Total: 82 sources, 162 receivers | `mode: 4sided` | In `acquisition` section |
| `model1.ACQMY = 1` | `acquisition.mode: 4sided` | 4-sided activation flag |

---

## Inversion bounds

| MATLAB | Python | Value |
|--------|--------|-------|
| `model1.epsr_low = 3.8*eps0` | `bounds.epsr_min: 3.8` | |
| `model1.epsr_high = 33.2*eps0` | `bounds.epsr_max: 33.2` | |
| `model1.sigr_low = 0.08e-3` | `bounds.sigma_min: 0.08e-3` | |
| `model1.sigr_high = 20.2e-3` | `bounds.sigma_max: 20.2e-3` | |
| `model1.LAMBDA_1 = 2e-4` | `regularization.alpha: 1e-4` | Tikhonov weight |
| `model1.STAG = 2` | `--stag2` CLI flag | Staggered grid variant |

---

## Source amplitude

```python
# MATLAB RHS_TE1.m:  f = -(omega * mu0 * j) / dh^2
# Python forward_fdfd.py:
src_amp = -(omega * MU0 * 1j) / dh**2
```
