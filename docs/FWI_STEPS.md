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

# RFDFWI — Inversion Algorithm: Step-by-Step

> **Reference implementation:** `scripts/inversion_fwi.py` · `examples/run_inversion_example.py`
> **Repository:** https://github.com/mklayek/rfdfwi

---

## Overview

The inversion solves the electromagnetic FWI problem in the 2-D frequency domain.
Two model parameters are recovered simultaneously:

| Parameter | Symbol | Units | Bounds |
|-----------|--------|-------|--------|
| Relative permittivity | εᵣ | dimensionless | [0.5, 10.0] |
| Electrical conductivity | σ | S/m | [0.05e-3, 15.0e-3] |

**Data:** `d_obs` — shape `(n_src, n_freq, n_rec)`, complex
**Frequencies:** GPRFM 10 discrete — 50, 60, 70, 80, 90, 100, 125, 150, 175, 200 MHz
**Acquisition:** 4-sided boundary — 82 sources, 162 receivers

---

## Pre-Iteration Setup

### Step 0A — True model
Build the `mkl_two_cross` synthetic model (εᵣ, σ) on the 200×200 FDFD grid
(180 interior + 10-cell CFS-PML on each side, dh = 0.05 m).

### Step 0B — Observed data (synthetic)
Run FDFD forward modelling for all sources × frequencies × receivers:

```
d_obs[s, f, r]  =  u(x_r)      for source s, frequency f, receiver r
```

Shape: `(82 × 10 × 162)`, stored as complex.

### Step 0C — Initial model
Gaussian-smoothed true model (σ_blur = 6 px, matching MATLAB `imgaussfilt(eps_model, 6)`) —
or homogeneous from YAML config. The 6-pixel blur matches the MATLAB RFDFWI.m default
and can be overridden via `--init-smooth PX` on the CLI.

---

## Iteration Loop  —  k = 1, 2, …, max_iter

---

### Step 1 — Forward Solve  *(per frequency ω, per source s)*

Assemble the 9-point CFS-PML Helmholtz impedance matrix **A** (stag2 by default):

```
A · u  =  b_src        where  b_src = −(jωμ₀ / dh²)  at source node k_src
```

Solved via direct sparse LU (`scipy.sparse.linalg.spsolve`).
Extract receiver wavefield: `d_calc[s, f, r] = u[k_rec_r]`

---

### Step 2 — Data Residual and L2 Misfit

```
res        =  d_calc[s,f,:]  −  d_obs[s,f,:]        (length: n_rec, complex)
L2_total  +=  0.5 × Σᵣ |res[r]|²
```

---

### Step 3 — Adjoint Solve  *(same A; adjoint RHS injected at receiver nodes)*

```
b_adj[k_rec_r]  =  res[r] / dh²        (for each receiver r)
A^H · λ         =  b_adj               (conjugate-transpose system)
```

The adjoint wavefield λ propagates the data residuals back through the medium.

---

### Step 4 — Gradient Accumulation  *(MATLAB `ass_grad_TEMKLnew.m`)*

```
grad_εᵣ  +=  Re( ω²  ·  conj(u)  ·  λ )
grad_σ   +=  Re( jω  ·  conj(u)  ·  λ )
```

*Repeat Steps 1–4 for all sources and all frequencies.*

---

### Step 5 — Pseudo-Hessian Diagonal  *(Born approximation, used for preconditioning)*

```
H_εᵣ  +=  ω⁴  ·  |u|²
H_σ   +=  ω²  ·  |u|²
```

Accumulates over all source-frequency pairs. Acts as an approximate curvature estimate.

---

### Step 6 — Convergence Check

```
ratio  =  L2 / L2[iter=1]
if ratio ≤ conv_ratio (default: 1×10⁻²)  →  STOP (converged)
```

A `conv_ratio` of 1% (1×10⁻²) is a realistic target for synthetic data with the
mkl_two_cross model. The earlier value of 5×10⁻⁵ was too strict and rarely
triggered in practice. An early-stop mechanism (patience=8 iterations after
warmup_iters=5) provides an additional exit condition.

---

### Step 7 — Tikhonov Laplacian Regularisation  *(MATLAB `Tikhonov_grad_TE.m`)*

Penalises spatial roughness of the model:

```
σ_r      =  σ  ·  (β_σ / σ₀)
tikh_σ   =  Λ₁  ·  β_σ  ·  ∇²(σ_r) / dh²        (Λ₁ = 2×10⁻⁴, σ₀ = 5.6×10⁻³ S/m)
tikh_εᵣ  =  Λ₂  ·  β_εᵣ ·  ∇²(εᵣ)  / dh²        (Λ₂ ≈ 0, inactive by default)

g_εᵣ  =  grad_εᵣ  +  tikh_εᵣ
g_σ   =  grad_σ   +  tikh_σ
```

---

### Step 8 — Hessian Preconditioning + Search Direction

Divide the regularised gradient by the pseudo-Hessian diagonal to rescale to O(1).
Operations are restricted to interior cells `[npml:nz-npml, npml:nx-npml]` to
avoid PML-amplified values biasing the Hessian maximum:

```
H_max    =  max over interior cells of ( max(H_εᵣ), max(H_σ) )
H_ε      =  1×10⁻⁵ · H_max              (regularisation floor — avoids division by zero)

d_εᵣ  =  − g_εᵣ  /  (H_εᵣ  +  H_ε)
d_σ   =  − g_σ   /  (H_σ   +  H_ε)

# PML border cells zeroed in direction arrays:
d_εᵣ[:npml, :] = d_εᵣ[-npml:, :] = d_εᵣ[:, :npml] = d_εᵣ[:, -npml:] = 0
d_σ [:npml, :] = d_σ [-npml:, :] = d_σ [:, :npml] = d_σ [:, -npml:] = 0
```

> **Critical note:** Without Hessian preconditioning, raw gradient magnitudes (~10²⁰)
> cause the auto-scaled step to be ~10⁻⁴², making the model update effectively zero
> and producing a completely flat misfit curve.

> **PML masking note:** PML cells have wavefield amplitudes 46–100× larger than
> interior cells. Without masking, the Hessian maximum is dominated by PML values,
> compressing the interior search direction toward zero and suppressing updates in
> the physical domain.

---

### Step 9 — Initial Step Size

Separate initial steps are used for each parameter because the εᵣ range (~7) and
σ range (~0.02 S/m) differ by a factor of ~350:

```
step_e  =  step_init_epsr   (default: 0.5  — ~7% of εᵣ range 7.0)
step_s  =  step_init_sigma  (default: 5×10⁻⁴ — ~2.5% of σ range 0.02 S/m)
```

Both steps are set from `input/input_inversion.yaml` and can be overridden via
`--step-epsr` and `--step-sigma` CLI flags. Both are halved together during Armijo
backtracking.

---

### Step 10 — Armijo Backtracking Line Search  *(max `STEPMAX = 12` trials)*

**Normalised search directions** (interior cells only, independent per parameter):

```
d_εᵣ_norm  =  d_εᵣ  /  max(|d_εᵣ_interior|)
d_σ_norm   =  d_σ   /  max(|d_σ_interior|)
```

Independent per-parameter normalisation ensures both εᵣ and σ receive equal-scale
updates, preventing the σ direction from being suppressed relative to εᵣ due to
the ω⁴ vs ω² Hessian scaling difference. The maximum is computed over interior
cells only to exclude PML-amplified values.

**Trial model update:**

```
εᵣ_try  =  εᵣ  +  step_e · d_εᵣ_norm
σ_try   =  σ   +  step_s · d_σ_norm
```

**Simplified decrease condition:**

```
L2_try  =  forward_misfit( εᵣ_try,  σ_try )
Accept  :  L2_try  <  L2_current           (simple strict decrease)
Reject  :  step_e /= SCALEFAC (= 2.0),  step_s /= SCALEFAC,  retry
```

If no trial satisfies the condition within `STEPMAX=12` halvings, the smallest
step is accepted. This simpler condition avoids the over-lenient Armijo threshold
that previously allowed steps that marginally increased the misfit.

`STEPMAX = 12` allows backtracking through 12 halvings (step range
`[step, step/4096]`), sufficient to handle cases where the initial step is too
large.

---

### Step 11 — Model Update

```
εᵣ  ←  εᵣ  +  step_e · d_εᵣ_norm
σ   ←  σ   +  step_s · d_σ_norm
```

---

### Step 12 — Hard Box Bounds

```
εᵣ  clipped to  [0.5,  10.0]
σ   clipped to  [0.05e-3,  15.0e-3]  S/m
```

Bounds are set to bracket the mkl_two_cross true model values with margin:
- Lower εᵣ bound (0.5) lies below the dry-sand cross1 value (true εᵣ = 1.0)
- Upper εᵣ bound (10.0) lies above the clay cross2 value (true εᵣ = 8.0)
- Lower σ bound (0.05e-3) lies below the dry-sand σ = 0.1e-3 S/m
- Upper σ bound (15.0e-3) lies above the clay σ = 10e-3 S/m

---

### Step 13 — Per-Iteration Outputs

Filenames follow MATLAB RFDFWI.m conventions. `NNNN` = 4-digit zero-padded iteration
number; `<nw>` = number of GPRFM frequencies (10 by default).

| Output | Path | Notes |
|--------|------|-------|
| Recovered εᵣ image | `models/000Output_model_at_iteration=NNNN_epsr.png` | seismic cmap, vmin=0, vmax=10 |
| Recovered σ image | `models/000Output_model_at_iteration=NNNN_sigma.png` | seismic cmap, vmin=0, vmax=10 |
| Gradient εᵣ | `gradient/02grad_iteration_<nw>_iter=NNNN_epsr.png` | diverging seismic |
| Gradient σ | `gradient/02grad_iteration_<nw>_iter=NNNN_sigma.png` | diverging seismic |
| Pseudo-Hessian εᵣ | `hessian/03HESS_iteration_<nw>_NNNN_epsr.png` | inferno |
| Pseudo-Hessian σ | `hessian/03HESS_iteration_<nw>_NNNN_sigma.png` | inferno |
| Search direction εᵣ | `search_direction/04Hgrad_iteration_NNNN_epsr.png` | diverging seismic |
| Search direction σ | `search_direction/04Hgrad_iteration_NNNN_sigma.png` | diverging seismic |
| Tikhonov term | `tikhonov/05Tikhonov_iter=NNNN_sigma.png` | |
| Live misfit curve | `misfit/#Output_L2_ratio_curve.png` | **overwritten each iteration** |
| Progress log | `logs/progress_YYYYMMDD_HHMMSS.txt` | timestamped per line |

Pre-loop outputs (written once before the iteration starts):

| Output | Path | Notes |
|--------|------|-------|
| True εᵣ model | `models/true_model_epsr.png` | seismic cmap |
| True σ model | `models/true_model_sigma.png` | seismic cmap |
| Initial εᵣ model | `models/#0initial_model_epsr.png` | seismic cmap |
| Initial σ model | `models/#0initial_model_sigma.png` | seismic cmap |

---

### Step 14 — Step Size Expansion for Next Iteration

```
step  ←  min( step × SCALEFAC,  step_max_cap )
```

---

## Termination

Loop exits when any of the following conditions is met:

- `ratio ≤ conv_ratio` (L2/L2[0] ≤ 1×10⁻², converged), **or**
- `k = max_iter` (iteration limit reached, default 50), **or**
- Early-stop: after the first `warmup_iters` (default 5) iterations, if the misfit
  has not decreased for `patience` (default 8) consecutive iterations

Final εᵣ and σ saved to `models/final_result.npz`,
`models/#Output_FINAL_Converged_Models_at_iteration=NNNN_epsr.png`,
`models/#Output_FINAL_Converged_Models_at_iteration=NNNN_sigma.png`.

---

## Key Parameters  (`input/input_inversion.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_iter` | 50 | Maximum iterations |
| `patience` | 8 | Early-stop: consecutive non-decreasing iters before stopping (after warmup) |
| `warmup_iters` | 5 | Iterations to skip before early-stop activates |
| `conv_ratio` | 1×10⁻² | L2 ratio convergence threshold (1% of initial L2) |
| `step_init_epsr` | 0.5 | Max Δεᵣ per iteration (~7% of εᵣ range 7.0) |
| `step_init_sigma` | 5×10⁻⁴ | Max Δσ per iteration (~2.5% of σ range 0.02 S/m) |
| `lambda_sigma` (Λ₁) | 2×10⁻⁴ | Tikhonov weight for σ |
| `lambda_epsr` (Λ₂) | 0 | Tikhonov weight for εᵣ (inactive by default) |
| `sigma0` | 5.6×10⁻³ S/m | Reference conductivity for Tikhonov |
| `stepmax` | 12 | Max Armijo backtracking trials (covers 2^12 step range) |
| `scale_fac` | 2.0 | Step reduction factor |
| `c1_wolfe` | 1×10⁻⁴ | Armijo C1 constant (used only for logging; decrease condition is simple L2 decrease) |
| `H_eps_frac` | 1×10⁻⁵ | Hessian regularisation floor fraction |
| `bounds.epsr_min` | 0.5 | Lower εᵣ bound (covers dry-sand cross1, true εᵣ=1.0) |
| `bounds.epsr_max` | 10.0 | Upper εᵣ bound (covers clay cross2, true εᵣ=8.0) |
| `bounds.sigma_min` | 0.05e-3 | Lower σ bound [S/m] (below dry-sand σ=0.1e-3) |
| `bounds.sigma_max` | 15.0e-3 | Upper σ bound [S/m] (above clay σ=10e-3) |

---

## References

- Lavoué et al. (2014)
- Layek & Sengupta (2019, 2021, 2024)
- Köhn, D., De Nil, D. and Rabbel, W. (2017). Tutorial: GERMAINE.
  DOI: 10.13140/RG.2.2.29354.03523
- Layek, M. K., & Sengupta, P. (2024). Multi-parameter imaging by finite difference
  frequency domain full waveform inversion of GPR data. *Pure and Applied Geophysics*,
  181, 2107–2130. https://doi.org/10.1007/s00024-024-03520-1
