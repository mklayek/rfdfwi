# =============================================================================
# RFDFWI — Full-Waveform Inversion (FWI) of GPR Data  |  By Mrinal
#
# This code is a Python implementation for Full-Waveform Inversion (FWI)
# of Ground Penetrating Radar (GPR) data. FWI is a geophysical imaging
# technique used to reconstruct subsurface properties (electromagnetic
# permittivity and conductivity) by iteratively comparing modelled and
# observed data.
#
# References:
#   Lavoué et al. (2014); Layek & Sengupta (2019, 2021, & 2024)
#   Köhn, D., De Nil, D. and Rabbel, W. (2017) Tutorial: Introduction to
#   frequency domain modelling and FWI of georadar data with GERMAINE.
#   DOI: 10.13140/RG.2.2.29354.03523
#   ____________________________
#   Layek, M. K., & Sengupta, P. (2024). Multi-parameter imaging by finite
#   difference frequency domain full waveform inversion of GPR data: A guide
#   for sedimentary architecture modeling. Pure and Applied Geophysics, 181,
#   2107–2130. https://doi.org/10.1007/s00024-024-03520-1
#
# Copyright © Mrinal Kanti Layek
# Original MATLAB written during PhD @ 2018–19:
#   Mrinal Kanti Layek, Senior Research Fellow (Geophysics)
#   Department of Geology and Geophysics, IIT Kharagpur – 721302, INDIA
#   layek.mk@gmail.com | https://www.researchgate.net/profile/Mrinal_Layek
#
# Python code written during Postdoc @ March 2026:
#   Dr. Mrinal Kanti Layek — Postdoctoral Researcher | 박사후 연구원
#   Geophysics & AI Lab, Department of Energy & Resources Engineering
#   Chonnam National University, Gwangju, Republic of Korea [61186]
#   지구물리 및 인공지능 연구실, 에너지자원공학과, 전남대학교, 광주광역시 [61186]
#   Email: layek.mk@gmail.com
# =============================================================================
"""
2D FDFD Full Waveform Inversion (FWI).

Matches MATLAB RFDFWI.m + grad_obj_MKLnew.m + ass_grad_TEMKLnew.m:

  - Multi-frequency adjoint-state gradient (GPRFM 10 discrete or custom)
  - Tikhonov Laplacian regularisation  (LAMBDA_1 for sigma, LAMBDA_2 for epsr)
  - Armijo backtracking line search
  - Convergence: ratio = L2 / L2[0] <= conv_ratio  (MATLAB: 5e-5)

Data convention
---------------
d_obs / d_calc : ndarray, shape (n_src, n_freq, n_rec), complex
    Axis 0 — sources, axis 1 — frequencies, axis 2 — receivers.
    Matches MATLAB precobs(ntr, nw, nshots) re-ordered to (nshots, nw, ntr).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import linalg as sp_linalg

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from scripts.forward_fdfd import build_helmholtz_2d, solve_forward

# Physical constants
EPS0: float = 8.854187817e-12   # F/m
MU0:  float = 4e-7 * np.pi      # H/m

# GPRFM 10 discrete frequencies matching MATLAB RFDFWI.m
GPRFM_FREQS_HZ: list[float] = [
    50e6, 60e6, 70e6, 80e6, 90e6, 100e6, 125e6, 150e6, 175e6, 200e6,
]


# ---------------------------------------------------------------------------
# Forward data assembly
# ---------------------------------------------------------------------------

def compute_forward_data(
    epsr:      np.ndarray,
    sigma:     np.ndarray,
    dh:        float,
    npml:      int,
    a0_cfs:    float,
    freqs:     np.ndarray,
    sources:   list[tuple[int, int]],
    receivers: list[tuple[int, int]],
    grid_style: str = "stag1",
    n_workers:  int = 1,
) -> np.ndarray:
    """
    Run FDFD forward at every (source, frequency) pair.

    Source amplitude follows MATLAB RHS_TE1.m:
        amp = -(omega * mu0 * j) / dh^2

    Parameters
    ----------
    epsr, sigma : (nz, nx)  Current model.
    dh          : float     Grid spacing [m].
    npml        : int       PML thickness [cells].
    a0_cfs      : float     CFS-PML sigma_max.
    freqs       : (nf,)     Frequency array [Hz].
    sources     : list of (ix, iz)  Source positions.
    receivers   : list of (ix, iz)  Receiver positions (same for all sources).
    grid_style  : "stag1" or "stag2".
    n_workers   : int  Parallel workers for source solves (per frequency).

    Returns
    -------
    d_calc : (n_src, n_freq, n_rec)  Complex receiver responses.
    """
    nz, nx = epsr.shape
    N      = nx * nz
    n_src  = len(sources)
    n_freq = len(freqs)

    # Pre-extract receiver indices as arrays (once, outside all loops)
    rec_ix  = np.array([r[0] for r in receivers], dtype=np.intp)
    rec_iz  = np.array([r[1] for r in receivers], dtype=np.intp)

    d_calc = np.zeros((n_src, n_freq, len(receivers)), dtype=complex)

    for fi, freq in enumerate(freqs):
        omega   = 2.0 * np.pi * freq
        src_amp = -(omega * MU0 * 1j) / dh ** 2

        A  = build_helmholtz_2d(epsr, sigma, dh, omega, npml, a0_cfs,
                                grid_style=grid_style)
        # Factor A once for all sources at this frequency
        lu = sp_linalg.splu(A.tocsc())

        def _fwd(si: int) -> tuple[int, np.ndarray]:
            ix, iz = sources[si]
            b = np.zeros(N, dtype=np.complex128)
            b[iz * nx + ix] = src_amp
            u_flat = lu.solve(b)
            u      = u_flat.reshape(nz, nx)
            row    = u[rec_iz, rec_ix]          # vectorized
            return si, row

        if n_workers > 1 and n_src > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for si, row in ex.map(_fwd, range(n_src)):
                    d_calc[si, fi, :] = row
        else:
            for si in range(n_src):
                _, row = _fwd(si)
                d_calc[si, fi, :] = row

    return d_calc


# ---------------------------------------------------------------------------
# Adjoint-state gradient
# ---------------------------------------------------------------------------

def compute_gradient(
    epsr:      np.ndarray,
    sigma:     np.ndarray,
    dh:        float,
    npml:      int,
    a0_cfs:    float,
    freqs:     np.ndarray,
    sources:   list[tuple[int, int]],
    receivers: list[tuple[int, int]],
    d_obs:     np.ndarray,
    grid_style: str  = "stag1",
    n_workers:  int  = 1,
    verbose:    bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute the adjoint-state gradient, pseudo-Hessian diagonal, and L2 misfit.

    Matches MATLAB grad_obj_MKLnew.m + ass_grad_TEMKLnew.m:

        For each frequency omega and each source:
            Forward:  A  u   = -(omega*mu0*j)/dh^2  at k_src
            Residual: res    = d_calc - d_obs                (n_rec,)
            Adjoint:  A^H lam = res / dh^2            at k_rec positions
            Gradient (MATLAB ass_grad_TEMKLnew.m):
                grad_epsr  += Re( omega^2 * conj(u) * lam )
                grad_sigma += Re( j*omega * conj(u) * lam )
            Pseudo-Hessian diagonal (Born approximation):
                hess_epsr  += omega^4 * |u|^2
                hess_sigma += omega^2 * |u|^2

    Returns
    -------
    grad_epsr  : (nz, nx)  Raw adjoint gradient w.r.t. epsr.
    grad_sigma : (nz, nx)  Raw adjoint gradient w.r.t. sigma.
    hess_epsr  : (nz, nx)  Pseudo-Hessian diagonal w.r.t. epsr.
    hess_sigma : (nz, nx)  Pseudo-Hessian diagonal w.r.t. sigma.
    d_calc     : (n_src, n_freq, n_rec)  Calculated data.
    L2         : float  Total misfit  0.5 * sum |d_calc - d_obs|^2.
    """
    nz, nx = epsr.shape
    N      = nx * nz
    n_src  = len(sources)
    n_freq = len(freqs)

    # Pre-extract receiver indices as arrays (once, outside all loops)
    rec_ix   = np.array([r[0] for r in receivers], dtype=np.intp)
    rec_iz   = np.array([r[1] for r in receivers], dtype=np.intp)
    rec_flat = rec_iz * nx + rec_ix           # flat indices for adjoint RHS

    grad_epsr  = np.zeros((nz, nx), dtype=np.float64)
    grad_sigma = np.zeros((nz, nx), dtype=np.float64)
    hess_epsr  = np.zeros((nz, nx), dtype=np.float64)
    hess_sigma = np.zeros((nz, nx), dtype=np.float64)
    d_calc     = np.zeros((n_src, n_freq, len(receivers)), dtype=complex)
    L2_total   = 0.0

    for fi, freq in enumerate(freqs):
        omega   = 2.0 * np.pi * freq
        src_amp = -(omega * MU0 * 1j) / dh ** 2

        A = build_helmholtz_2d(epsr, sigma, dh, omega, npml, a0_cfs,
                               grid_style=grid_style)
        # Factor A and A^H once per frequency — critical optimisation
        lu     = sp_linalg.splu(A.tocsc())
        lu_adj = sp_linalg.splu(A.conj().T.tocsc())

        def _process_source(si: int):
            ix, iz = sources[si]

            # --- Forward solve ---
            b_fwd = np.zeros(N, dtype=np.complex128)
            b_fwd[iz * nx + ix] = src_amp
            u_flat = lu.solve(b_fwd)

            # --- Receiver extraction (vectorised) ---
            u  = u_flat.reshape(nz, nx)
            dc = u[rec_iz, rec_ix]
            res = dc - d_obs[si, fi, :]

            # --- Adjoint RHS (vectorised) ---
            b_adj = np.zeros(N, dtype=np.complex128)
            np.add.at(b_adj, rec_flat, res / dh ** 2)
            lam_flat = lu_adj.solve(b_adj)

            return si, dc, res, u_flat, lam_flat

        if n_workers > 1 and n_src > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                src_results = list(ex.map(_process_source, range(n_src)))
        else:
            src_results = [_process_source(si) for si in range(n_src)]

        # --- Accumulate gradients (serial — no race conditions) ---
        for si, dc, res, u_flat, lam_flat in src_results:
            u   = u_flat.reshape(nz, nx)
            lam = lam_flat.reshape(nz, nx)
            d_calc[si, fi, :] = dc
            L2_total += 0.5 * float(np.sum(np.abs(res) ** 2))
            cu = np.conj(u)
            grad_epsr  += np.real(omega ** 2 * cu * lam)
            grad_sigma += np.real(1j * omega  * cu * lam)
            u_sq = np.abs(u) ** 2
            hess_epsr  += (omega ** 4) * u_sq
            hess_sigma += (omega ** 2) * u_sq

        if verbose:
            n_contrib = (fi + 1) * n_src
            ge_rms = float(np.sqrt(np.mean(grad_epsr ** 2))) / n_contrib
            gs_rms = float(np.sqrt(np.mean(grad_sigma ** 2))) / n_contrib
            print(f"    [freq {fi+1:2d}/{n_freq}] {freq/1e6:.1f} MHz — {n_src} src done  grad/src: ε={ge_rms:.3e}  σ={gs_rms:.3e}")

    if verbose:
        print(f"  gradient done — L2={L2_total:.6e}")

    return grad_epsr, grad_sigma, hess_epsr, hess_sigma, d_calc, L2_total


# ---------------------------------------------------------------------------
# Tikhonov regularisation
# ---------------------------------------------------------------------------

def tikhonov_sigma(
    sigma:      np.ndarray,
    dh:         float,
    lambda1:    float,
    beta_sigma: float,
    sigma0:     float,
) -> np.ndarray:
    """
    Tikhonov Laplacian term for sigma (MATLAB Tikhonov_grad_TE.m).

        sigmar = sigma * (beta_sigma / sigma0)
        tikh   = LAMBDA_1 * beta_sigma * Laplacian(sigmar) / dh^2

    Added to grad_sigma before the model update.
    Returns zeros if lambda1 == 0.
    """
    if lambda1 == 0.0:
        return np.zeros_like(sigma)
    sigmar = sigma * (beta_sigma / sigma0)
    lap = np.zeros_like(sigmar)
    lap[1:-1, 1:-1] = (
        sigmar[2:, 1:-1] + sigmar[:-2, 1:-1]
        + sigmar[1:-1, 2:] + sigmar[1:-1, :-2]
        - 4.0 * sigmar[1:-1, 1:-1]
    ) / dh ** 2
    return lambda1 * beta_sigma * lap


def tikhonov_epsr(
    epsr:      np.ndarray,
    dh:        float,
    lambda2:   float,
    beta_epsr: float,
    eps0:      float = EPS0,
) -> np.ndarray:
    """
    Tikhonov Laplacian term for epsr (MATLAB: LAMBDA_2 usually = 0).

    Returns zeros if lambda2 == 0 (default MATLAB behaviour).
    """
    if lambda2 == 0.0:
        return np.zeros_like(epsr)
    epsilonr = epsr * (beta_epsr / eps0)
    lap = np.zeros_like(epsilonr)
    lap[1:-1, 1:-1] = (
        epsilonr[2:, 1:-1] + epsilonr[:-2, 1:-1]
        + epsilonr[1:-1, 2:] + epsilonr[1:-1, :-2]
        - 4.0 * epsilonr[1:-1, 1:-1]
    ) / dh ** 2
    return lambda2 * beta_epsr * lap


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

def apply_bounds(
    epsr:   np.ndarray,
    sigma:  np.ndarray,
    bounds: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Clip model parameters to physical bounds (MATLAB epsr_low/high, sigr_low/high)."""
    epsr  = np.clip(epsr,  bounds.get("epsr_min",  1.0),  bounds.get("epsr_max",  80.0))
    sigma = np.clip(sigma, bounds.get("sigma_min", 0.0),  bounds.get("sigma_max", 1.0))
    return epsr, sigma


# ---------------------------------------------------------------------------
# Main inversion loop
# ---------------------------------------------------------------------------

def run_inversion(
    config:       dict[str, Any],
    d_obs:        np.ndarray,
    epsr_init:    np.ndarray | None = None,
    sigma_init:   np.ndarray | None = None,
    use_gpu:      bool = False,
    n_workers:    int  = 1,
    grid_style:   str  = "stag1",
    iter_callback: Callable[[int, np.ndarray, np.ndarray, dict], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Run the FWI iteration loop matching MATLAB RFDFWI.m.

    Algorithm per iteration:
        1. Compute adjoint-state gradient + L2 misfit
        2. Add Tikhonov regularisation to gradient
        3. Armijo backtracking line search (MATLAB wolfe_TENEW style)
        4. Model update + bounds projection
        5. Convergence check: L2 / L2[0] <= conv_ratio

    Parameters
    ----------
    config       : dict   Full configuration (forward, acquisition, inversion sections).
    d_obs        : (n_src, n_freq, n_rec)  Observed complex data at GPRFM frequencies.
    epsr_init    : (nz, nx) or None  Starting permittivity (built from config if None).
    sigma_init   : (nz, nx) or None  Starting conductivity (built from config if None).
    use_gpu      : bool   Reserved (not yet implemented).
    n_workers    : int    Parallel workers for source solves.
    grid_style   : "stag1" or "stag2".
    iter_callback: callable(iter_num, epsr, sigma, extras_dict) — called after each step.
                   extras_dict keys: L2, grad_epsr, grad_sigma, hess_epsr, hess_sigma,
                   tikh_epsr, tikh_sigma, reg_grad_epsr, reg_grad_sigma,
                   dir_epsr, dir_sigma, step, delta_epsr, delta_sigma.

    Returns
    -------
    epsr_final  : (nz, nx)
    sigma_final : (nz, nx)
    history     : dict with keys 'misfit' (list[float]), 'step' (list[float]).
    """
    from create_models.build_models import build_model_from_config, build_4sided_acquisition

    # ---- Parse config ----
    fwd_cfg = config.get("forward", config)
    domain  = fwd_cfg.get("domain", fwd_cfg)
    nx      = int(domain.get("nx", 200))
    nz      = int(domain.get("nz", 200))
    dh      = float(domain.get("dx", 0.05))
    pml_cfg = fwd_cfg.get("pml", {})
    npml    = int(pml_cfg.get("npx", 10))
    a0_cfs  = float(pml_cfg.get("a0_cfs", 9e8))

    inv_cfg    = config.get("inversion", {})
    max_iter   = int(inv_cfg.get("max_iter", 20))
    bounds     = inv_cfg.get("bounds", {})
    reg        = inv_cfg.get("regularization", {})
    lambda1    = float(reg.get("lambda_sigma", reg.get("alpha", 2e-4)))  # MATLAB LAMBDA_1
    lambda2    = float(reg.get("lambda_epsr",  0.0))                      # MATLAB LAMBDA_2
    beta_sigma = float(reg.get("beta_sigma",   1.0))
    beta_epsr  = float(reg.get("beta_epsr",    1.0))
    sigma0     = float(inv_cfg.get("sigma0",   5.6e-3))   # MATLAB sig0
    step_init  = float(inv_cfg.get("step_init", -1.0))    # <=0 → auto-scale
    step_init_e = float(inv_cfg.get("step_init_epsr",  step_init if step_init > 0 else 0.5))
    step_init_s = float(inv_cfg.get("step_init_sigma", step_init if step_init > 0 else 5e-4))
    stepmax    = int(inv_cfg.get("stepmax",    3))         # MATLAB STEPMAX
    scale_fac  = float(inv_cfg.get("scale_fac", 2.0))     # MATLAB SCALEFAC
    c1_wolfe   = float(inv_cfg.get("c1_wolfe", 1e-4))      # Armijo C1
    conv_ratio = float(inv_cfg.get("conv_ratio", 5e-5))    # MATLAB convergence ratio
    patience     = int(inv_cfg.get("patience",     5))   # early-stop window (iters)
    warmup_iters = int(inv_cfg.get("warmup_iters", 3))   # iters to skip before checking

    # ---- Frequencies ----
    freqs_cfg = inv_cfg.get("freqs_hz", None)
    freqs = np.array(freqs_cfg if freqs_cfg else GPRFM_FREQS_HZ, dtype=float)
    n_freq = len(freqs)

    # ---- Acquisition ----
    acq = config.get("acquisition", {})
    if acq.get("mode") == "4sided":
        npml_acq = int(acq.get("npml", npml))
        nsrc_ps  = int(acq.get("nsrc_per_side", 20))
        nrec_ps  = int(acq.get("nrec_per_side", 40))
        src_list, rec_list = build_4sided_acquisition(npml_acq, nrec_ps, nsrc_ps)
        sources   = [(int(s["ix"]), int(s["iz"])) for s in src_list]
        receivers = [(int(r["ix"]), int(r["iz"])) for r in rec_list]
    else:
        src_list = acq.get("sources", [{"ix": 99, "iz": 20}])
        sources  = [(int(s["ix"]), int(s["iz"])) for s in src_list]
        rec_cfg  = acq.get("receivers", {})
        if isinstance(rec_cfg, dict) and rec_cfg.get("mode") == "line":
            iz_r = int(rec_cfg.get("iz", 20))
            xs   = int(rec_cfg.get("ix_start", 20))
            xe   = int(rec_cfg.get("ix_end", 179))
            receivers = [(ix, iz_r) for ix in range(xs, xe + 1)]
        else:
            receivers = [(int(r["ix"]), int(r["iz"])) for r in (rec_cfg or [])]

    n_src = len(sources)
    n_rec = len(receivers)

    # ---- Initial model ----
    if epsr_init is None or sigma_init is None:
        init_cfg = config.get("initial_model", config.get("model", {}))
        if isinstance(init_cfg, dict) and init_cfg.get("type") == "homogeneous":
            epsr_init  = np.full((nz, nx), float(init_cfg.get("epsr",  4.0)))
            sigma_init = np.full((nz, nx), float(init_cfg.get("sigma", 3e-3)))
        else:
            epsr_init, sigma_init = build_model_from_config(fwd_cfg, nx, nz)

    epsr  = np.array(epsr_init,  dtype=np.float64)
    sigma = np.array(sigma_init, dtype=np.float64)
    epsr, sigma = apply_bounds(epsr, sigma, bounds)

    history: dict[str, list] = {"misfit": [], "step": []}
    L2_first: float | None = None
    step = step_init  # may be overridden by auto-scale below

    print(f"  Grid style : {grid_style}")
    print(f"  Sources    : {n_src}  |  Receivers: {n_rec}")
    print(f"  Frequencies: {n_freq}  ({freqs[0]/1e6:.0f}–{freqs[-1]/1e6:.0f} MHz)")
    print(f"  Max iter   : {max_iter}  |  conv_ratio={conv_ratio:.1e}")
    print(f"  LAMBDA_1   : {lambda1}  |  LAMBDA_2: {lambda2}")
    print(f"  sigma0     : {sigma0:.3e}  |  beta_sigma={beta_sigma}, beta_epsr={beta_epsr}")
    print(f"  STEPMAX    : {stepmax}  |  SCALEFAC={scale_fac}  |  C1={c1_wolfe:.1e}")
    print(f"  Patience   : {patience}  (early stop if no decrease for {patience} iters)")
    print(f"  Warmup     : {warmup_iters}  (first {warmup_iters} iters excluded from early-stop check)")

    # ---- Iteration loop ----
    for it in range(max_iter):
        print(f"\n{'='*60}")
        print(f"[iter {it+1}/{max_iter}] Computing adjoint-state gradient"
              f"  ({n_src} src × {n_freq} freq) ...")

        grad_epsr, grad_sigma, hess_epsr, hess_sigma, d_calc, L2 = compute_gradient(
            epsr, sigma, dh, npml, a0_cfs, freqs,
            sources, receivers, d_obs,
            grid_style=grid_style, n_workers=n_workers, verbose=True,
        )
        history["misfit"].append(L2)

        if L2_first is None:
            L2_first = max(L2, 1e-300)

        ratio = L2 / L2_first
        print(f"  >> L2={L2:.6e}  ratio={ratio:.3e}", end="")

        # ---- Convergence ----
        if ratio <= conv_ratio:
            print("  [CONVERGED — ratio threshold reached]")
            break

        # ---- Early stopping: skip warmup_iters, then check patience-wide window ----
        # Requires at least (warmup_iters + patience) values in history before triggering.
        # E.g. warmup=3, patience=5 → checks only when iter >= 8, using iters 4–8.
        if len(history["misfit"]) >= warmup_iters + patience:
            recent = history["misfit"][-patience:]   # last `patience` values (post-warmup)
            if all(recent[i] >= recent[i - 1] for i in range(1, patience)):
                print(f"\n  [EARLY STOP] L2 did not decrease for {patience} consecutive "
                      f"iterations after {warmup_iters}-iter warmup "
                      f"(iters {len(history['misfit'])-patience+1}–{len(history['misfit'])}: "
                      f"{', '.join(f'{v:.4e}' for v in recent)}).")
                break

        # ---- Tikhonov regularisation (added to gradient) ----
        tikh_s = tikhonov_sigma(sigma, dh, lambda1, beta_sigma, sigma0)
        tikh_e = tikhonov_epsr(epsr,  dh, lambda2, beta_epsr)
        g_sigma = grad_sigma + tikh_s
        g_epsr  = grad_epsr  + tikh_e

        # ---- Hessian preconditioning + search direction (interior cells only) ----
        # PML cells have amplified |u|² → Hessian up to 100× larger than interior.
        # Masking to interior prevents PML from dominating normalization and
        # squashing interior updates.
        int_s = np.s_[npml:nz-npml, npml:nx-npml]   # interior slice
        H_max_e = max(float(hess_epsr[int_s].max()), 1e-300)
        H_max_s = max(float(hess_sigma[int_s].max()), 1e-300)
        eps_H   = 0.01           # 1 % water-level on normalised Hessian

        # Compute direction for interior only; leave PML cells at zero
        dir_epsr  = np.zeros((nz, nx), dtype=np.float64)
        dir_sigma = np.zeros((nz, nx), dtype=np.float64)
        dir_epsr[int_s]  = -g_epsr[int_s]  / (hess_epsr[int_s]  / H_max_e + eps_H)
        dir_sigma[int_s] = -g_sigma[int_s] / (hess_sigma[int_s] / H_max_s + eps_H)

        # Normalize by interior max (PML cells are already zero)
        d_max_e = max(float(np.max(np.abs(dir_epsr[int_s]))), 1e-300)
        d_max_s = max(float(np.max(np.abs(dir_sigma[int_s]))), 1e-300)
        dir_epsr  /= d_max_e
        dir_sigma /= d_max_s
        print(f"\n  Hessian precond [interior {nz-2*npml}×{nx-2*npml}]:"
              f"  H_max_e={H_max_e:.3e}  H_max_s={H_max_s:.3e}  eps_H={eps_H}"
              f"  d_max_e={d_max_e:.3e}  d_max_s={d_max_s:.3e}"
              f"\n  |d_epsr|_int_max={np.max(np.abs(dir_epsr[int_s])):.3e}"
              f"  |d_sigma|_int_max={np.max(np.abs(dir_sigma[int_s])):.3e}")

        # ---- Per-parameter step sizes (reset each iter) ----
        # With unit-max normalized directions, step_e limits max Δεᵣ per iter
        # and step_s limits max Δσ per iter.  Separate sizes are needed because
        # εᵣ range (~7) and σ range (~0.02 S/m) differ by ~350×.
        step_e = step_init_e
        step_s = step_init_s
        print(f"  step_e={step_e:.3e} (max Δεᵣ per iter)"
              f"  step_s={step_s:.3e} (max Δσ per iter)")

        # ---- Armijo backtracking — simple sufficient decrease ----
        # Using simple decrease (L2_try < phi0) avoids scale-mismatch issues
        # between raw gradient (~1e20) and unit-normalised direction (~1.0).
        phi0     = L2
        accepted = False

        for ls in range(stepmax):
            e_try = epsr  + step_e * dir_epsr
            s_try = sigma + step_s * dir_sigma
            e_try, s_try = apply_bounds(e_try, s_try, bounds)

            d_try = compute_forward_data(
                e_try, s_try, dh, npml, a0_cfs, freqs,
                sources, receivers, grid_style=grid_style, n_workers=n_workers,
            )
            L2_try = 0.5 * float(np.sum(np.abs(d_try - d_obs) ** 2))
            ok = L2_try < phi0
            print(f"    [ls {ls+1}/{stepmax}] step_e={step_e:.3e} step_s={step_s:.3e}"
                  f"  L2_try={L2_try:.6e}  Δ={phi0-L2_try:+.3e}"
                  f"  {'ACCEPT' if ok else 'reject'}")

            if ok:
                accepted = True
                break

            step_e /= scale_fac
            step_s /= scale_fac

        if not accepted:
            print(f"  [WARNING] No decrease after {stepmax} trials"
                  f" — accepting last step_e={step_e:.3e}  step_s={step_s:.3e}")

        # ---- Apply update ----
        delta_epsr  = step_e * dir_epsr
        delta_sigma = step_s * dir_sigma
        epsr  = epsr  + delta_epsr
        sigma = sigma + delta_sigma
        epsr, sigma = apply_bounds(epsr, sigma, bounds)
        history["step"].append(step_e)   # store epsr step as reference
        print(f"  update: |Δε|_max={np.max(np.abs(delta_epsr)):.3e}"
              f"  |Δσ|_max={np.max(np.abs(delta_sigma)):.3e}"
              f"  εᵣ=[{epsr[int_s].min():.2f},{epsr[int_s].max():.2f}]"
              f"  σ=[{sigma[int_s].min():.2e},{sigma[int_s].max():.2e}]")

        # ---- Per-iteration callback ----
        if iter_callback is not None:
            extras = {
                "L2":             L2,
                "grad_epsr":      grad_epsr,
                "grad_sigma":     grad_sigma,
                "hess_epsr":      hess_epsr,
                "hess_sigma":     hess_sigma,
                "tikh_epsr":      tikh_e,
                "tikh_sigma":     tikh_s,
                "reg_grad_epsr":  g_epsr,
                "reg_grad_sigma": g_sigma,
                "dir_epsr":       dir_epsr,
                "dir_sigma":      dir_sigma,
                "step":           step_e,
                "delta_epsr":     delta_epsr,
                "delta_sigma":    delta_sigma,
            }
            print(f"  Callback: saving iter {it+1} images ...")
            iter_callback(it + 1, epsr, sigma, extras)

    return epsr, sigma, history
