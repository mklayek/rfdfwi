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
Build 2D GPR models: permittivity (epsr) and conductivity (sigma) on a grid.
"""
from pathlib import Path
from typing import Any

import numpy as np


def homogeneous_model(
    nx: int,
    nz: int,
    epsr: float = 9.0,
    sigma: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Constant relative permittivity and conductivity."""
    epsr_grid = np.full((nz, nx), epsr, dtype=np.float64)
    sigma_grid = np.full((nz, nx), sigma, dtype=np.float64)
    return epsr_grid, sigma_grid


def two_cross_model(
    nx: int,
    nz: int,
    dx: float = 0.01,
    dz: float = 0.01,
    epsr_bg: float = 4.5,
    sigma_bg: float = 0.0045,
    cross1_center_x: float = 3.0,
    cross1_center_z: float = 3.0,
    cross1_epsr: float = 2.0,
    cross1_sigma: float = 0.002,
    cross2_center_x: float = 6.0,
    cross2_center_z: float = 6.0,
    cross2_epsr: float = 7.0,
    cross2_sigma: float = 0.007,
    cross_half_len: float = 0.75,
    cross_width: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-cross model (create_models_mkl-style): 9 m x 9 m domain, background 4.5,
    upper-left cross at (3,3) m with value 2, lower-right cross at (6,6) m with value 7.
    Sigma in S/m (stored); display in mS/m = sigma * 1000.
    """
    epsr_grid = np.full((nz, nx), epsr_bg, dtype=np.float64)
    sigma_grid = np.full((nz, nx), sigma_bg, dtype=np.float64)

    def draw_cross(cx_m: float, cz_m: float, e_val: float, s_val: float) -> None:
        ix_c = int(round(cx_m / dx))
        iz_c = int(round(cz_m / dz))
        half_cells = max(1, int(round(cross_half_len / min(dx, dz))))
        width_cells = max(1, int(round(cross_width / min(dx, dz))))
        ixl = max(0, ix_c - half_cells)
        ixr = min(nx, ix_c + half_cells + 1)
        izt = max(0, iz_c - half_cells)
        izb = min(nz, iz_c + half_cells + 1)
        ixl_w = max(0, ix_c - width_cells)
        ixr_w = min(nx, ix_c + width_cells + 1)
        izt_w = max(0, iz_c - width_cells)
        izb_w = min(nz, iz_c + width_cells + 1)
        epsr_grid[izt:izb, ixl_w:ixr_w] = e_val
        sigma_grid[izt:izb, ixl_w:ixr_w] = s_val
        epsr_grid[izt_w:izb_w, ixl:ixr] = e_val
        sigma_grid[izt_w:izb_w, ixl:ixr] = s_val

    draw_cross(cross1_center_x, cross1_center_z, cross1_epsr, cross1_sigma)
    draw_cross(cross2_center_x, cross2_center_z, cross2_epsr, cross2_sigma)
    return epsr_grid, sigma_grid


def layered_model(
    nx: int,
    nz: int,
    interfaces: list[float],
    epsr_values: list[float],
    sigma_values: list[float],
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal layers. interfaces[i] is depth (m) of top of layer i (interface 0 = 0)."""
    z = np.arange(nz) * dz
    epsr_grid = np.zeros((nz, nx))
    sigma_grid = np.zeros((nz, nx))
    for i in range(len(interfaces) - 1):
        mask = (z >= interfaces[i]) & (z < interfaces[i + 1])
        epsr_grid[mask, :] = epsr_values[i]
        sigma_grid[mask, :] = sigma_values[i]
    mask = z >= interfaces[-1]
    epsr_grid[mask, :] = epsr_values[-1]
    sigma_grid[mask, :] = sigma_values[-1]
    return epsr_grid, sigma_grid


def load_model_from_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load epsr and sigma from .npz (keys 'epsr', 'sigma') or two .npy files."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return data["epsr"], data["sigma"]
    epsr = np.load(path.with_name(path.stem + "_epsr.npy"))
    sigma = np.load(path.with_name(path.stem + "_sigma.npy"))
    return epsr, sigma


def mkl_two_cross_model(
    nx: int,
    nz: int,
    npml: int = 10,
    epsr_bg: float = 4.0,
    sigma_bg: float = 3.0e-3,
    cross1_epsr: float = 1.0,
    cross1_sigma: float = 0.1e-3,
    cross2_epsr: float = 8.0,
    cross2_sigma: float = 10.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact replica of MATLAB create_models_mkl.m two-cross model.

    MATLAB uses a 180x180 interior grid (1-indexed).  Python uses a
    (nz, nx) = (200, 200) extended grid (0-indexed, includes PML).
    Mapping: python_idx = matlab_idx + npml - 1.

    Cross 1 (dry sand, epsr=1.0, sigma=0.1e-3):
      Vertical bar : epsilonr(50:80, 60:70)  -> iz[59:90],  ix[69:80]
      Horizontal   : epsilonr(60:70, 50:80)  -> iz[69:80],  ix[59:90]

    Cross 2 (dry clay, epsr=8.0, sigma=10e-3):
      Vertical bar : epsilonr(100:130, 110:120) -> iz[109:140], ix[119:130]
      Horizontal   : epsilonr(110:120, 100:130) -> iz[119:130], ix[109:140]
    """
    offset = npml - 1          # MATLAB 1-indexed → Python 0-indexed

    epsr_grid  = np.full((nz, nx), epsr_bg,  dtype=np.float64)
    sigma_grid = np.full((nz, nx), sigma_bg, dtype=np.float64)

    def _fill(iz0, iz1, ix0, ix1, e, s):
        epsr_grid[iz0:iz1,  ix0:ix1]  = e
        sigma_grid[iz0:iz1, ix0:ix1]  = s

    # Cross 1 — dry sand
    _fill(offset+50, offset+81, offset+60, offset+71, cross1_epsr, cross1_sigma)  # vertical
    _fill(offset+60, offset+71, offset+50, offset+81, cross1_epsr, cross1_sigma)  # horizontal

    # Cross 2 — dry clay
    _fill(offset+100, offset+131, offset+110, offset+121, cross2_epsr, cross2_sigma)  # vertical
    _fill(offset+110, offset+121, offset+100, offset+131, cross2_epsr, cross2_sigma)  # horizontal

    return epsr_grid, sigma_grid


def build_4sided_acquisition(
    npml: int = 10,
    nrec_per_side: int = 40,
    nsrc_per_side: int = 20,
) -> tuple[list[dict], list[dict]]:
    """
    Generate MATLAB-style 4-sided acquisition geometry.

    MATLAB create_models_mkl.m places sources/receivers at:
      xrec_1 = min(x) + 10*dh = dh + npml*dh = (npml+1)*dh  -> Python ix = npml
      xrec_2 = max(x) - 10*dh = 180*dh - npml*dh            -> Python ix = 180-1+npml-1 = npml+179-1

    With npml=10: ix_min=20, ix_max=179 (same for iz).

    Returns
    -------
    sources   : list of {ix, iz}   (82 total: 21+21+20+20)
    receivers : list of {ix, iz}  (162 total: 41+41+40+40)
    """
    # MATLAB: model1.x = dh*(1:180),  xrec_1 = min(x)+10*dh = 11*dh -> i_matlab=11
    #         xrec_2 = max(x)-10*dh = 170*dh -> i_matlab=170
    # Python ix = npml + i_matlab - 1   (0-indexed, includes PML)
    #   ix_lo = npml + 11 - 1 = 10+11-1 = 20   (= 2*npml for npml=10)
    #   ix_hi = npml + 170 - 1 = 179
    ix_lo = npml + npml           # = 20 for npml=10
    ix_hi = npml + 170 - 1        # = 179 for npml=10  (180-10=170 cells from right)
    iz_lo = ix_lo
    iz_hi = ix_hi

    def _linspace_int(a, b, n):
        return list(np.round(np.linspace(a, b, n)).astype(int))

    # ---- Receivers (nrec_per_side per side) ----
    ix_top  = _linspace_int(ix_lo, ix_hi, nrec_per_side + 1)   # 41
    ix_bot  = ix_top                                             # 41
    iz_left = _linspace_int(iz_lo, iz_hi, nrec_per_side)        # 40
    iz_right = iz_left                                           # 40

    receivers = (
        [{"ix": int(x), "iz": iz_lo}  for x in ix_top]   +
        [{"ix": int(x), "iz": iz_hi}  for x in ix_bot]   +
        [{"ix": ix_lo,  "iz": int(z)} for z in iz_left]  +
        [{"ix": ix_hi,  "iz": int(z)} for z in iz_right]
    )

    # ---- Sources (nsrc_per_side per side) ----
    ix_src_top  = _linspace_int(ix_lo, ix_hi, nsrc_per_side + 1)  # 21
    ix_src_bot  = ix_src_top                                        # 21
    iz_src_side = _linspace_int(iz_lo, iz_hi, nsrc_per_side)       # 20

    sources = (
        [{"ix": int(x), "iz": iz_lo}  for x in ix_src_top]   +
        [{"ix": int(x), "iz": iz_hi}  for x in ix_src_bot]   +
        [{"ix": ix_lo,  "iz": int(z)} for z in iz_src_side]  +
        [{"ix": ix_hi,  "iz": int(z)} for z in iz_src_side]
    )

    return sources, receivers


def build_model_from_config(config: dict[str, Any], nx: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (epsr, sigma) from config model section."""
    model_cfg = config.get("model", {})
    mtype = model_cfg.get("type", "homogeneous")
    if mtype == "homogeneous":
        return homogeneous_model(
            nx, nz,
            epsr=float(model_cfg.get("epsr", 9.0)),
            sigma=float(model_cfg.get("sigma", 0.01)),
        )
    if mtype == "file":
        return load_model_from_file(model_cfg.get("path", "model.npz"))
    if mtype == "layered":
        interfaces = model_cfg.get("interfaces", [0.0, 0.5])
        epsr_vals = model_cfg.get("epsr", [4.0, 9.0])
        sigma_vals = model_cfg.get("sigma", [0.001, 0.01])
        dz = config.get("domain", config).get("dz", 0.01)
        return layered_model(nx, nz, interfaces, epsr_vals, sigma_vals, dz)
    if mtype == "two_cross":
        domain = config.get("domain", config)
        dx = float(domain.get("dx", 0.01))
        dz = float(domain.get("dz", 0.01))
        return two_cross_model(
            nx, nz, dx=dx, dz=dz,
            epsr_bg=float(model_cfg.get("epsr_bg", 4.5)),
            sigma_bg=float(model_cfg.get("sigma_bg", 0.0045)),
            cross1_center_x=float(model_cfg.get("cross1_center_x", 3.0)),
            cross1_center_z=float(model_cfg.get("cross1_center_z", 3.0)),
            cross1_epsr=float(model_cfg.get("cross1_epsr", 2.0)),
            cross1_sigma=float(model_cfg.get("cross1_sigma", 0.002)),
            cross2_center_x=float(model_cfg.get("cross2_center_x", 6.0)),
            cross2_center_z=float(model_cfg.get("cross2_center_z", 6.0)),
            cross2_epsr=float(model_cfg.get("cross2_epsr", 7.0)),
            cross2_sigma=float(model_cfg.get("cross2_sigma", 0.007)),
            cross_half_len=float(model_cfg.get("cross_half_len", 0.75)),
            cross_width=float(model_cfg.get("cross_width", 0.15)),
        )
    if mtype == "mkl_two_cross":
        pml = config.get("pml", {})
        npml = int(pml.get("npx", 10)) if isinstance(pml, dict) else 10
        return mkl_two_cross_model(
            nx, nz, npml=npml,
            epsr_bg=float(model_cfg.get("epsr_bg", 4.0)),
            sigma_bg=float(model_cfg.get("sigma_bg", 3.0e-3)),
            cross1_epsr=float(model_cfg.get("cross1_epsr", 1.0)),
            cross1_sigma=float(model_cfg.get("cross1_sigma", 0.1e-3)),
            cross2_epsr=float(model_cfg.get("cross2_epsr", 8.0)),
            cross2_sigma=float(model_cfg.get("cross2_sigma", 10.0e-3)),
        )
    return homogeneous_model(nx, nz)
