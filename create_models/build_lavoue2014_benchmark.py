# =============================================================================
# RFDFWI - Full-Waveform Inversion (FWI) of GPR Data  |  By Mrinal
#
# This code is a Python implementation for Full-Waveform Inversion (FWI)
# of Ground Penetrating Radar (GPR) data. FWI is a geophysical imaging
# technique used to reconstruct subsurface properties (electromagnetic
# permittivity and conductivity) by iteratively comparing modelled and
# observed data.
#
# References:
#   Lavoue, F. (2014). 2D full waveform inversion of ground penetrating
#   radar data: towards multiparameter imaging from surface data. PhD thesis,
#   Universite de Grenoble.
#
#   Lavoue et al. (2014). A strategy for multiparameter FWI of on-ground
#   GPR data. Pure Appl. Geophys. Figure 9.
#
# Copyright (c) Mrinal Kanti Layek
# =============================================================================
"""
Lavoue (2014) GPR Benchmark Model — Fluvial deposits, Grenoble.

The geometry is defined by **spline-traced boundary curves** whose control
points were digitised from the published figure (Lavoue et al. 2014, Fig. 9).
Each layer boundary is a smooth cubic-spline function z = f(x) in physical
metres, so the resulting model has naturally curved interfaces — no
staircase artefacts.

Reference:
  Lavoue, F. (2014). PhD thesis, Universite de Grenoble.
  Lavoue et al. (2014). Geophysics, Fig. 9.
"""
from __future__ import annotations

import os
import numpy as np
from scipy.interpolate import CubicSpline


# ---------------------------------------------------------------------------
# Traced boundary control points  (x [m], z [m below ground surface])
# Digitised from Lavoue et al. (2014) Fig. 9.
# ---------------------------------------------------------------------------
# B1 — bottom of silty soil (eps_r 11 → 4)
_B1_X = [0.0, 1.5, 3.0, 5.0, 7.0, 8.5, 10.0]
_B1_Z = [0.30, 0.31, 0.33, 0.36, 0.34, 0.31, 0.28]

# B2 — top of thin sigma=6 mS/m layer
_B2_X = [0.0, 1.5, 3.0, 5.0, 7.0, 8.5, 10.0]
_B2_Z = [2.15, 2.25, 2.40, 2.55, 2.60, 2.50, 2.30]

# B3 — bottom of thin sigma=6 mS/m layer
_B3_X = [0.0, 1.5, 3.0, 5.0, 7.0, 8.5, 10.0]
_B3_Z = [2.55, 2.65, 2.80, 2.95, 3.00, 2.90, 2.70]

# B4 — top of alternating zone (eps_r 4/22)
_B4_X = [0.0, 1.5, 3.0, 5.0, 7.0, 8.5, 10.0]
_B4_Z = [2.85, 2.95, 3.10, 3.25, 3.25, 3.15, 2.95]

# B5 — bottom of alternating / top of attenuating layer
_B5_X = [0.0, 1.5, 3.0, 5.0, 7.0, 8.5, 10.0]
_B5_Z = [3.30, 3.40, 3.50, 3.60, 3.60, 3.50, 3.40]

# B6 — water table / top of saturated clay
_B6_X = [0.0, 1.5, 3.0, 5.0, 7.0, 8.5, 10.0]
_B6_Z = [3.65, 3.75, 3.85, 3.95, 3.90, 3.80, 3.70]

# Clay lenses — top-right (x_start [m], z_top [m], z_bot [m])
_LENSES = [
    # lens 1: shallowest, widest
    {"x_start": 5.5, "z_top": 0.02, "z_bot": 0.15,
     "epsr": 31.0, "sigma": 15.0e-3,
     "wobble_amp": 0.03, "wobble_freq": 2.5},
    # lens 2: middle depth
    {"x_start": 6.0, "z_top": 0.25, "z_bot": 0.40,
     "epsr": 32.0, "sigma": 15.0e-3,
     "wobble_amp": 0.02, "wobble_freq": 3.0},
    # lens 3: deepest, narrowest
    {"x_start": 6.5, "z_top": 0.48, "z_bot": 0.62,
     "epsr": 28.0, "sigma": 12.0e-3,
     "wobble_amp": 0.02, "wobble_freq": 2.0},
]


# ---------------------------------------------------------------------------
# Layer parameter defaults (Lavoue 2014, Fig. 9)
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    epsr_air=1.0,           sigma_air=0.0,
    epsr_silt=11.0,         sigma_silt=0.1e-3,
    epsr_sand=4.0,          sigma_sand=0.1e-3,
    epsr_thin=6.0,          sigma_thin=6.0e-3,
    epsr_sand_lower=4.0,    sigma_sand_lower=1.0e-3,
    epsr_alt_high=22.0,     sigma_alt_high=8.0e-3,
    epsr_alt_low=4.0,       sigma_alt_low=1.0e-3,
    epsr_atten=18.0,        sigma_atten=10.0e-3,
    epsr_clay=32.0,         sigma_clay=20.0e-3,
)


def _spline(x_ctrl, z_ctrl, x_eval):
    """Cubic-spline interpolation of boundary control points."""
    cs = CubicSpline(x_ctrl, z_ctrl, bc_type="natural")
    return cs(x_eval)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_lavoue2014_benchmark_model(
    nx: int,
    nz: int,
    npml: int,
    dh: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the Lavoue (2014) Fig. 9 benchmark GPR model.

    Boundaries are cubic-spline curves traced from the published figure.
    Each grid cell is assigned to the layer whose depth range contains
    the cell centre, giving naturally smooth curved interfaces.

    Parameters
    ----------
    nx, nz : int
        Full grid dimensions (including PML).
    npml : int
        PML cells per side.
    dh : float
        Grid spacing [m].

    Returns
    -------
    epsr_2d, sigma_2d : ndarray, shape (nz, nx)
    """
    p = {**_DEFAULTS, **kwargs}

    nx_int = nx - 2 * npml
    nz_int = nz - 2 * npml
    n_air = int(round(0.5 / dh))          # air cells at top of interior

    # Physical coordinates of interior cell centres [m]
    x_m = (np.arange(nx_int) + 0.5) * dh  # 0..10 m
    z_m = (np.arange(nz_int) + 0.5) * dh - n_air * dh  # -0.5..5.0 m

    # ---- Evaluate spline boundaries on the interior x-grid ----
    b1 = _spline(_B1_X, _B1_Z, x_m)      # silty soil bottom
    b2 = _spline(_B2_X, _B2_Z, x_m)      # thin layer top
    b3 = _spline(_B3_X, _B3_Z, x_m)      # thin layer bottom
    b4 = _spline(_B4_X, _B4_Z, x_m)      # alternating zone top
    b5 = _spline(_B5_X, _B5_Z, x_m)      # attenuating layer top
    b6 = _spline(_B6_X, _B6_Z, x_m)      # water table

    # ---- Allocate interior arrays (air by default) ----
    epsr = np.full((nz_int, nx_int), p["epsr_air"], dtype=np.float64)
    sigma = np.full((nz_int, nx_int), p["sigma_air"], dtype=np.float64)

    # ---- Fill layers using physical coordinates ----
    for ix in range(nx_int):
        for iz in range(nz_int):
            z = z_m[iz]
            if z < 0.0:
                continue                           # air
            elif z < b1[ix]:
                epsr[iz, ix] = p["epsr_silt"]      # silty soil
                sigma[iz, ix] = p["sigma_silt"]
            elif z < b2[ix]:
                epsr[iz, ix] = p["epsr_sand"]      # dry sand (upper)
                sigma[iz, ix] = p["sigma_sand"]
            elif z < b3[ix]:
                epsr[iz, ix] = p["epsr_thin"]      # thin sigma=6 layer
                sigma[iz, ix] = p["sigma_thin"]
            elif z < b4[ix]:
                epsr[iz, ix] = p["epsr_sand_lower"]  # sand (lower)
                sigma[iz, ix] = p["sigma_sand_lower"]
            elif z < b5[ix]:
                # Alternating zone: sublayers every 0.10 m (2 cells)
                rel_z = z - b4[ix]
                sub_idx = int(rel_z / (2.0 * dh))
                if sub_idx % 2 == 0:
                    epsr[iz, ix] = p["epsr_alt_high"]
                    sigma[iz, ix] = p["sigma_alt_high"]
                else:
                    epsr[iz, ix] = p["epsr_alt_low"]
                    sigma[iz, ix] = p["sigma_alt_low"]
            elif z < b6[ix]:
                epsr[iz, ix] = p["epsr_atten"]     # attenuating layer
                sigma[iz, ix] = p["sigma_atten"]
            else:
                epsr[iz, ix] = p["epsr_clay"]      # saturated clay
                sigma[iz, ix] = p["sigma_clay"]

    # ---- Clay lenses (top-right) ----
    for lens in _LENSES:
        for ix in range(nx_int):
            xv = x_m[ix]
            if xv < lens["x_start"]:
                continue
            # gentle sinusoidal wobble on top/bottom
            wb = lens["wobble_amp"] * np.sin(
                2.0 * np.pi * xv * lens["wobble_freq"])
            zt = lens["z_top"] + wb
            zb = lens["z_bot"] + wb
            for iz in range(nz_int):
                z = z_m[iz]
                if zt <= z <= zb:
                    epsr[iz, ix] = lens["epsr"]
                    sigma[iz, ix] = lens["sigma"]

    # ---- Mild smoothing to remove single-cell staircase artefacts ----
    from scipy.ndimage import gaussian_filter
    # Anisotropic: heavier along x (2 cells) than z (0.6 cells)
    iz_ground_int = n_air  # first subsurface row in interior array
    epsr[iz_ground_int:, :] = gaussian_filter(
        epsr[iz_ground_int:, :], sigma=[0.6, 2.0])
    sigma[iz_ground_int:, :] = gaussian_filter(
        sigma[iz_ground_int:, :], sigma=[0.6, 2.0])

    # ---- Embed interior into full grid (with PML) ----
    epsr_full = np.ones((nz, nx), dtype=np.float64)
    sigma_full = np.zeros((nz, nx), dtype=np.float64)
    epsr_full[npml:nz - npml, npml:nx - npml] = epsr
    sigma_full[npml:nz - npml, npml:nx - npml] = sigma

    # PML padding: extend edge values
    # Left
    epsr_full[npml:, :npml] = epsr_full[npml:, npml:npml + 1]
    sigma_full[npml:, :npml] = sigma_full[npml:, npml:npml + 1]
    # Right
    epsr_full[npml:, nx - npml:] = epsr_full[npml:, nx - npml - 1:nx - npml]
    sigma_full[npml:, nx - npml:] = sigma_full[npml:, nx - npml - 1:nx - npml]
    # Bottom
    epsr_full[nz - npml:, :] = epsr_full[nz - npml - 1:nz - npml, :]
    sigma_full[nz - npml:, :] = sigma_full[nz - npml - 1:nz - npml, :]

    return epsr_full, sigma_full


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------
def print_model_summary(
    nx: int, nz: int, npml: int, dh: float,
) -> None:
    """Print a boxed summary of the Lavoue (2014) benchmark model geometry."""
    nz_int = nz - 2 * npml
    nx_int = nx - 2 * npml
    print("=" * 65)
    print("Lavoue (2014) GPR Benchmark -- Fluvial Deposits, Grenoble")
    print("  Ref: Lavoue et al. (2014) Pure Appl. Geophys., Fig. 9")
    print("=" * 65)
    print(f"  Full grid      : {nx} x {nz}  (dh = {dh} m)")
    print(f"  PML            : {npml} cells per side")
    print(f"  Interior grid  : {nx_int} x {nz_int}  "
          f"= {nx_int * dh:.1f} m x {nz_int * dh:.1f} m")
    print(f"  Air            : 10 cells (0.5 m) at top of interior")
    print(f"  Ground surface : iz = {npml + 10} (full-grid index)")
    print("-" * 65)
    print("  Layers (spline-traced from figure, below ground surface):")
    print("    Silty soil (0-0.3m)      : epsr=11.0, sigma=  0.1 mS/m")
    print("    Dry sand (0.3-2.3m)      : epsr= 4.0, sigma=  0.1 mS/m")
    print("    Thin layer (2.3-2.7m)    : epsr= 6.0, sigma=  6.0 mS/m")
    print("    Dry sand (2.7-3.0m)      : epsr= 4.0, sigma=  1.0 mS/m")
    print("    Alternating (3.0-3.5m)   : epsr=4/22, sigma= 1-8 mS/m")
    print("    Attenuating (3.5-3.8m)   : epsr=18.0, sigma= 10.0 mS/m")
    print("    Saturated clay (>3.8m)   : epsr=32.0, sigma= 20.0 mS/m")
    print("-" * 65)
    print("  Clay lenses (top-right):")
    print("    Lens 1 (0.1m, x>5.5m)   : epsr=31.0, sigma= 15.0 mS/m")
    print("    Lens 2 (0.3m, x>6.0m)   : epsr=32.0, sigma= 15.0 mS/m")
    print("    Lens 3 (0.5m, x>6.5m)   : epsr=28.0, sigma= 12.0 mS/m")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _nx, _nz, _npml, _dh = 220, 130, 10, 0.05
    print_model_summary(_nx, _nz, _npml, _dh)

    epsr, sigma = build_lavoue2014_benchmark_model(_nx, _nz, _npml, _dh)
    print(f"  epsr  range: [{epsr.min():.2f}, {epsr.max():.2f}]")
    print(f"  sigma range: [{sigma.min():.5f}, {sigma.max():.5f}] S/m")

    # Interior slice
    _interior_epsr = epsr[_npml:_nz - _npml, _npml:_nx - _npml]
    _interior_sigma = sigma[_npml:_nz - _npml, _npml:_nx - _npml] * 1e3

    # Axis: depth below ground (0 = ground surface)
    _n_air = 10
    _x_ext = [0.0, (_nx - 2 * _npml) * _dh]
    _z_ext = [-_n_air * _dh, (_nz - 2 * _npml - _n_air) * _dh]

    # Output directory
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(_project_root, "results", "benchmark")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    im0 = axes[0].imshow(
        _interior_epsr,
        extent=[_x_ext[0], _x_ext[1], _z_ext[1], _z_ext[0]],
        aspect="auto", cmap="RdBu_r", vmin=1, vmax=32,
        interpolation="bicubic",
    )
    axes[0].set_title("(a) Relative permittivity ($\\varepsilon_r$)",
                       fontsize=13)
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    fig.colorbar(im0, ax=axes[0], label="Relative permittivity",
                 shrink=0.85)

    im1 = axes[1].imshow(
        _interior_sigma,
        extent=[_x_ext[0], _x_ext[1], _z_ext[1], _z_ext[0]],
        aspect="auto", cmap="RdBu_r", vmin=0, vmax=20,
        interpolation="bicubic",
    )
    axes[1].set_title("(b) Conductivity ($\\sigma$)", fontsize=13)
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Depth (m)")
    fig.colorbar(im1, ax=axes[1], label="Conductivity (mS/m)",
                 shrink=0.85)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "lavoue2014_benchmark_model.png")
    fig.savefig(out_path, dpi=300)
    print(f"  Saved test figure: {out_path}")
