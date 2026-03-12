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
#   Universite de Grenoble. Section 2.2.3.1, Figure 2.14.
#
#   Lavoue et al. (2014). A strategy for multiparameter FWI of on-ground
#   GPR data. Pure Appl. Geophys. Figure 9.
#
#   Kohn, D., De Nil, D. and Rabbel, W. (2017) Tutorial: Introduction to
#   frequency domain modelling and FWI of georadar data with GERMAINE.
#   DOI: 10.13140/RG.2.2.29354.03523
#
# Copyright (c) Mrinal Kanti Layek
# =============================================================================
"""
Build a realistic subsurface GPR benchmark model based on Lavoue (2014).

Reproduces the fluvial-deposit cross-section from Figure 2.14 / Figure 9:

    Lavoue, F. (2014). PhD thesis, Universite de Grenoble.
    Lavoue et al. (2014). Pure Appl. Geophys.

The model represents a cross-section of fluvial deposits near Grenoble,
France (5 m deep x 10 m long).  Structure (top to bottom):

    1. Air (epsr=1, sigma=0)
    2. Silty soil, epsr=11.0, sigma=0.1 mS/m
    3. Clay lenses top-right: thin horizontal streaks, epsr=31-32
    4. Dry sand body (dominant): epsr=4.0, sigma=0.1 mS/m
    5. Thin sigma=6 mS/m layer within sand at z~2.5 m
    6. Alternating zone at z~3.0 m (epsr 4/22 contrast)
    7. Strongly attenuating layer z~3.5 m (sigma=10 mS/m)
    8. Saturated clay below water table z~3.8 m (epsr=32, sigma=20 mS/m)

Two paths: SEGY-based (requires segyio) or synthetic fallback (NumPy only).
"""
from __future__ import annotations

import os
import numpy as np


# ---------------------------------------------------------------------------
# Petrophysical defaults (used only by SEGY path)
# ---------------------------------------------------------------------------
_DEFAULTS = dict(
    rho_matrix=2.65,         # g/cm3  quartz grain density (Marmousi-II units)
    rho_fluid=1.00,          # g/cm3  freshwater
    eps_matrix=5.0,          # quartz dielectric constant
    eps_water=80.0,          # freshwater dielectric constant
    Sw=1.0,                  # water saturation (fully saturated)
    sigma_fluid=0.05,        # S/m  freshwater conductivity
    m_cementation=1.8,       # Archie cementation exponent
    n_saturation=2.0,        # Archie saturation exponent
    # Target GPR value ranges (Lavoue 2014, Fig. 9)
    epsr_target_min=4.0,     # minimum epsr in subsurface
    epsr_target_max=32.0,    # maximum epsr (saturated clay)
    sigma_target_min=0.1e-3, # S/m  minimum sigma (0.1 mS/m)
    sigma_target_max=20.0e-3,  # S/m  maximum sigma (20 mS/m)
)


# ---------------------------------------------------------------------------
# SEGY-based builder
# ---------------------------------------------------------------------------
def build_marmousi_gpr_from_segy(
    nx: int,
    nz: int,
    npml: int,
    dh: float,
    *,
    segy_dir: str,
    rho_matrix: float = _DEFAULTS["rho_matrix"],
    rho_fluid: float = _DEFAULTS["rho_fluid"],
    eps_matrix: float = _DEFAULTS["eps_matrix"],
    eps_water: float = _DEFAULTS["eps_water"],
    Sw: float = _DEFAULTS["Sw"],
    sigma_fluid: float = _DEFAULTS["sigma_fluid"],
    m_cementation: float = _DEFAULTS["m_cementation"],
    n_saturation: float = _DEFAULTS["n_saturation"],
    epsr_target_min: float = _DEFAULTS["epsr_target_min"],
    epsr_target_max: float = _DEFAULTS["epsr_target_max"],
    sigma_target_min: float = _DEFAULTS["sigma_target_min"],
    sigma_target_max: float = _DEFAULTS["sigma_target_max"],
    skip_water_rows: int = 370,
    z_crop_rows: int = 600,
    x_crop_start: int = 3000,
    x_crop_end: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read Marmousi-II density SEGY, convert to GPR parameters, and rescale
    to match Lavoue (2014) value ranges.

    The 1.25 m grid version of Marmousi-II has density in **g/cm3** (not
    kg/m3), NX=13601 traces, NZ=2801 samples.  The water column occupies
    rows 0-~369 (Vp=1500, rho=1.0).

    Conversion chain (pixel-by-pixel, preserves ALL structural features):
      1. rho_bulk -> porosity via density mixing
      2. porosity -> eps_r via CRIM
      3. porosity -> sigma via Archie's Law
      4. Resample cropped region to target FDFD grid
      5. Rescale to paper value ranges [epsr_target_min..max],
         [sigma_target_min..max]

    Parameters
    ----------
    segy_dir : str
        Directory containing ``MODEL_DENSITY_1.25m.segy``.
    skip_water_rows : int
        Rows to skip for water column (default 370 at DH=1.25m = 462m).
    z_crop_rows : int
        Number of sediment rows to keep (default 600 = 750m depth).
    x_crop_start, x_crop_end : int
        Lateral column range to crop (default 3000:10000 = central 8.75km).
    epsr_target_min/max : float
        Target epsr range after rescaling (paper: 4-32).
    sigma_target_min/max : float
        Target sigma range after rescaling (paper: 0.1-20 mS/m).
    """
    import segyio
    from scipy.ndimage import zoom

    # --- locate density SEGY ------------------------------------------------
    density_file = None
    for fn in os.listdir(segy_dir):
        fl = fn.lower()
        if "density" in fl and (fl.endswith(".segy") or fl.endswith(".sgy")):
            density_file = os.path.join(segy_dir, fn)
            break
    if density_file is None:
        raise FileNotFoundError(
            f"No density SEGY in {segy_dir}. "
            "Expected MODEL_DENSITY_1.25m.segy."
        )

    # --- read SEGY ----------------------------------------------------------
    with segyio.open(density_file, ignore_geometry=True) as f:
        rho_raw = np.array([f.trace[i] for i in range(f.tracecount)])
    rho_T = rho_raw.T  # (NZ, NX) — density in g/cm3
    print(f"  SEGY: shape {rho_T.shape}, rho=[{rho_T.min():.3f}, "
          f"{rho_T.max():.3f}] g/cm3")

    # --- crop: skip water, select sediment window ----------------------------
    rho_crop = rho_T[skip_water_rows:skip_water_rows + z_crop_rows,
                     x_crop_start:x_crop_end]
    print(f"  Crop: {rho_crop.shape}, rho=[{rho_crop.min():.3f}, "
          f"{rho_crop.max():.3f}]")

    # --- resample to interior grid FIRST (preserves structure) ---------------
    nz_int = nz - 2 * npml
    nx_int = nx - 2 * npml
    zoom_z = nz_int / rho_crop.shape[0]
    zoom_x = nx_int / rho_crop.shape[1]
    rho_gpr = zoom(rho_crop, (zoom_z, zoom_x), order=1)[:nz_int, :nx_int]
    print(f"  Resampled: {rho_gpr.shape}, zoom=({zoom_z:.4f}, {zoom_x:.4f})")

    # --- petrophysical conversion on resampled grid --------------------------
    phi = np.clip(
        (rho_matrix - rho_gpr) / (rho_matrix - rho_fluid), 0.01, 0.50
    )

    # CRIM -> epsr
    sqrt_eps = ((1 - phi) * np.sqrt(eps_matrix)
                + phi * (Sw * np.sqrt(eps_water)
                         + (1 - Sw) * np.sqrt(1.0)))
    eps_r = sqrt_eps ** 2

    # Archie -> sigma
    sigma = np.clip(
        sigma_fluid * (phi ** m_cementation) * (Sw ** n_saturation),
        1e-4, 1.0,
    )
    print(f"  Raw conversion: epsr=[{eps_r.min():.1f}, {eps_r.max():.1f}], "
          f"sigma=[{sigma.min()*1e3:.2f}, {sigma.max()*1e3:.2f}] mS/m")

    # --- rescale to paper value ranges (Lavoue 2014) -------------------------
    e_lo, e_hi = eps_r.min(), eps_r.max()
    s_lo, s_hi = sigma.min(), sigma.max()
    e_range = epsr_target_max - epsr_target_min
    s_range = sigma_target_max - sigma_target_min

    eps_r_scaled = epsr_target_min + (eps_r - e_lo) / max(e_hi - e_lo, 1e-12) * e_range
    sigma_scaled = sigma_target_min + (sigma - s_lo) / max(s_hi - s_lo, 1e-12) * s_range
    print(f"  Scaled: epsr=[{eps_r_scaled.min():.1f}, {eps_r_scaled.max():.1f}], "
          f"sigma=[{sigma_scaled.min()*1e3:.1f}, {sigma_scaled.max()*1e3:.1f}] mS/m")

    # --- embed in full grid with PML + air top -------------------------------
    epsr_2d = np.ones((nz, nx), dtype=np.float64)
    sigma_2d = np.zeros((nz, nx), dtype=np.float64)

    epsr_2d[npml:npml + nz_int, npml:npml + nx_int] = eps_r_scaled
    sigma_2d[npml:npml + nz_int, npml:npml + nx_int] = sigma_scaled

    # Extend edges into PML padding
    epsr_2d[npml:, :npml] = epsr_2d[npml:, npml:npml + 1]
    sigma_2d[npml:, :npml] = sigma_2d[npml:, npml:npml + 1]
    epsr_2d[npml:, nx - npml:] = epsr_2d[npml:, nx - npml - 1:nx - npml]
    sigma_2d[npml:, nx - npml:] = sigma_2d[npml:, nx - npml - 1:nx - npml]
    epsr_2d[nz - npml:, :] = epsr_2d[nz - npml - 1:nz - npml, :]
    sigma_2d[nz - npml:, :] = sigma_2d[nz - npml - 1:nz - npml, :]

    return epsr_2d, sigma_2d


# ---------------------------------------------------------------------------
# Synthetic Lavoue (2014) Fig. 2.14 / Fig. 9 model
# ---------------------------------------------------------------------------
def build_synthetic_marmousi_model(
    nx: int,
    nz: int,
    npml: int,
    dh: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a realistic subsurface benchmark after Lavoue (2014), Fig. 2.14.

    Reproduces the fluvial-deposit cross-section near Grenoble, France.
    Layer depths follow Lavoue (2014): silty soil 0-0.3 m, dry sand
    0.5-3.0 m (with thin sigma=6 mS/m layer at ~2.5 m), alternating
    zone ~3.0 m, attenuating layer ~3.5 m (sigma=10 mS/m), saturated
    clay below water table ~3.8 m (epsr=32, sigma=20 mS/m).

    Parameters
    ----------
    nx, nz : int
        Full grid dimensions (including PML).
    npml : int
        Number of PML cells on each side.
    dh : float
        Grid spacing [m].

    Returns
    -------
    epsr_2d : ndarray, shape (nz, nx)
        Relative permittivity model.
    sigma_2d : ndarray, shape (nz, nx)
        Conductivity model [S/m].

    References
    ----------
    Lavoue, F. (2014). PhD thesis, Universite de Grenoble. Fig. 2.14.
    Lavoue et al. (2014). Pure Appl. Geophys. Fig. 9.
    """
    epsr_2d = np.ones((nz, nx), dtype=np.float64)   # air default
    sigma_2d = np.zeros((nz, nx), dtype=np.float64)  # air default

    nz_int = nz - 2 * npml
    nx_int = nx - 2 * npml

    # Ground surface at iz = npml (air above, subsurface below)
    iz_ground = npml

    # x-coordinates for smooth interface perturbations
    x = np.arange(nx, dtype=np.float64)

    # ------------------------------------------------------------------
    # Layer interfaces (full-grid z-indices)
    # Depth values from Lavoue (2014) Fig. 9 annotations:
    #   silty soil 0-0.3m, transition 0.3-0.5m, dry sand 0.5-~3.0m,
    #   sigma=6 mS/m layer at ~2.3-2.7m within sand,
    #   alternating zone ~3.0-3.5m, attenuating layer ~3.5-3.8m,
    #   saturated clay (water table) below ~3.8m
    # ------------------------------------------------------------------

    # Interface 1: bottom of silty soil (z ~ 0.3 m = 6 cells)
    iz_int1 = iz_ground + 6 + (
        1.0 * np.sin(2.0 * np.pi * x / nx * 2.0)
    ).astype(int)

    # Interface 2: top of dry sand body (z ~ 0.5 m = 10 cells)
    iz_int2 = iz_ground + 10 + (
        1.5 * np.sin(2.0 * np.pi * x / nx * 1.5 + 0.3)
    ).astype(int)

    # Interface 3a: top of thin sigma=6 mS/m layer (z ~ 2.3 m = 46 cells)
    iz_int3t = iz_ground + 46 + (
        1.5 * np.sin(2.0 * np.pi * x / nx * 1.2 + 0.8)
    ).astype(int)

    # Interface 3b: bottom of thin sigma=6 mS/m layer (z ~ 2.7 m = 54 cells)
    iz_int3b = iz_ground + 54 + (
        1.5 * np.sin(2.0 * np.pi * x / nx * 1.2 + 0.8)
    ).astype(int)

    # Interface 4: bottom of dry sand / top of alternating (z ~ 3.0 m = 60 cells)
    iz_int4 = iz_ground + 60 + (
        2.0 * np.sin(2.0 * np.pi * x / nx * 1.5 + 1.0)
    ).astype(int)

    # Interface 5: bottom of alternating / top of attenuating (z ~ 3.5 m = 70 cells)
    iz_int5 = iz_ground + 70 + (
        1.5 * np.sin(2.0 * np.pi * x / nx * 1.3 + 0.6)
    ).astype(int)

    # Interface 6: water table / top of saturated clay (z ~ 3.8 m = 76 cells)
    iz_int6 = iz_ground + 76 + (
        1.0 * np.sin(2.0 * np.pi * x / nx * 1.0 + 0.4)
    ).astype(int)

    # ------------------------------------------------------------------
    # Fill layers column by column
    # ------------------------------------------------------------------
    for ix in range(nx):
        # Default fill below ground: silty soil (epsr=11, sigma=0.1 mS/m)
        epsr_2d[iz_ground:, ix] = 11.0
        sigma_2d[iz_ground:, ix] = 0.1e-3

        # Transition below silty soil (epsr=8, sigma=2 mS/m)
        z1 = max(iz_ground, iz_int1[ix])
        z2 = max(z1, iz_int2[ix])
        if z2 > z1:
            epsr_2d[z1:z2, ix] = 8.0
            sigma_2d[z1:z2, ix] = 2.0e-3

        # Dry sand body — upper part (epsr=4, sigma=0.1 mS/m)
        z3t = max(z2, iz_int3t[ix])
        if z3t > z2:
            epsr_2d[z2:z3t, ix] = 4.0
            sigma_2d[z2:z3t, ix] = 0.1e-3

        # Thin sigma=6 mS/m layer within sand (z~2.3-2.7m)
        z3b = max(z3t, iz_int3b[ix])
        if z3b > z3t:
            epsr_2d[z3t:z3b, ix] = 6.0
            sigma_2d[z3t:z3b, ix] = 6.0e-3

        # Dry sand body — lower part continues (epsr=4, sigma=0.1 mS/m)
        z4 = max(z3b, iz_int4[ix])
        if z4 > z3b:
            epsr_2d[z3b:z4, ix] = 4.0
            sigma_2d[z3b:z4, ix] = 0.1e-3

        # Alternating transition zone (epsr 4/22 contrast)
        z5 = max(z4, iz_int5[ix])
        for iz in range(z4, min(z5, nz)):
            sub_idx = (iz - z4) // 3  # alternate every 3 cells (~15 cm)
            if sub_idx % 2 == 0:
                epsr_2d[iz, ix] = 22.0
                sigma_2d[iz, ix] = 8.0e-3
            else:
                epsr_2d[iz, ix] = 4.0
                sigma_2d[iz, ix] = 1.0e-3

        # Strongly attenuating layer (epsr=18, sigma=10 mS/m)
        z6 = max(z5, iz_int6[ix])
        if z6 > z5:
            epsr_2d[z5:z6, ix] = 18.0
            sigma_2d[z5:z6, ix] = 10.0e-3

        # Below water table: saturated clay (epsr=32, sigma=20 mS/m)
        if nz > z6:
            epsr_2d[z6:, ix] = 32.0
            sigma_2d[z6:, ix] = 20.0e-3

    # ------------------------------------------------------------------
    # Clay lenses in the top-right: thin horizontal streaks
    # (Lavoue 2014 Fig. 9: 2-3 stacked thin clay bands, x > ~55% domain,
    #  each ~5-10 cm thick, with slight undulation)
    # ------------------------------------------------------------------
    ix_lens_start1 = npml + int(nx_int * 0.55)
    ix_lens_start2 = npml + int(nx_int * 0.60)
    ix_lens_start3 = npml + int(nx_int * 0.65)
    ix_lens_end = nx - npml  # right interior edge

    # Lens 1: z ~ 0.15-0.25 m (3-5 cells below ground), epsr=31
    for ix in range(ix_lens_start1, min(ix_lens_end, nx)):
        wobble = int(0.5 * np.sin(2.0 * np.pi * ix / nx * 4.0))
        z_top = max(iz_ground, iz_ground + 3 + wobble)
        z_bot = min(nz, iz_ground + 5 + wobble)
        epsr_2d[z_top:z_bot, ix] = 31.0
        sigma_2d[z_top:z_bot, ix] = 15.0e-3

    # Lens 2: z ~ 0.35-0.45 m (7-9 cells), epsr=32
    for ix in range(ix_lens_start2, min(ix_lens_end, nx)):
        wobble = int(0.5 * np.sin(2.0 * np.pi * ix / nx * 3.5 + 1.0))
        z_top = max(iz_ground, iz_ground + 7 + wobble)
        z_bot = min(nz, iz_ground + 9 + wobble)
        epsr_2d[z_top:z_bot, ix] = 32.0
        sigma_2d[z_top:z_bot, ix] = 15.0e-3

    # Lens 3: z ~ 0.55-0.65 m (11-13 cells), epsr=28
    for ix in range(ix_lens_start3, min(ix_lens_end, nx)):
        wobble = int(0.5 * np.sin(2.0 * np.pi * ix / nx * 3.0 + 2.0))
        z_top = max(iz_ground, iz_ground + 11 + wobble)
        z_bot = min(nz, iz_ground + 13 + wobble)
        epsr_2d[z_top:z_bot, ix] = 28.0
        sigma_2d[z_top:z_bot, ix] = 12.0e-3

    # ------------------------------------------------------------------
    # Extend edges into PML padding
    # ------------------------------------------------------------------
    # Left PML
    epsr_2d[npml:, :npml] = epsr_2d[npml:, npml:npml + 1]
    sigma_2d[npml:, :npml] = sigma_2d[npml:, npml:npml + 1]
    # Right PML
    epsr_2d[npml:, nx - npml:] = epsr_2d[npml:, nx - npml - 1:nx - npml]
    sigma_2d[npml:, nx - npml:] = sigma_2d[npml:, nx - npml - 1:nx - npml]
    # Bottom PML
    epsr_2d[nz - npml:, :] = epsr_2d[nz - npml - 1:nz - npml, :]
    sigma_2d[nz - npml:, :] = sigma_2d[nz - npml - 1:nz - npml, :]

    return epsr_2d, sigma_2d


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def build_marmousi_gpr_model(
    nx: int,
    nz: int,
    npml: int,
    dh: float,
    *,
    segy_dir: str | None = None,
    rho_matrix: float = _DEFAULTS["rho_matrix"],
    rho_fluid: float = _DEFAULTS["rho_fluid"],
    eps_matrix: float = _DEFAULTS["eps_matrix"],
    eps_water: float = _DEFAULTS["eps_water"],
    Sw: float = _DEFAULTS["Sw"],
    sigma_fluid: float = _DEFAULTS["sigma_fluid"],
    m_cementation: float = _DEFAULTS["m_cementation"],
    n_saturation: float = _DEFAULTS["n_saturation"],
    epsr_target_min: float = _DEFAULTS["epsr_target_min"],
    epsr_target_max: float = _DEFAULTS["epsr_target_max"],
    sigma_target_min: float = _DEFAULTS["sigma_target_min"],
    sigma_target_max: float = _DEFAULTS["sigma_target_max"],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a GPR model from Marmousi-II elastic data.

    Tries the SEGY path first (if *segy_dir* is provided and ``segyio`` is
    installed).  Falls back to the synthetic Lavoue (2014) model otherwise.
    """
    # --- Try SEGY path first ------------------------------------------------
    if segy_dir is not None and os.path.isdir(str(segy_dir)):
        try:
            print(f"  Attempting SEGY-based Marmousi-II conversion "
                  f"from: {segy_dir}")
            return build_marmousi_gpr_from_segy(
                nx, nz, npml, dh,
                segy_dir=str(segy_dir),
                rho_matrix=rho_matrix,
                rho_fluid=rho_fluid,
                eps_matrix=eps_matrix,
                eps_water=eps_water,
                Sw=Sw,
                sigma_fluid=sigma_fluid,
                m_cementation=m_cementation,
                n_saturation=n_saturation,
                epsr_target_min=epsr_target_min,
                epsr_target_max=epsr_target_max,
                sigma_target_min=sigma_target_min,
                sigma_target_max=sigma_target_max,
            )
        except ImportError:
            print("  segyio not installed -- falling back to synthetic model.")
        except FileNotFoundError as exc:
            print(f"  {exc}")
            print("  Falling back to synthetic model.")
    else:
        if segy_dir is not None:
            print(f"  SEGY directory not found: {segy_dir}")
        print("  Using synthetic Lavoue (2014) GPR model.")

    # --- Fallback: synthetic Lavoue (2014) model ----------------------------
    return build_synthetic_marmousi_model(nx, nz, npml, dh)


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------
def print_model_summary(
    nx: int, nz: int, npml: int, dh: float,
) -> None:
    """Print a summary of the Lavoue (2014) GPR benchmark model geometry."""
    nz_int = nz - 2 * npml
    nx_int = nx - 2 * npml
    print("=" * 65)
    print("Lavoue (2014) GPR Benchmark -- Fluvial Deposits, Grenoble")
    print("  Ref: Lavoue (2014) Fig. 2.14 / Lavoue et al. (2014) Fig. 9")
    print("=" * 65)
    print(f"  Full grid      : {nx} x {nz}  (dh = {dh} m)")
    print(f"  PML            : {npml} cells per side")
    print(f"  Interior grid  : {nx_int} x {nz_int}  "
          f"= {nx_int * dh:.1f} m x {nz_int * dh:.1f} m")
    print("-" * 65)
    print("  Layers (Lavoue 2014, Fig. 9 annotations):")
    print("    Air (above ground)       : epsr= 1.0, sigma=  0.0 mS/m")
    print("    Silty soil (0-0.3m)      : epsr=11.0, sigma=  0.1 mS/m")
    print("    Clay lenses (top-right)  : epsr=28-32, sigma=12-15 mS/m")
    print("    Dry sand (0.5-3.0m)      : epsr= 4.0, sigma=  0.1 mS/m")
    print("    Thin layer (~2.5m)       : epsr= 6.0, sigma=  6.0 mS/m")
    print("    Alternating (~3.0m)      : epsr=4/22, sigma= 1-8 mS/m")
    print("    Attenuating (~3.5m)      : epsr=18.0, sigma= 10.0 mS/m")
    print("    Saturated clay (>3.8m)   : epsr=32.0, sigma= 20.0 mS/m")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _nx, _nz, _npml, _dh = 300, 200, 10, 0.05
    print_model_summary(_nx, _nz, _npml, _dh)

    # Auto-detect SEGY directory relative to this script
    _segy_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                             "elastic-marmousi-model", "model")
    if not os.path.isdir(_segy_dir):
        _segy_dir = None   # fall back to synthetic model

    epsr, sigma = build_marmousi_gpr_model(_nx, _nz, _npml, _dh,
                                           segy_dir=_segy_dir)
    print(f"  epsr  range: [{epsr.min():.2f}, {epsr.max():.2f}]")
    print(f"  sigma range: [{sigma.min():.5f}, {sigma.max():.5f}] S/m")

    # Interior extent for axis labels
    x_ext = [_npml * _dh, (_nx - _npml) * _dh]
    z_ext = [_npml * _dh, (_nz - _npml) * _dh]

    # Create output directory
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           "results", "marmousi_work")
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(
        epsr[_npml:_nz - _npml, _npml:_nx - _npml],
        extent=[x_ext[0], x_ext[1], z_ext[1], z_ext[0]],
        aspect="auto", cmap="jet",
    )
    axes[0].set_title("Relative permittivity", fontsize=14)
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    fig.colorbar(im0, ax=axes[0], label="Relative permittivity")

    im1 = axes[1].imshow(
        sigma[_npml:_nz - _npml, _npml:_nx - _npml] * 1e3,
        extent=[x_ext[0], x_ext[1], z_ext[1], z_ext[0]],
        aspect="auto", cmap="jet",
    )
    axes[1].set_title("Conductivity", fontsize=14)
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Depth (m)")
    fig.colorbar(im1, ax=axes[1], label="Conductivity (mS/m)")

    fig.tight_layout()
    out_path = os.path.join(out_dir, "marmousi_gpr_model_test.png")
    fig.savefig(out_path, dpi=150)
    print(f"  Saved test figure: {out_path}")
