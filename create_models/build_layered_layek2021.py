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
#   Layek, M. K., & Sengupta, P. (2021). An Improved and Novel Approach
#   for Frequency Domain Forward Modeling of GPR Data Using the Finite
#   Difference Staggered Grid Technique. Pure Appl. Geophys., 178, 959-972.
#   https://doi.org/10.1007/s00024-021-02685-3
#
# Copyright (c) Mrinal Kanti Layek
# =============================================================================
"""
Build the synthetic layered model from Layek & Sengupta (2021), Section 3.2.1,
Figures 5-7.

Model description (from the paper):
------------------------------------
A 10 m x 10 m subsurface model (but the FD grid is 120 x 80 cells at
dh = 0.05 m = 6.0 m x 4.0 m total, with 10-cell PML on each side giving
100 x 60 interior cells = 5.0 m x 3.0 m physical interior domain).

Three sedimentary layers with a clay lens/patch in the lowermost layer:

  Layer 1 (Silty soil):
      epsr = 11.0, sigma = 0.1 mS/m (0.1e-3 S/m)
      Extends from surface to ~1.0 m depth (interior rows 0-19)

  Layer 2 (Dry sandy soil):
      epsr = 4.0, sigma = 6.0 mS/m (6.0e-3 S/m)
      Extends from ~1.0 m to ~2.0 m depth (interior rows 20-39)

  Layer 3 (Dry clay):
      epsr = 8.0, sigma = 5.0 mS/m (5.0e-3 S/m)
      Extends from ~2.0 m depth to bottom (interior rows 40-59)

  Clay lens/patch (in lower-right of Layer 3):
      epsr = 35.0, sigma = 20.0 mS/m (20.0e-3 S/m)
      Located approximately at interior rows 40-55, cols 60-85
      (depth ~2.5-3.25 m, horizontal ~3.5-4.75 m in interior coordinates)

The clay lens causes greater attenuation than the overlying layers, visible
as a low/nil amplitude region in the wavefield (Fig. 6) and dipping events
in the shot gather (Fig. 7).

Note: The "10 m x 10 m" mentioned in the paper text appears to be a
rounded description; the actual FD grid is 120 x 80 at dh=0.05 m.
"""
from __future__ import annotations

import numpy as np


def build_layered_layek2021_model(
    nx: int,
    nz: int,
    npml: int,
    dh: float,
    *,
    # Layer 1 - Silty soil (top)
    epsr_layer1: float = 11.0,
    sigma_layer1: float = 0.1e-3,
    # Layer 2 - Dry sandy soil
    epsr_layer2: float = 4.0,
    sigma_layer2: float = 6.0e-3,
    # Layer 3 - Dry clay (bottom)
    epsr_layer3: float = 8.0,
    sigma_layer3: float = 5.0e-3,
    # Clay lens/patch inside Layer 3
    epsr_clay_lens: float = 35.0,
    sigma_clay_lens: float = 20.0e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the Layek & Sengupta (2021) synthetic layered model.

    Parameters
    ----------
    nx, nz : int
        Full grid dimensions (including PML).
    npml : int
        Number of PML cells on each side.
    dh : float
        Grid spacing [m].
    epsr_layer1, sigma_layer1 : float
        Silty soil (top layer) properties.
    epsr_layer2, sigma_layer2 : float
        Dry sandy soil (middle layer) properties.
    epsr_layer3, sigma_layer3 : float
        Dry clay (bottom layer) properties.
    epsr_clay_lens, sigma_clay_lens : float
        Clay lens/patch anomaly properties.

    Returns
    -------
    epsr_2d : ndarray, shape (nz, nx)
        Relative permittivity model.
    sigma_2d : ndarray, shape (nz, nx)
        Conductivity model [S/m].
    """
    epsr_2d = np.ones((nz, nx), dtype=np.float64)
    sigma_2d = np.zeros((nz, nx), dtype=np.float64)

    # Interior domain indices
    # Interior spans from npml to (nz - npml) in z and npml to (nx - npml) in x
    iz_int_start = npml          # first interior row
    ix_int_start = npml          # first interior col
    nz_int = nz - 2 * npml      # number of interior rows (60 for 80-20)
    nx_int = nx - 2 * npml      # number of interior cols (100 for 120-20)

    # Layer boundaries in interior cells (0-indexed within interior)
    # Layer 1: rows 0-19  (1.0 m / 0.05 m = 20 cells)
    # Layer 2: rows 20-39 (1.0 m thick = 20 cells)
    # Layer 3: rows 40-59 (remaining to bottom)
    layer1_thickness_cells = 20  # 1.0 m at dh=0.05 m
    layer2_thickness_cells = 20  # 1.0 m at dh=0.05 m

    # Convert interior row indices to full-grid indices
    iz_layer1_end = iz_int_start + layer1_thickness_cells  # row where layer 2 starts
    iz_layer2_end = iz_layer1_end + layer2_thickness_cells  # row where layer 3 starts

    # ---- Fill interior + side/bottom PML with Layer 1 ----
    # Top PML (iz < npml) stays as air: epsr=1, sigma=0  (air half-space above
    # the ground surface).  This creates the air-ground interface at iz=npml,
    # which generates air waves, head waves and surface waves — essential for
    # realistic GPR shot gathers.
    epsr_2d[iz_int_start:, :] = epsr_layer1
    sigma_2d[iz_int_start:, :] = sigma_layer1

    # ---- Layer 2: Dry sandy soil ----
    epsr_2d[iz_layer1_end:iz_layer2_end, :] = epsr_layer2
    sigma_2d[iz_layer1_end:iz_layer2_end, :] = sigma_layer2

    # ---- Layer 3: Dry clay (from layer2_end to bottom) ----
    epsr_2d[iz_layer2_end:, :] = epsr_layer3
    sigma_2d[iz_layer2_end:, :] = sigma_layer3

    # ---- Clay lens/patch in Layer 3 (lower-right portion) ----
    # From Figure 5: the clay patch is roughly in the right-centre to
    # lower-right of the model. Estimating from the colormap:
    #   Interior rows ~40-55 (depth 2.0-2.75 m from interior top)
    #   Interior cols ~60-85 (horizontal 3.0-4.25 m from interior left)
    # In full-grid coords: rows (npml+40) to (npml+55), cols (npml+60) to (npml+85)
    iz_lens_start = iz_int_start + 40   # depth start of clay lens (interior row 40)
    iz_lens_end = iz_int_start + 56     # depth end (interior row 55, inclusive)
    ix_lens_start = ix_int_start + 60   # horizontal start (interior col 60)
    ix_lens_end = ix_int_start + 86     # horizontal end (interior col 85, inclusive)

    # Clip to grid bounds
    iz_lens_end = min(iz_lens_end, nz)
    ix_lens_end = min(ix_lens_end, nx)

    epsr_2d[iz_lens_start:iz_lens_end, ix_lens_start:ix_lens_end] = epsr_clay_lens
    sigma_2d[iz_lens_start:iz_lens_end, ix_lens_start:ix_lens_end] = sigma_clay_lens

    return epsr_2d, sigma_2d


def print_model_summary(
    nx: int, nz: int, npml: int, dh: float,
) -> None:
    """Print a summary of the Layek 2021 layered model geometry."""
    nz_int = nz - 2 * npml
    nx_int = nx - 2 * npml
    print("=" * 65)
    print("Layek & Sengupta (2021) - Synthetic Layered Model (Sec. 3.2.1)")
    print("=" * 65)
    print(f"  Full grid      : {nx} x {nz}  (dh = {dh} m)")
    print(f"  PML            : {npml} cells per side")
    print(f"  Interior grid  : {nx_int} x {nz_int}  "
          f"= {nx_int*dh:.1f} m x {nz_int*dh:.1f} m")
    print(f"  Layer 1 (silty soil)  : z = 0 - 1.0 m   "
          f"epsr=11.0, sigma=0.1 mS/m")
    print(f"  Layer 2 (dry sand)    : z = 1.0 - 2.0 m  "
          f"epsr=4.0,  sigma=6.0 mS/m")
    print(f"  Layer 3 (dry clay)    : z = 2.0 - 3.0 m  "
          f"epsr=8.0,  sigma=5.0 mS/m")
    print(f"  Clay lens/patch       : z ~ 2.0-2.8 m, x ~ 3.0-4.3 m  "
          f"epsr=35.0, sigma=20.0 mS/m")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _nx, _nz, _npml, _dh = 120, 80, 10, 0.05
    print_model_summary(_nx, _nz, _npml, _dh)

    epsr, sigma = build_layered_layek2021_model(_nx, _nz, _npml, _dh)
    print(f"  epsr  range: [{epsr.min():.1f}, {epsr.max():.1f}]")
    print(f"  sigma range: [{sigma.min():.4f}, {sigma.max():.4f}] S/m")

    # Interior extent for axis labels
    x_ext = [_npml * _dh, (_nx - _npml) * _dh]
    z_ext = [_npml * _dh, (_nz - _npml) * _dh]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    fig.savefig("layek2021_model_test.png", dpi=150)
    print("  Saved test figure: layek2021_model_test.png")
