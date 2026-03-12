# =============================================================================
# RFDFWI - Full-Waveform Inversion (FWI) of GPR Data  |  By Mrinal
#
# Trace the Lavoue (2014) Fig. 9(a) permittivity model directly from the
# published figure image.  The colorbar is sampled to build a
# colour -> epsilon_r lookup, then every pixel in the plot area is
# converted to a dielectric value and resampled to the FDFD grid.
#
# Copyright (c) Mrinal Kanti Layek
# =============================================================================
"""
trace_lavoue2014_figure.py
==========================
Extract the eps_r model from the Lavoue (2014) Fig. 9(a) image.

Pipeline
--------
1. Load image  ->  detect plot area + colorbar automatically
2. Sample colorbar strip  ->  build KD-tree colour-to-value lookup
3. Map every plot pixel to eps_r  (nearest-colour in colorbar)
4. Median-filter to remove JPEG / anti-aliasing noise
5. Resample to simulation grid  (scipy.ndimage.zoom)
6. Infer sigma from eps_r using known layer assignments
7. Save  ->  .npz  +  comparison figure  +  model builder integration

Usage
-----
    cd D:/rfdfwi
    python create_models/trace_lavoue2014_figure.py

Output  (all in results/benchmark/)
------
    traced_epsr_raw.png          raw pixel-level extraction
    traced_model.npz             eps_r + sigma on simulation grid
    traced_model_comparison.png  4-panel diagnostic figure
"""
from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom, median_filter
from scipy.spatial import cKDTree

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# 1.  Load image and detect plot / colorbar regions
# -----------------------------------------------------------------------
def _detect_plot_and_cbar(arr: np.ndarray):
    """
    Auto-detect the data-area rectangle and the colorbar column.

    Strategy
    --------
    * 'Coloured' pixels are those with saturation > 25 (i.e. not white,
      not black, not grey).
    * The plot area is the bounding box of the coloured-pixel region.
    * The colorbar is a narrow vertical strip whose right edge coincides
      with the rightmost coloured column.  We walk leftward from there
      until horizontal colour uniformity breaks.

    Returns  (plot_left, plot_top, plot_right, plot_bottom, cbar_col)
    """
    r = arr[:, :, 0].astype(np.float64)
    g = arr[:, :, 1].astype(np.float64)
    b = arr[:, :, 2].astype(np.float64)

    # Saturation proxy: max_channel - min_channel
    mx = np.maximum(r, np.maximum(g, b))
    mn = np.minimum(r, np.minimum(g, b))
    sat = mx - mn
    is_coloured = sat > 25

    # Bounding box of coloured pixels
    rows = np.any(is_coloured, axis=1)
    cols = np.any(is_coloured, axis=0)
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]

    plot_top    = int(row_idx[0])
    plot_bottom = int(row_idx[-1]) + 1
    plot_left   = int(col_idx[0])
    far_right   = int(col_idx[-1]) + 1

    # ---- Separate the colorbar from the plot area ----
    # The colorbar is the rightmost narrow strip of coloured pixels.
    # Walk leftward from far_right and check horizontal uniformity.
    mid_row = (plot_top + plot_bottom) // 2
    cbar_right = far_right
    cbar_left  = cbar_right

    for c in range(far_right - 2, plot_left, -1):
        col_strip = arr[plot_top:plot_bottom, c, :].astype(np.float64)
        adj_strip = arr[plot_top:plot_bottom, c + 1, :].astype(np.float64)
        diff = np.abs(col_strip - adj_strip).mean()
        if diff > 20:          # big horizontal change → left edge of cbar
            cbar_left = c + 1
            break

    # The plot right edge is 2 px before the colorbar
    plot_right = max(plot_left + 10, cbar_left - 2)

    # Use the middle column of the colorbar for sampling
    cbar_col = (cbar_left + cbar_right) // 2

    return plot_left, plot_top, plot_right, plot_bottom, cbar_col


# -----------------------------------------------------------------------
# 2.  Build colour → value lookup from the colorbar
# -----------------------------------------------------------------------
def _build_lut(arr, plot_top, plot_bottom, cbar_col, vmin=1.0, vmax=31.0):
    """
    Build a KD-tree colour-to-value lookup using the **known** RdBu_r
    colormap rather than sampling the (often too-small) colorbar strip
    in the image.

    We synthesise 512 RGB samples from matplotlib's RdBu_r spanning
    [vmin, vmax] and build the tree from those.  This is far more robust
    than relying on a tiny, partially-cropped colorbar.
    """
    cmap = plt.get_cmap("RdBu_r")
    n = 512
    values = np.linspace(vmin, vmax, n)
    norm_vals = (values - vmin) / (vmax - vmin)       # 0..1
    rgba = cmap(norm_vals)                             # (n, 4) float 0..1
    rgb = (rgba[:, :3] * 255.0)                       # to 0..255 scale
    tree = cKDTree(rgb)

    # Keep a reference strip for the diagnostic figure
    strip = arr[plot_top:plot_bottom, cbar_col, :].astype(np.float64)
    return tree, values, strip


# -----------------------------------------------------------------------
# 3.  Map every plot pixel to eps_r
# -----------------------------------------------------------------------
def _pixels_to_epsr(arr, plot_left, plot_top, plot_right, plot_bottom,
                    tree, values):
    """Nearest-colour lookup for every pixel in the plot area."""
    plot_rgb = arr[plot_top:plot_bottom, plot_left:plot_right, :]
    h, w = plot_rgb.shape[:2]
    flat = plot_rgb.reshape(-1, 3).astype(np.float64)
    _, idx = tree.query(flat)
    return values[idx].reshape(h, w)


# -----------------------------------------------------------------------
# 4.  Resample to simulation grid
# -----------------------------------------------------------------------
def _resample(epsr_raw, nz_out, nx_out):
    """Zoom to target grid, clip, and median-filter."""
    # Light median filter first to reduce noise
    epsr_clean = median_filter(epsr_raw, size=3)
    zf_z = nz_out / epsr_clean.shape[0]
    zf_x = nx_out / epsr_clean.shape[1]
    epsr_grid = zoom(epsr_clean, (zf_z, zf_x), order=1)
    return np.clip(epsr_grid, 1.0, 32.0)


# -----------------------------------------------------------------------
# 5.  Infer conductivity from eps_r  (known layer assignments)
# -----------------------------------------------------------------------
def _epsr_to_sigma(epsr):
    """
    Piecewise mapping from eps_r to sigma [S/m] using the Lavoue (2014)
    benchmark layer values.

    eps_r range     | material           | sigma
    ----------------+--------------------+-----------
      < 2           | air                |  0
      2 –  5.5      | dry sand           |  0.1  mS/m
      5.5 – 8       | thin layer (eps 6) |  6.0  mS/m
      8 – 14        | silty soil         |  0.1  mS/m
     14 – 20        | attenuating        | 10.0  mS/m
     20 – 27        | alternating high   |  8.0  mS/m
     >= 27          | saturated clay     | 20.0  mS/m
    """
    sigma = np.zeros_like(epsr)
    sigma[epsr < 2.0]                            = 0.0
    sigma[(epsr >= 2.0)  & (epsr < 5.5)]         = 0.1e-3
    sigma[(epsr >= 5.5)  & (epsr < 8.0)]         = 6.0e-3
    sigma[(epsr >= 8.0)  & (epsr < 14.0)]        = 0.1e-3
    sigma[(epsr >= 14.0) & (epsr < 20.0)]        = 10.0e-3
    sigma[(epsr >= 20.0) & (epsr < 27.0)]        = 8.0e-3
    sigma[epsr >= 27.0]                           = 20.0e-3
    return sigma


# -----------------------------------------------------------------------
# 6.  Build full-grid model (with PML) for FDFD
# -----------------------------------------------------------------------
def build_traced_model(epsr_interior, sigma_interior, nx, nz, npml):
    """Embed interior arrays in the full grid with PML padding."""
    epsr  = np.ones((nz, nx), dtype=np.float64)
    sigma = np.zeros((nz, nx), dtype=np.float64)
    epsr[npml:nz - npml,  npml:nx - npml] = epsr_interior
    sigma[npml:nz - npml, npml:nx - npml] = sigma_interior

    # Left / right / bottom PML padding
    epsr[npml:,  :npml]       = epsr[npml:,  npml:npml + 1]
    sigma[npml:, :npml]       = sigma[npml:, npml:npml + 1]
    epsr[npml:,  nx - npml:]  = epsr[npml:,  nx - npml - 1:nx - npml]
    sigma[npml:, nx - npml:]  = sigma[npml:, nx - npml - 1:nx - npml]
    epsr[nz - npml:, :]       = epsr[nz - npml - 1:nz - npml, :]
    sigma[nz - npml:, :]      = sigma[nz - npml - 1:nz - npml, :]
    return epsr, sigma


# -----------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------
def trace_figure(image_path, out_dir,
                 nx_int=200, nz_int=110, dh=0.05,
                 cbar_vmin=1.0, cbar_vmax=31.0):
    """
    Full pipeline: image → simulation-ready eps_r and sigma grids.
    """
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. Load ----
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h_img, w_img = arr.shape[:2]
    print(f"  Image: {image_path}")
    print(f"  Size : {w_img} x {h_img} px")

    # ---- 2. Detect regions ----
    pl, pt, pr, pb, ccol = _detect_plot_and_cbar(arr)
    print(f"  Plot area : x=[{pl},{pr}]  y=[{pt},{pb}]  "
          f"({pr - pl} x {pb - pt} px)")
    print(f"  Colorbar col: {ccol}")

    # ---- 3. Colorbar LUT ----
    tree, values, cbar_strip = _build_lut(
        arr, pt, pb, ccol, vmin=cbar_vmin, vmax=cbar_vmax)
    print(f"  Colorbar LUT: {len(cbar_strip)} entries  "
          f"[{cbar_vmin:.0f} .. {cbar_vmax:.0f}]")

    # ---- 4. Pixel → eps_r ----
    epsr_raw = _pixels_to_epsr(arr, pl, pt, pr, pb, tree, values)
    print(f"  Raw eps_r: shape {epsr_raw.shape}  "
          f"range [{epsr_raw.min():.1f}, {epsr_raw.max():.1f}]")

    # Save raw extraction
    fig_raw, ax_raw = plt.subplots(figsize=(10, 4))
    im_raw = ax_raw.imshow(epsr_raw, cmap="RdBu_r", vmin=1, vmax=32,
                           aspect="auto", interpolation="nearest")
    ax_raw.set_title("Raw pixel extraction (before resampling)")
    fig_raw.colorbar(im_raw, ax=ax_raw, label="$\\varepsilon_r$")
    fig_raw.tight_layout()
    fig_raw.savefig(os.path.join(out_dir, "traced_epsr_raw.png"), dpi=200)
    plt.close(fig_raw)

    # ---- 5. Resample ----
    epsr_grid = _resample(epsr_raw, nz_int, nx_int)
    print(f"  Resampled : ({nz_int}, {nx_int})  "
          f"range [{epsr_grid.min():.1f}, {epsr_grid.max():.1f}]")

    # ---- 6. Infer sigma ----
    sigma_grid = _epsr_to_sigma(epsr_grid)
    print(f"  Sigma     : [{sigma_grid.min()*1e3:.2f}, "
          f"{sigma_grid.max()*1e3:.1f}] mS/m")

    # ---- 7. Save .npz ----
    npz_path = os.path.join(out_dir, "traced_model.npz")
    np.savez(npz_path,
             epsr=epsr_grid, sigma=sigma_grid,
             nx_int=nx_int, nz_int=nz_int, dh=dh,
             cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax)
    print(f"  Data -> {npz_path}")

    # ---- 8. Diagnostic figure ----
    n_air = 10
    x_ext = [0.0, nx_int * dh]
    z_ext = [-n_air * dh, (nz_int - n_air) * dh]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # (a) Original image with detected boundaries
    axes[0, 0].imshow(arr)
    axes[0, 0].set_title("(a) Original image + detected regions", fontsize=11)
    for y in (pt, pb):
        axes[0, 0].axhline(y, color="lime", ls="--", lw=0.8)
    for x in (pl, pr):
        axes[0, 0].axvline(x, color="lime", ls="--", lw=0.8)
    axes[0, 0].axvline(ccol, color="red", ls="--", lw=0.8, label="cbar")
    axes[0, 0].legend(fontsize=8)

    # (b) Raw pixel extraction
    im1 = axes[0, 1].imshow(epsr_raw, cmap="RdBu_r", vmin=1, vmax=32,
                             aspect="auto", interpolation="nearest")
    axes[0, 1].set_title(f"(b) Raw extraction  "
                         f"({epsr_raw.shape[0]}x{epsr_raw.shape[1]} px)",
                         fontsize=11)
    fig.colorbar(im1, ax=axes[0, 1], shrink=0.7, label="$\\varepsilon_r$")

    # (c) Resampled eps_r on simulation grid
    im2 = axes[1, 0].imshow(
        epsr_grid,
        extent=[x_ext[0], x_ext[1], z_ext[1], z_ext[0]],
        cmap="RdBu_r", vmin=1, vmax=32,
        aspect="auto", interpolation="bicubic")
    axes[1, 0].set_title(f"(c) Extracted $\\varepsilon_r$  "
                         f"({nz_int}x{nx_int})", fontsize=11)
    axes[1, 0].set_xlabel("Distance (m)")
    axes[1, 0].set_ylabel("Depth (m)")
    fig.colorbar(im2, ax=axes[1, 0], shrink=0.7, label="$\\varepsilon_r$")

    # (d) Inferred sigma
    im3 = axes[1, 1].imshow(
        sigma_grid * 1e3,
        extent=[x_ext[0], x_ext[1], z_ext[1], z_ext[0]],
        cmap="RdBu_r", vmin=0, vmax=20,
        aspect="auto", interpolation="bicubic")
    axes[1, 1].set_title(f"(d) Inferred $\\sigma$  "
                         f"({nz_int}x{nx_int})", fontsize=11)
    axes[1, 1].set_xlabel("Distance (m)")
    axes[1, 1].set_ylabel("Depth (m)")
    fig.colorbar(im3, ax=axes[1, 1], shrink=0.7, label="mS/m")

    fig.suptitle("Lavoue (2014) Fig. 9(a) - Image-to-Model Extraction",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig_path = os.path.join(out_dir, "traced_model_comparison.png")
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {fig_path}")

    return epsr_grid, sigma_grid


# -----------------------------------------------------------------------
if __name__ == "__main__":
    image_path = root / "Lavoue_benchmark_model1.png"
    out_dir    = root / "results" / "benchmark"

    print("=" * 60)
    print("Lavoue (2014) Fig. 9(a) - Image Tracing")
    print("=" * 60)

    epsr, sigma = trace_figure(str(image_path), str(out_dir))

    print("-" * 60)
    print("Done.  The traced model can be loaded with:")
    print("  data = np.load('results/benchmark/traced_model.npz')")
    print("  epsr = data['epsr']   # shape (110, 200)")
    print("  sigma = data['sigma'] # shape (110, 200)")
    print("=" * 60)
