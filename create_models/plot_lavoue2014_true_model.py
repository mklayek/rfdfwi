# =============================================================================
# RFDFWI - Full-Waveform Inversion (FWI) of GPR Data  |  By Mrinal
#
# Plot the Lavoue (2014) Fig. 2.14 true models (permittivity & conductivity)
# from the traced model data.  Style matches the reference figure exactly.
#
# Copyright (c) Mrinal Kanti Layek
# =============================================================================
"""
plot_lavoue2014_true_model.py
=============================
Generate publication-quality true-model plots matching Lavoue (2014)
Fig. 2.14 panels (a) eps_r and (c) sigma from the traced model.

Usage
-----
    cd D:/rfdfwi
    python create_models/plot_lavoue2014_true_model.py

Output  (in results/benchmark/)
------
    lavoue2014_true_model.png    2-panel figure (epsr + sigma)
"""
from __future__ import annotations

import os
import sys
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # ---- Load traced model ------------------------------------------------
    npz_path = root / "results" / "benchmark" / "traced_model.npz"
    data = np.load(str(npz_path))
    epsr  = data["epsr"]       # (110, 200)
    sigma = data["sigma"]      # (110, 200)
    dh    = float(data["dh"])  # 0.05 m

    nz_int, nx_int = epsr.shape
    print(f"Loaded: {npz_path}")
    print(f"  epsr : shape {epsr.shape}, "
          f"range [{epsr.min():.1f}, {epsr.max():.1f}]")
    print(f"  sigma: shape {sigma.shape}, "
          f"range [{sigma.min()*1e3:.2f}, {sigma.max()*1e3:.1f}] mS/m")

    # ---- Physical coordinates ---------------------------------------------
    # Interior grid: 110 rows x 200 cols, dh=0.05 m
    # First 10 rows = air (0.5 m above ground)
    # Depth starts at 0 m (ground surface).
    n_air = 10
    x_max = nx_int * dh                       # 10.0 m
    z_surface = 0.0                            # ground surface
    z_top = z_surface - n_air * dh             # -0.5 m (top of air)
    z_bot = (nz_int - n_air) * dh              # 5.0 m  (bottom)

    extent = [0, x_max, z_bot, z_top]          # imshow: [L, R, bottom, top]

    # ---- Create figure (2 rows x 1 col) -----------------------------------
    fig, (ax_a, ax_c) = plt.subplots(
        2, 1, figsize=(7.5, 7.0), sharex=False,
        gridspec_kw={"hspace": 0.30})

    # ========== Panel (a): True permittivity ================================
    im_a = ax_a.imshow(
        epsr, extent=extent, cmap="RdBu_r",
        vmin=1, vmax=31, aspect="auto", interpolation="bicubic")

    ax_a.set_ylabel("Depth (m)", fontsize=11)
    ax_a.set_yticks([0, 1, 2, 3, 4, 5])

    # x-axis on TOP (matching reference Fig. 2.14a)
    ax_a.xaxis.set_label_position("top")
    ax_a.xaxis.tick_top()
    ax_a.set_xlabel("Distance (m)", fontsize=11)
    ax_a.set_xticks([0, 5, 10])

    # Colorbar
    cb_a = fig.colorbar(im_a, ax=ax_a, shrink=0.85, pad=0.02)
    cb_a.set_ticks([1, 11, 21, 31])
    cb_a.set_label("Relative permittivity", fontsize=10)

    # ========== Panel (c): True conductivity ================================
    im_c = ax_c.imshow(
        sigma * 1e3, extent=extent, cmap="RdBu_r",
        vmin=0, vmax=20, aspect="auto", interpolation="bicubic")

    ax_c.set_ylabel("Depth (m)", fontsize=11)
    ax_c.set_yticks([0, 1, 2, 3, 4, 5])

    # x-axis on TOP (matching reference Fig. 2.14c)
    ax_c.xaxis.set_label_position("top")
    ax_c.xaxis.tick_top()
    ax_c.set_xlabel("Distance (m)", fontsize=11)
    ax_c.set_xticks([0, 5, 10])

    # Colorbar
    cb_c = fig.colorbar(im_c, ax=ax_c, shrink=0.85, pad=0.02)
    cb_c.set_ticks([0, 5, 10, 15, 20])
    cb_c.set_label("Conductivity (mS/m)", fontsize=10)

    # ---- Save --------------------------------------------------------------
    out_dir = root / "results" / "benchmark"
    os.makedirs(str(out_dir), exist_ok=True)
    out_path = out_dir / "lavoue2014_true_model.png"
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {out_path}")


if __name__ == "__main__":
    main()
