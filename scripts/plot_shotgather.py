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
Shot gather processing and visualization — GPRFM style.

FD→TD algorithm is identical to CMP (Hermitian IFFT via freq_to_timedomain).
Post-processing matches GPRFM.m shot gather section:

    data = interp2(X, Y, data, X2, Y2, 'spline')  ← 10× spline upsample
    outData = -agc2(outData, 60, 10)               ← AGC + polarity flip
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plot_cmp import agc2, spline_upsample, GPRFM_FREQS_HZ   # noqa: F401

SHOTGATHER_DPI = 600


# ---------------------------------------------------------------------------
# Colour shot gather plot  (GPRFM style)
# ---------------------------------------------------------------------------

def plot_shotgather_color(
    sg: np.ndarray,
    offsets_m: np.ndarray,
    t_ns: np.ndarray,
    grid_style: str,
    freqs_hz: np.ndarray,
    save_path: Path,
    src_pos_m: float | None = None,
    agc_window: int  = 60,
    upsample: int    = 10,
    clip_pct: float  = 99.0,
    dpi: int         = SHOTGATHER_DPI,
) -> None:
    """
    Colour (image) shot gather — full GPRFM post-processing pipeline:

        1. 10× spline upsample  (GPRFM: interp2 'spline')
        2. AGC2 + polarity flip (GPRFM: outData = -agc2(outData, 60, 10))
        3. imagesc with seismic colormap, percentile colour limits

    Parameters
    ----------
    sg          : (nt, ntr)   Real time-domain shot gather.
    offsets_m   : (ntr,)      Signed receiver offsets from source [m].
                              Negative = left of source, positive = right.
    t_ns        : (nt,)       Time axis [ns].
    grid_style  : str         Stencil label for title.
    freqs_hz    : (nf,)       Frequencies used [Hz].
    save_path   : Path        Output file path.
    src_pos_m   : float|None  Source absolute x-position [m] for title.
    """
    if upsample > 1:
        data_up = spline_upsample(sg, factor=upsample)
    else:
        data_up = sg.copy().astype(float)

    nt_up, ntr_up = data_up.shape
    t_up  = np.linspace(t_ns[0],      t_ns[-1],      nt_up)
    x_up  = np.linspace(offsets_m[0], offsets_m[-1], ntr_up)

    data_agc = -agc2(data_up, window=agc_window)

    vmax = float(np.percentile(np.abs(data_agc), clip_pct))
    vmax = vmax if vmax > 0 else 1.0

    extent = [x_up[0], x_up[-1], t_up[-1], t_up[0]]
    fc_lo  = freqs_hz[0]  / 1e6
    fc_hi  = freqs_hz[-1] / 1e6

    src_str = f"  src={src_pos_m:.2f} m" if src_pos_m is not None else ""
    title   = f"Shot gather  {grid_style}  ({fc_lo:.0f}–{fc_hi:.0f} MHz){src_str}"

    fig, ax = plt.subplots(figsize=(10, 8), dpi=dpi)

    im = ax.imshow(
        data_agc,
        extent=extent,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax, vmax=vmax,
        interpolation="bilinear",
    )

    ax.set_xlabel("Offset [m]",              fontsize=14)
    ax.set_ylabel("Two-way travel time [ns]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=14, pad=6)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    # Mark zero offset (source position) with a dashed vertical line
    ax.axvline(0.0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Re(Ey)  [norm.]", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Wiggle shot gather plot  (GPRFM style)
# ---------------------------------------------------------------------------

def plot_shotgather_wiggle(
    sg: np.ndarray,
    offsets_m: np.ndarray,
    t_ns: np.ndarray,
    grid_style: str,
    freqs_hz: np.ndarray,
    save_path: Path,
    src_pos_m: float | None = None,
    agc_window: int  = 60,
    upsample: int    = 10,
    gain: float      = 1.5,
    dpi: int         = SHOTGATHER_DPI,
) -> None:
    """
    Wiggle-trace shot gather — same GPRFM post-processing as colour plot
    (spline upsample → AGC2 + polarity flip), then wiggle display with
    positive-amplitude fill.

    Parameters
    ----------
    sg          : (nt, ntr)   Real time-domain shot gather.
    offsets_m   : (ntr,)      Signed receiver offsets from source [m].
    gain        : float       Wiggle scale relative to trace spacing.
    """
    if upsample > 1:
        data_up = spline_upsample(sg, factor=upsample)
    else:
        data_up = sg.copy().astype(float)

    nt_up, ntr_up = data_up.shape
    t_up  = np.linspace(t_ns[0],      t_ns[-1],      nt_up)
    x_up  = np.linspace(offsets_m[0], offsets_m[-1], ntr_up)
    dx    = float(x_up[1] - x_up[0]) if ntr_up > 1 else 1.0

    data_agc     = -agc2(data_up, window=agc_window)
    wiggle_scale = gain * dx

    fc_lo   = freqs_hz[0]  / 1e6
    fc_hi   = freqs_hz[-1] / 1e6
    src_str = f"  src={src_pos_m:.2f} m" if src_pos_m is not None else ""
    title   = f"Shot gather wiggle  {grid_style}  ({fc_lo:.0f}–{fc_hi:.0f} MHz){src_str}"

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    for col in range(ntr_up):
        x0     = x_up[col]
        trace  = data_agc[:, col]
        wiggle = x0 + wiggle_scale * trace

        ax.plot(wiggle, t_up, color="black", linewidth=0.3, alpha=0.8)
        ax.fill_betweenx(t_up, x0, wiggle,
                         where=(trace >= 0),
                         color="black", alpha=0.7, linewidth=0)

    ax.axvline(0.0, color="red", linewidth=0.8, linestyle="--", alpha=0.7,
               label="source")
    ax.set_xlim(x_up[0] - dx, x_up[-1] + dx)
    ax.set_ylim(t_up[-1], t_up[0])
    ax.set_xlabel("Offset [m]",              fontsize=14)
    ax.set_ylabel("Two-way travel time [ns]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(title, fontsize=14, pad=6)
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
