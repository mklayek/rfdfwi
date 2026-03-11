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
    agc_threshold: float = 0.0,
    upsample: int    = 10,
    clip_pct: float  = 99.0,
    clip_frac: float | None = None,
    dpi: int         = SHOTGATHER_DPI,
    x_label: str     = "Offset [m]",
    cmap_name: str   = "seismic_r",
    title: str       = "Synthetic GPR Shot Gather",
    cbar_label: str  = "Normalized Amplitude",
) -> None:
    """
    Publishable colour (image) shot gather.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    if upsample > 1:
        data_up = spline_upsample(sg, factor=upsample)
    else:
        data_up = sg.copy().astype(float)

    nt_up, ntr_up = data_up.shape
    t_up  = np.linspace(t_ns[0],      t_ns[-1],      nt_up)
    x_up  = np.linspace(offsets_m[0], offsets_m[-1], ntr_up)

    if agc_window > 0:
        data_plot = -agc2(data_up, window=agc_window, threshold=agc_threshold)
    else:
        data_plot = data_up

    if clip_frac is not None and clip_frac > 0:
        vmax = float(np.max(np.abs(data_plot))) * clip_frac
    else:
        vmax = float(np.percentile(np.abs(data_plot), clip_pct))
    vmax = vmax if vmax > 0 else 1.0

    extent = [x_up[0], x_up[-1], t_up[-1], t_up[0]]

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    im = ax.imshow(
        data_plot,
        extent=extent,
        aspect="auto",
        cmap=cmap_name,
        vmin=-vmax, vmax=vmax,
        interpolation="bilinear",
    )

    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label(cbar_label, fontsize=14)
    cb.ax.tick_params(labelsize=12)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Time (ns)")
    ax.set_title(title, pad=8)

    if "Distance" in x_label and src_pos_m is not None:
        ax.axvline(src_pos_m, color="white", linewidth=0.8, linestyle="--", alpha=0.6)
    else:
        ax.axvline(0.0, color="white", linewidth=0.8, linestyle="--", alpha=0.6)

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
    agc_window: int  = 0,
    upsample: int    = 1,
    gain: float      = 0.8,
    every_nth: int   = 3,
    clip_percentile: float = 99.5,
    dpi: int         = SHOTGATHER_DPI,
    x_label: str     = "Offset [m]",
) -> None:
    """Publishable wiggle-trace shot gather — clean thin traces matching grayscale."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    if upsample > 1:
        data_up = spline_upsample(sg, factor=upsample)
    else:
        data_up = sg.copy().astype(float)

    nt_up, ntr_up = data_up.shape
    t_up  = np.linspace(t_ns[0],  t_ns[-1],      nt_up)
    x_up  = np.linspace(offsets_m[0], offsets_m[-1], ntr_up)

    if agc_window > 0:
        data_plot = -agc2(data_up, window=agc_window)
    else:
        data_plot = data_up

    # Clip amplitudes at percentile to prevent thick saturation
    clip_val = float(np.percentile(np.abs(data_plot), clip_percentile))
    if clip_val > 0:
        data_plot = np.clip(data_plot, -clip_val, clip_val)

    # Select every nth trace for clean readability
    trace_idx = np.arange(0, ntr_up, every_nth)
    x_sel = x_up[trace_idx]
    data_sel = data_plot[:, trace_idx]
    ntr = len(trace_idx)

    dx = float(x_sel[1] - x_sel[0]) if ntr > 1 else 1.0
    wiggle_scale = gain * dx

    # Per-trace normalization so all traces have same visual weight
    for j in range(ntr):
        tmax = np.max(np.abs(data_sel[:, j]))
        if tmax > 0:
            data_sel[:, j] /= tmax

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    for j in range(ntr):
        x0    = x_sel[j]
        trace = data_sel[:, j]
        wiggle = x0 + wiggle_scale * trace

        ax.plot(wiggle, t_up, color="black", linewidth=0.3)
        ax.fill_betweenx(t_up, x0, wiggle,
                         where=(trace >= 0),
                         color="black", alpha=0.6, linewidth=0)

    if "Distance" in x_label and src_pos_m is not None:
        ax.axvline(src_pos_m, color="red", linewidth=1.0, linestyle="--",
                   alpha=0.7)
    else:
        ax.axvline(0.0, color="red", linewidth=1.0, linestyle="--",
                   alpha=0.7)
    ax.set_xlim(x_up[0] - dx * 0.5, x_up[-1] + dx * 0.5)
    ax.set_ylim(t_up[-1], t_up[0])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time (ns)")
    ax.set_title("Synthetic GPR Shot Gather (wiggle)", pad=8)
    ax.grid(True, linewidth=0.2, color="gray", alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
