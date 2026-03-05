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
Shared plotting utilities: style defaults, colormaps, figure layout (MATLAB-like).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Style defaults — match typical MATLAB GPR figures
# ---------------------------------------------------------------------------
FIGURE_DPI      = 150
FONT_SIZE       = 10
TITLE_FONT_SIZE = 12
COLORMAP_AMP    = "seismic"
COLORMAP_GREY   = "gray"
AXIS_LABEL_FONT = 11


# ---------------------------------------------------------------------------
# Figure / axes helpers
# ---------------------------------------------------------------------------

def setup_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: tuple[float, float] = (8, 5),
    dpi: int = FIGURE_DPI,
) -> tuple[plt.Figure, Any]:
    """Create a figure and axes array with consistent tick-label sizing."""
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    if nrows == 1 and ncols == 1:
        ax = np.array([ax])
    for a in np.atleast_1d(ax).flat:
        a.tick_params(labelsize=FONT_SIZE)
    return fig, ax


def save_figure(fig: plt.Figure, path: str | Path, close: bool = True) -> None:
    """Save *fig* to *path* (creates parent dirs). Closes figure by default."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    if close:
        plt.close(fig)


def plot_image(
    ax: plt.Axes,
    data: np.ndarray,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    xlabel: str = "Position (m)",
    ylabel: str = "Depth (m)",
    title: str = "",
    cmap: str = COLORMAP_GREY,
    aspect: str = "auto",
    clim: tuple[float, float] | None = None,
) -> Any:
    """Plot a 2-D array as an image with physical x/y extent."""
    if x is None:
        x = np.arange(data.shape[1])
    if y is None:
        y = np.arange(data.shape[0])
    extent = [float(x[0]), float(x[-1]), float(y[-1]), float(y[0])] if len(y) > 1 else [float(x[0]), float(x[-1]), 0.0, 1.0]
    im = ax.imshow(data, extent=extent, aspect=aspect, cmap=cmap, clim=clim)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONT)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT)
    if title:
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    return im


def add_colorbar(im: Any, ax: plt.Axes, fig: plt.Figure, label: str = "") -> None:
    """Attach a colour bar to *ax*."""
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    if label:
        cbar.set_label(label, fontsize=FONT_SIZE)


# ---------------------------------------------------------------------------
# PML boundary overlay — shared across build_model and wavefield plots
# ---------------------------------------------------------------------------

def draw_pml_boundary(
    ax: plt.Axes,
    x_inner_min: float,
    x_inner_max: float,
    z_inner_min: float,
    z_inner_max: float,
    n_mark: int = 60,
    color: str = "white",
    linewidth: float = 1.0,
    markersize: float = 3.0,
) -> None:
    """
    Draw the PML inner-domain boundary as a dashed rectangle with 'x' markers.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x_inner_min, x_inner_max : float
        Left/right x-coordinates of the inner (non-PML) domain boundary [m].
    z_inner_min, z_inner_max : float
        Top/bottom z-coordinates of the inner (non-PML) domain boundary [m].
        Depth increases downward so z_inner_min < z_inner_max.
    n_mark : int
        Approximate number of 'x' markers per side.
    color : str
        Line and marker colour (use 'white' on seismic background, 'black' on light).
    linewidth : float
        Dashed-line width.
    markersize : float
        Marker size.
    """
    # Build rectangle: bottom → right → top → left
    xb = np.linspace(x_inner_min, x_inner_max, n_mark); zb = np.full_like(xb, z_inner_max)
    xr = np.full(n_mark, x_inner_max);                   zr = np.linspace(z_inner_max, z_inner_min, n_mark)
    xt = np.linspace(x_inner_max, x_inner_min, n_mark); zt = np.full_like(xt, z_inner_min)
    xl = np.full(n_mark, x_inner_min);                   zl = np.linspace(z_inner_min, z_inner_max, n_mark)

    x_all = np.concatenate([xb, xr, xt, xl])
    z_all = np.concatenate([zb, zr, zt, zl])

    ax.plot(x_all, z_all, "--", color=color, linewidth=linewidth)
    ax.plot(x_all, z_all, "x",  color=color, markersize=markersize, markeredgewidth=0.8)
