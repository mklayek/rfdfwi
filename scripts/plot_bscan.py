"""
B-scan plotting: position vs depth/time, amplitude (real or envelope).
Clean layout inspired by standard GPR visualization (e.g. gprMax-style).
"""
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from .plot_utils import setup_figure, save_figure, plot_image, add_colorbar, FIGURE_DPI


def _ensure_2d(traces: np.ndarray) -> np.ndarray:
    """traces: (n_positions,) or (n_positions, n_samples). For 1D single trace, treat as (1, n_samples)."""
    if traces.ndim == 1:
        return traces.reshape(1, -1)
    return traces


def _amplitude_for_display(traces: np.ndarray, mode: str = "real") -> np.ndarray:
    """Convert complex traces to displayable 2D. mode: 'real', 'imag', 'amplitude', 'envelope'."""
    traces = _ensure_2d(traces)
    if np.iscomplexobj(traces):
        if mode == "real":
            out = np.real(traces)
        elif mode == "imag":
            out = np.imag(traces)
        elif mode in ("amplitude", "envelope"):
            out = np.abs(traces)
        else:
            out = np.real(traces)
    else:
        out = traces
    return out


def plot_bscan(
    traces: np.ndarray,
    dx: float = 0.01,
    dz: float = 0.01,
    x_start: float = 0.0,
    z_start: float = 0.0,
    mode: str = "real",
    xlabel: str = "Position (m)",
    ylabel: str = "Depth (m)",
    title: str = "B-scan",
    clim: tuple[float, float] | None = None,
    cmap: str = "gray",
    figsize: tuple[float, float] = (10, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot B-scan: horizontal axis = position, vertical axis = depth (or time).
    traces: (n_positions, n_samples) or (n_positions,) complex or real.
    """
    data = _amplitude_for_display(traces, mode=mode)
    n_pos, n_samp = data.shape
    x = x_start + np.arange(n_pos) * dx
    z = z_start + np.arange(n_samp) * dz
    if clim is None:
        v = np.abs(data).max()
        clim = (-v, v) if mode in ("real", "imag") else (0, v)
    fig, ax = setup_figure(1, 1, figsize=figsize)
    ax = ax[0]
    im = plot_image(
        ax, data,
        x=x, y=z,
        xlabel=xlabel, ylabel=ylabel, title=title,
        cmap=cmap, clim=clim,
    )
    add_colorbar(im, ax, fig, label="Amplitude")
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_bscan_comparison(
    observed: np.ndarray,
    synthetic: np.ndarray,
    dx: float = 0.01,
    dz: float = 0.01,
    x_start: float = 0.0,
    z_start: float = 0.0,
    titles: tuple[str, str] = ("Observed", "Synthetic"),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Two-panel B-scan: observed vs synthetic."""
    obs = _amplitude_for_display(observed, "real")
    syn = _amplitude_for_display(synthetic, "real")
    v = max(np.abs(obs).max(), np.abs(syn).max())
    clim = (-v, v)
    n_pos, n_samp = obs.shape
    x = x_start + np.arange(n_pos) * dx
    z = z_start + np.arange(n_samp) * dz
    fig, axes = setup_figure(1, 2, figsize=(12, 5))
    for ax, data, t in zip(axes, [obs, syn], titles):
        im = plot_image(ax, data, x=x, y=z, title=t, cmap="gray", clim=clim)
        add_colorbar(im, ax, fig, label="Amplitude")
    if save_path:
        save_figure(fig, save_path)
    return fig


if __name__ == "__main__":
    # Demo: synthetic B-scan
    n_pos, n_samp = 50, 80
    t = np.linspace(0, 1, n_samp)
    traces = np.random.randn(n_pos, n_samp) * np.exp(-t)
    plot_bscan(traces, dx=0.02, dz=0.01, title="B-scan (demo)", save_path="output/bscan_demo.png")
