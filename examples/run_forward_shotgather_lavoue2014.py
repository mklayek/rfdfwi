# =============================================================================
# RFDFWI - Full-Waveform Inversion (FWI) of GPR Data  |  By Mrinal
#
# Standalone forward shotgather for the Lavoue (2014) benchmark model.
# Loads the traced model, runs FDFD at multiple frequencies, converts
# to time domain via GERMAINE FD2TD, and produces publication-quality
# CSG figures matching the style of Roncoroni et al. (2024).
#
# Copyright (c) Mrinal Kanti Layek
# =============================================================================
"""
run_forward_shotgather_lavoue2014.py
====================================
FDFD shot gather for the Lavoue (2014) realistic subsurface benchmark.

Pipeline
--------
1. Load traced model from ``results/benchmark/traced_model.npz``
2. Build full grid with PML padding (nx=220, nz=130, npml=10)
3. Solve FDFD at nf linspace frequencies (GERMAINE style)
4. FD2TD conversion (Tukey window + Hermitian IFFT)
5. Generate publication-quality CSG figures

Usage
-----
    cd D:/rfdfwi
    python examples/run_forward_shotgather_lavoue2014.py --stag2 --ncpus 15

Output  (all in results/benchmark/shotgather/)
------
    lavoue2014_csg_gray.png           grayscale image CSG
    lavoue2014_csg_wiggle.png         variable-area wiggle CSG
    lavoue2014_csg_combined.png       image + wiggle overlay (paper style)
    lavoue2014_csg_color.png          colour (seismic) CSG
    lavoue2014_shotgather.npz         data file
"""
from __future__ import annotations

import argparse
import sys
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy import signal

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.forward_fdfd import build_helmholtz_2d, solve_forward, MU0
from scripts.plot_cmp import agc2
from create_models.trace_lavoue2014_figure import build_traced_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NX, NZ, NPML, DH = 220, 130, 10, 0.05
N_AIR = 10          # air cells in interior
IZ_GROUND = NPML + N_AIR   # = 20 in full grid
IZ_SRC = 18         # 2 cells above ground (in air)
FC_LOW, FC_HIGH = 1e6, 200e6
NF = 80
A0_CFS = 9.0e8
DPI = 600


# ---------------------------------------------------------------------------
# GERMAINE FD2TD conversion
# ---------------------------------------------------------------------------
def germaine_fd2td(freq_data, nf, df, tmax_td):
    """GERMAINE-style frequency-to-time-domain conversion."""
    ntr = freq_data.shape[1]
    try:
        window = signal.windows.tukey(nf, alpha=1.0)
    except AttributeError:
        window = np.hanning(nf)

    tapered = freq_data.copy()
    for i in range(ntr):
        tapered[:, i] *= window

    S = np.concatenate([
        tapered,
        np.zeros((nf, ntr), dtype=complex),
        np.zeros((nf, ntr), dtype=complex),
        np.flipud(tapered),
    ], axis=0)

    td = np.real(np.fft.ifft(S, axis=0))

    TmaxFD = 0.25 / df
    dt = TmaxFD / nf
    nmaxFD = min(int(tmax_td / dt), td.shape[0])

    sg = td[1:nmaxFD, :]
    t_s = np.arange(1, nmaxFD) * dt
    return sg, t_s, dt


# ---------------------------------------------------------------------------
# Publication-quality CSG plots (Roncoroni et al. 2024 style)
# ---------------------------------------------------------------------------
def _pub_rcparams():
    """Set rcParams to match the paper's clean serif style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


def plot_csg_grayscale(sg, x_m, t_ns, save_path, agc_window=40,
                       clip_pct=99.0, tmax_ns=None, dpi=DPI):
    """Grayscale image CSG (paper style)."""
    _pub_rcparams()
    data = sg.copy().astype(float)
    if agc_window > 0:
        data = -agc2(data, window=agc_window, threshold=0.1)

    vmax = float(np.percentile(np.abs(data), clip_pct))
    if vmax == 0:
        vmax = 1.0

    if tmax_ns is not None:
        mask = t_ns <= tmax_ns
        data = data[mask, :]
        t_plot = t_ns[mask]
    else:
        t_plot = t_ns

    extent = [x_m[0], x_m[-1], t_plot[-1], t_plot[0]]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    ax.imshow(data, extent=extent, aspect="auto", cmap="gray",
              vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Time [ns]")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_csg_wiggle(sg, x_m, t_ns, save_path, agc_window=40,
                    every_nth=2, gain=0.8, clip_pct=99.5,
                    tmax_ns=None, dpi=DPI):
    """Variable-area wiggle CSG (paper style)."""
    _pub_rcparams()
    data = sg.copy().astype(float)
    if agc_window > 0:
        data = -agc2(data, window=agc_window, threshold=0.1)

    clip_val = float(np.percentile(np.abs(data), clip_pct))
    if clip_val > 0:
        data = np.clip(data, -clip_val, clip_val)

    if tmax_ns is not None:
        mask = t_ns <= tmax_ns
        data = data[mask, :]
        t_plot = t_ns[mask]
    else:
        t_plot = t_ns

    trace_idx = np.arange(0, data.shape[1], every_nth)
    x_sel = x_m[trace_idx]
    data_sel = data[:, trace_idx]
    ntr = len(trace_idx)

    dx = float(x_sel[1] - x_sel[0]) if ntr > 1 else 1.0
    wiggle_scale = gain * dx

    for j in range(ntr):
        tmax_tr = np.max(np.abs(data_sel[:, j]))
        if tmax_tr > 0:
            data_sel[:, j] /= tmax_tr

    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    ax.set_facecolor("#e8e8e8")

    for j in range(ntr):
        x0 = x_sel[j]
        trace = data_sel[:, j]
        wiggle = x0 + wiggle_scale * trace
        ax.fill_betweenx(t_plot, x0, wiggle, where=(trace >= 0),
                         color="black", linewidth=0)
        ax.plot(wiggle, t_plot, color="black", linewidth=0.25)

    ax.set_xlim(x_m[0] - dx * 0.3, x_m[-1] + dx * 0.3)
    ax.set_ylim(t_plot[-1], t_plot[0])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Time [ns]")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_csg_combined(sg, x_m, t_ns, save_path, agc_window=40,
                      every_nth=3, gain=0.7, clip_pct=99.0,
                      tmax_ns=None, dpi=DPI):
    """
    Combined grayscale image + variable-area wiggle overlay.
    This is the exact style used in Roncoroni et al. (2024) CSG figures.
    """
    _pub_rcparams()
    data = sg.copy().astype(float)
    if agc_window > 0:
        data = -agc2(data, window=agc_window, threshold=0.1)

    clip_val = float(np.percentile(np.abs(data), clip_pct))
    if clip_val == 0:
        clip_val = 1.0

    if tmax_ns is not None:
        mask = t_ns <= tmax_ns
        data = data[mask, :]
        t_plot = t_ns[mask]
    else:
        t_plot = t_ns

    extent = [x_m[0], x_m[-1], t_plot[-1], t_plot[0]]

    # Wiggle traces
    trace_idx = np.arange(0, data.shape[1], every_nth)
    x_sel = x_m[trace_idx]
    data_sel = data[:, trace_idx].copy()
    ntr = len(trace_idx)
    dx = float(x_sel[1] - x_sel[0]) if ntr > 1 else 1.0
    wiggle_scale = gain * dx

    for j in range(ntr):
        tmax_tr = np.max(np.abs(data_sel[:, j]))
        if tmax_tr > 0:
            data_sel[:, j] /= tmax_tr

    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)

    # Background: grayscale image
    ax.imshow(data, extent=extent, aspect="auto", cmap="gray",
              vmin=-clip_val, vmax=clip_val, interpolation="bilinear")

    # Overlay: variable-area wiggle traces
    for j in range(ntr):
        x0 = x_sel[j]
        trace = data_sel[:, j]
        wiggle = x0 + wiggle_scale * trace
        ax.fill_betweenx(t_plot, x0, wiggle, where=(trace >= 0),
                         color="black", linewidth=0, alpha=0.85)
        ax.plot(wiggle, t_plot, color="black", linewidth=0.2)

    ax.set_xlim(x_m[0], x_m[-1])
    ax.set_ylim(t_plot[-1], t_plot[0])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Time [ns]")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_csg_color(sg, x_m, t_ns, save_path, agc_window=40,
                   clip_pct=99.0, tmax_ns=None, dpi=DPI):
    """Colour (seismic) CSG with colorbar."""
    _pub_rcparams()
    data = sg.copy().astype(float)
    if agc_window > 0:
        data = -agc2(data, window=agc_window, threshold=0.1)

    vmax = float(np.percentile(np.abs(data), clip_pct))
    if vmax == 0:
        vmax = 1.0

    if tmax_ns is not None:
        mask = t_ns <= tmax_ns
        data = data[mask, :]
        t_plot = t_ns[mask]
    else:
        t_plot = t_ns

    extent = [x_m[0], x_m[-1], t_plot[-1], t_plot[0]]
    fig, ax = plt.subplots(figsize=(7, 6), dpi=dpi)
    im = ax.imshow(data, extent=extent, aspect="auto", cmap="seismic",
                   vmin=-vmax, vmax=vmax, interpolation="bilinear")
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("Normalized Amplitude", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Time [ns]")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(
        description="FDFD shot gather for Lavoue (2014) benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stag1", dest="grid_style", action="store_const",
                   const="stag1", default="stag2")
    p.add_argument("--stag2", dest="grid_style", action="store_const",
                   const="stag2")
    p.add_argument("--ncpus", type=int, default=1)
    p.add_argument("--src-ix", type=int, default=110,
                   help="Source x-index (full grid).")
    p.add_argument("--nf", type=int, default=NF)
    p.add_argument("--display-tmax-ns", type=float, default=300.0,
                   help="Display time window [ns].")
    p.add_argument("--agc-window", type=int, default=40)
    p.add_argument("--no-agc", action="store_true", default=False)
    p.add_argument("--wiggle-gain", type=float, default=2.0)
    p.add_argument("--wiggle-every", type=int, default=2,
                   help="Plot every Nth trace in wiggle plots.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = _parse_args()
    grid_style = args.grid_style
    n_workers = args.ncpus
    src_ix = args.src_ix
    nf = args.nf
    display_tmax_ns = args.display_tmax_ns
    tmax_td = display_tmax_ns * 1e-9
    agc_win = 0 if args.no_agc else args.agc_window

    out_dir = root / "results" / "benchmark" / "shotgather"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load traced model ------------------------------------------------
    npz_path = root / "results" / "benchmark" / "traced_model.npz"
    data = np.load(str(npz_path))
    epsr_int = data["epsr"]      # (110, 200) interior
    sigma_int = data["sigma"]

    print("=" * 60)
    print("Lavoue (2014) Benchmark - Shot Gather")
    print("=" * 60)
    print(f"  Traced model : {npz_path}")
    print(f"  Interior     : {epsr_int.shape}")
    print(f"  epsr range   : [{epsr_int.min():.1f}, {epsr_int.max():.1f}]")
    print(f"  sigma range  : [{sigma_int.min()*1e3:.2f}, "
          f"{sigma_int.max()*1e3:.1f}] mS/m")

    # ---- Build full grid with PML -----------------------------------------
    epsr, sigma = build_traced_model(epsr_int, sigma_int, NX, NZ, NPML)
    print(f"  Full grid    : ({NZ}, {NX}) with PML={NPML}")

    # ---- Receivers: 101 receivers at iz=18, ix=10..210, step=2 -------------
    rec_iz = IZ_SRC
    rec_indices = np.arange(NPML, NX - NPML, 2)   # ix = 10,12,...,208
    ntr = len(rec_indices)
    x_m = rec_indices.astype(float) * DH

    # ---- Frequency setup ---------------------------------------------------
    freqs = np.linspace(FC_LOW, FC_HIGH, nf)
    df = (FC_HIGH - FC_LOW) / (nf - 1) if nf > 1 else FC_HIGH

    TmaxFD = 0.25 / df
    dt_fd = TmaxFD / nf
    nmaxFD = int(tmax_td / dt_fd)

    print(f"  Source       : ix={src_ix} ({src_ix*DH:.2f}m)  iz={IZ_SRC}")
    print(f"  Receivers    : iz={rec_iz}  ix=[{rec_indices[0]}..{rec_indices[-1]}]"
          f"  step=2  ntr={ntr}")
    print(f"  Freq         : {FC_LOW/1e6:.1f}-{FC_HIGH/1e6:.0f} MHz"
          f"  nf={nf}  df={df/1e6:.3f} MHz")
    print(f"  GERMAINE     : TmaxFD={TmaxFD*1e9:.1f} ns"
          f"  dt={dt_fd*1e9:.3f} ns  nmaxFD={nmaxFD}")
    print(f"  Display      : 0-{display_tmax_ns:.0f} ns")
    print(f"  AGC          : {'off' if agc_win == 0 else f'window={agc_win}'}")
    print(f"  Grid style   : {grid_style}")
    print(f"  Workers      : {n_workers}")

    # ==================================================================
    # FDFD solve
    # ==================================================================
    freq_data = np.zeros((nf, ntr), dtype=complex)

    def _solve_one(fi):
        f_hz = freqs[fi]
        omega = 2.0 * np.pi * f_hz
        A = build_helmholtz_2d(epsr, sigma, DH, omega, NPML,
                               a0_cfs=A0_CFS, grid_style=grid_style)
        src_amp = -(omega * MU0 * 1j) / DH**2
        Ez = solve_forward(A, src_ix, IZ_SRC, NX, NZ,
                           source_amplitude=src_amp)
        row = Ez[rec_iz, rec_indices]
        return fi, row

    print(f"\nSolving {nf} FDFD systems ...")

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for fi, row in ex.map(_solve_one, range(nf)):
                freq_data[fi, :] = row
                print(f"  [{fi+1:3d}/{nf}]  f={freqs[fi]/1e6:7.2f} MHz  "
                      f"|E|={np.max(np.abs(row)):.3e}")
    else:
        for fi in range(nf):
            _, row = _solve_one(fi)
            freq_data[fi, :] = row
            print(f"  [{fi+1:3d}/{nf}]  f={freqs[fi]/1e6:7.2f} MHz  "
                  f"|E|={np.max(np.abs(row)):.3e}")

    # ==================================================================
    # GERMAINE FD2TD conversion
    # ==================================================================
    print("\nGERMAINE FD2TD conversion ...")
    sg, t_s, dt_actual = germaine_fd2td(freq_data, nf, df, tmax_td)
    t_ns = t_s * 1e9

    sg_max = np.max(np.abs(sg))
    if sg_max > 0:
        sg = sg / sg_max
    print(f"  Shape: {sg.shape}  t=[{t_ns[0]:.2f}, {t_ns[-1]:.2f}] ns")
    print(f"  dt={dt_actual*1e9:.3f} ns  max_before_norm={sg_max:.3e}")

    # ---- Save data --------------------------------------------------------
    npz_out = out_dir / "lavoue2014_shotgather.npz"
    np.savez(npz_out, sg=sg, x_m=x_m, t_ns=t_ns, freqs=freqs,
             dh=DH, src_ix=src_ix, src_iz=IZ_SRC, rec_iz=rec_iz,
             rec_indices=rec_indices)
    print(f"  Data -> {npz_out}")

    # ==================================================================
    # Generate figures - existing style (from run_forward_shotgather.py)
    # ==================================================================
    print("\nGenerating figures ...")
    mn = "lavoue2014_shotgather"
    src_pos_m = float(src_ix * DH)
    x_label = "Distance (m)"
    tmax_disp = display_tmax_ns

    # ---- 1. Color plot (seismic_r) ----
    from scripts.plot_shotgather import plot_shotgather_color, plot_shotgather_wiggle

    f_color = out_dir / f"{mn}_color.png"
    plot_shotgather_color(
        sg, x_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f_color, src_pos_m=src_pos_m,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
    )
    print(f"  Color       -> {f_color}")

    # ---- 2. Grayscale plot ----
    f_gray = out_dir / f"{mn}_gray.png"
    plot_shotgather_color(
        sg, x_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f_gray, src_pos_m=src_pos_m,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
        cmap_name="gray",
        title="Synthetic GPR Shot Gather (grayscale)",
        cbar_label="Normalized Amplitude",
    )
    print(f"  Gray        -> {f_gray}")

    # ---- 3. Standard wiggle plot (mild AGC) ----
    f_wiggle = out_dir / f"{mn}_wiggle.png"
    plot_shotgather_wiggle(
        sg, x_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f_wiggle, src_pos_m=src_pos_m,
        gain=args.wiggle_gain, agc_window=agc_win, x_label=x_label,
        upsample=1, every_nth=args.wiggle_every,
    )
    print(f"  Wiggle      -> {f_wiggle}")

    # ---- 4. Seismic-style wiggle (red/blue, mild AGC) ----
    # Import from the main shotgather script
    from examples.run_forward_shotgather import plot_shotgather_seismic_wiggle

    f_seis = out_dir / f"{mn}_seismic_wiggle.png"
    plot_shotgather_seismic_wiggle(
        sg, x_m, t_ns, save_path=f_seis,
        src_pos_m=src_pos_m,
        agc_window=agc_win, agc_threshold=0.1,
        every_nth=args.wiggle_every, gain=args.wiggle_gain,
        x_label=x_label,
    )
    print(f"  Seismic     -> {f_seis}")

    # ---- 5. Model with source & receiver geometry ----
    from examples.run_forward_shotgather import plot_model_with_geometry

    f_model = out_dir / f"{mn}_model.png"
    plot_model_with_geometry(
        epsr, sigma, NX, NZ, NPML, DH,
        src_ix, IZ_SRC, rec_indices, rec_iz,
        save_path=f_model,
    )
    print(f"  Model       -> {f_model}")

    # ==================================================================
    # Publication-quality CSG figures (Roncoroni et al. 2024 style)
    # ==================================================================
    print("\nGenerating paper-style CSG figures ...")

    # ---- 6. Grayscale image CSG ----
    p1 = out_dir / "lavoue2014_csg_gray.png"
    plot_csg_grayscale(sg, x_m, t_ns, p1, agc_window=agc_win,
                       tmax_ns=tmax_disp)
    print(f"  CSG Gray    -> {p1}")

    # ---- 7. Variable-area wiggle CSG ----
    p2 = out_dir / "lavoue2014_csg_wiggle.png"
    plot_csg_wiggle(sg, x_m, t_ns, p2, agc_window=agc_win,
                    tmax_ns=tmax_disp)
    print(f"  CSG Wiggle  -> {p2}")

    # ---- 8. Combined image + wiggle overlay (paper style) ----
    p3 = out_dir / "lavoue2014_csg_combined.png"
    plot_csg_combined(sg, x_m, t_ns, p3, agc_window=agc_win,
                      tmax_ns=tmax_disp)
    print(f"  CSG Combined-> {p3}")

    # ---- 9. Colour (seismic) CSG ----
    p4 = out_dir / "lavoue2014_csg_color.png"
    plot_csg_color(sg, x_m, t_ns, p4, agc_window=agc_win,
                   tmax_ns=tmax_disp)
    print(f"  CSG Color   -> {p4}")

    print("\n" + "=" * 60)
    print("Done. 9 figures saved to", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
