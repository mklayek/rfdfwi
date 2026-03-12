# =============================================================================
# RFDFWI - Full-Waveform Inversion (FWI) of GPR Data  |  By Mrinal
#
# Reproduce Lavoue (2014) Figure 2.15(a): time-domain shot gather for the
# subsurface benchmark, source at x = 0 m, using the time-derivative of a
# Ricker wavelet (fc = 100 MHz) convolved in the frequency domain.
#
# Copyright (c) Mrinal Kanti Layek
# =============================================================================
"""
run_shotgather_lavoue2014_fig215.py
====================================
Reproduce Lavoue (2014) Fig. 2.15 shot gathers exactly.

Key differences from GERMAINE FD2TD:
  - Source wavelet: time-derivative of Ricker (fc=100 MHz), NOT Tukey window
  - Source position: x = 0 m (left edge of domain)
  - X-axis: Offset (m)
  - Time window: 0-150 ns

Algorithm (from the thesis, Section 2.2.3.1):
  1. Solve FDFD at nf frequencies -> Green's function response G(f)
  2. Multiply by Ricker-derivative spectrum:  D(f) = G(f) * W(f)
     where W(f) = j*2*pi*f * R(f)  and  R(f) = (f/fc)^2 * exp(-(f/fc)^2)
  3. Build Hermitian spectrum and IFFT to time domain

Usage
-----
    cd D:/rfdfwi
    python examples/run_shotgather_lavoue2014_fig215.py --stag2 --ncpus 15

Output  (in results/benchmark/shotgather/)
------
    lavoue2014_fig215_shotgather.png     Fig. 2.15(a) replica
    lavoue2014_fig215_shotgather.npz     data file
"""
from __future__ import annotations

import argparse
import sys
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

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
# Grid and acquisition (Lavoue 2014, Section 2.2.3)
# ---------------------------------------------------------------------------
NX, NZ, NPML, DH = 220, 130, 10, 0.05
N_AIR       = 10
IZ_GROUND   = NPML + N_AIR     # = 20
IZ_SRC_REC  = 18                # 2 cells above ground (z = -0.1 m)
A0_CFS      = 9.0e8
FC_RICKER   = 100e6             # Ricker central frequency
DPI         = 600


# ---------------------------------------------------------------------------
# Ricker-derivative FD2TD  (Lavoue 2014 style)
# ---------------------------------------------------------------------------
def ricker_deriv_fd2td(freq_data, freqs, fc, tmax_td):
    """
    Lavoue (2014) style FD-to-TD conversion.

    Convolve with the time-derivative of a Ricker wavelet in the
    frequency domain, then Hermitian IFFT.

    Parameters
    ----------
    freq_data : (nf, ntr) complex  -- FDFD receiver responses (Green's fn)
    freqs     : (nf,)              -- frequency samples [Hz]
    fc        : float              -- Ricker peak frequency [Hz]
    tmax_td   : float              -- desired max time [s]

    Returns
    -------
    sg, t_s, dt
    """
    from scipy import signal

    nf, ntr = freq_data.shape

    # --- Ricker wavelet spectrum ---
    # R(f) = (f/fc)^2 * exp(-(f/fc)^2)   (normalisation cancels out)
    f_ratio = freqs / fc
    ricker_spec = f_ratio**2 * np.exp(-f_ratio**2)

    # --- Time derivative in freq domain: multiply by j*omega ---
    omega = 2.0 * np.pi * freqs
    wavelet_spec = 1j * omega * ricker_spec       # (nf,) complex

    # --- Hann window to taper band edges (prevents Gibbs ringing) ---
    try:
        hann = signal.windows.tukey(nf, alpha=1.0)
    except AttributeError:
        hann = np.hanning(nf)

    # --- Convolve: multiply each trace by wavelet spectrum * taper ---
    convolved = freq_data.copy()
    for i in range(ntr):
        convolved[:, i] *= wavelet_spec * hann

    # --- GERMAINE-style Hermitian IFFT: [E, 0, 0, flipud(E)] ---
    S = np.concatenate([
        convolved,
        np.zeros((nf, ntr), dtype=complex),
        np.zeros((nf, ntr), dtype=complex),
        np.flipud(convolved),
    ], axis=0)

    td = np.real(np.fft.ifft(S, axis=0))

    # --- Time axis ---
    df = (freqs[-1] - freqs[0]) / (nf - 1) if nf > 1 else freqs[-1]
    TmaxFD = 0.25 / df
    dt = TmaxFD / nf
    nmaxFD = min(int(tmax_td / dt), td.shape[0])

    sg = td[1:nmaxFD, :]
    t_s = np.arange(1, nmaxFD) * dt
    return sg, t_s, dt


# ---------------------------------------------------------------------------
# Figure 2.15 style plot
# ---------------------------------------------------------------------------
def plot_fig215(sg, offsets_m, t_ns, save_path, tmax_ns=150.0,
                agc_window=40, clip_pct=99.0, dpi=DPI):
    """
    Reproduce the exact style of Lavoue (2014) Fig. 2.15(a).

    Grayscale image display, x-axis = Offset (m) at top,
    y-axis = Time (ns) increasing downward.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })

    data = sg.copy().astype(float)

    # AGC to equalise amplitudes at depth (reveals deeper reflections)
    if agc_window > 0:
        data = -agc2(data, window=agc_window, threshold=0.1)

    # Clip to time window
    mask = t_ns <= tmax_ns
    data = data[mask, :]
    t_plot = t_ns[mask]

    vmax = float(np.percentile(np.abs(data), clip_pct))
    if vmax == 0:
        vmax = 1.0

    extent = [offsets_m[0], offsets_m[-1], t_plot[-1], t_plot[0]]

    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=dpi)

    ax.imshow(data, extent=extent, aspect="auto", cmap="gray",
              vmin=-vmax, vmax=vmax, interpolation="bilinear")

    # X-axis at top (matching paper)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("Offset (m)")
    ax.set_ylabel("Time (ns)")

    # Match paper tick spacing
    ax.set_xticks([0, 5, 10])
    ax.set_yticks(np.arange(0, tmax_ns + 1, 50))

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(
        description="Reproduce Lavoue (2014) Fig. 2.15 shot gather.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stag1", dest="grid_style", action="store_const",
                   const="stag1", default="stag2")
    p.add_argument("--stag2", dest="grid_style", action="store_const",
                   const="stag2")
    p.add_argument("--ncpus", type=int, default=1)
    p.add_argument("--nf", type=int, default=80,
                   help="Number of frequencies.")
    p.add_argument("--fc-low", type=float, default=1e6,
                   help="Low frequency [Hz].")
    p.add_argument("--fc-high", type=float, default=200e6,
                   help="High frequency [Hz].")
    p.add_argument("--fc-ricker", type=float, default=FC_RICKER,
                   help="Ricker wavelet central frequency [Hz].")
    p.add_argument("--tmax-ns", type=float, default=150.0,
                   help="Display time window [ns].")
    p.add_argument("--agc-window", type=int, default=40)
    p.add_argument("--no-agc", action="store_true", default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = _parse_args()
    grid_style = args.grid_style
    n_workers  = args.ncpus
    nf         = args.nf
    fc_low     = args.fc_low
    fc_high    = args.fc_high
    fc_ricker  = args.fc_ricker
    tmax_ns    = args.tmax_ns
    tmax_td    = tmax_ns * 1e-9
    agc_win    = 0 if args.no_agc else args.agc_window

    out_dir = root / "results" / "benchmark" / "shotgather"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Source at x = 0 m (left edge of interior) -----------------------
    src_ix = NPML          # ix = 10  ->  x = 0.5 m in full grid = 0 m interior
    src_iz = IZ_SRC_REC

    # ---- Receivers across entire interior at same depth ------------------
    rec_iz = IZ_SRC_REC
    rec_indices = np.arange(NPML, NX - NPML, 2)   # ix = 10,12,...,208
    ntr = len(rec_indices)
    offsets_m = (rec_indices - src_ix).astype(float) * DH   # 0, 0.1, 0.2, ..., 9.9 m

    # ---- Load traced model -----------------------------------------------
    npz_path = root / "results" / "benchmark" / "traced_model.npz"
    npz_data = np.load(str(npz_path))
    epsr_int  = npz_data["epsr"]
    sigma_int = npz_data["sigma"]

    # ---- Build full grid with PML ----------------------------------------
    epsr, sigma = build_traced_model(epsr_int, sigma_int, NX, NZ, NPML)

    # ---- Frequency setup -------------------------------------------------
    freqs = np.linspace(fc_low, fc_high, nf)
    df = (fc_high - fc_low) / (nf - 1) if nf > 1 else fc_high

    TmaxFD = 0.25 / df
    dt_fd  = TmaxFD / nf

    print("=" * 60)
    print("Lavoue (2014) Fig. 2.15 - Shot Gather")
    print("=" * 60)
    print(f"  Model      : {npz_path}")
    print(f"  Grid       : ({NZ}, {NX})  PML={NPML}  dh={DH} m")
    print(f"  Source     : ix={src_ix} (x=0 m)  iz={src_iz}")
    print(f"  Receivers  : iz={rec_iz}  ntr={ntr}  "
          f"offset=[{offsets_m[0]:.1f}, {offsets_m[-1]:.1f}] m")
    print(f"  Freq       : {fc_low/1e6:.1f}-{fc_high/1e6:.0f} MHz  nf={nf}"
          f"  df={df/1e6:.3f} MHz")
    print(f"  Ricker fc  : {fc_ricker/1e6:.0f} MHz"
          f"  (time-derivative convolution)")
    print(f"  GERMAINE   : TmaxFD={TmaxFD*1e9:.1f} ns"
          f"  dt={dt_fd*1e9:.3f} ns")
    print(f"  Display    : 0-{tmax_ns:.0f} ns")
    print(f"  AGC        : {'off' if agc_win == 0 else f'window={agc_win}'}")
    print(f"  Grid style : {grid_style}")
    print(f"  Workers    : {n_workers}")

    # ==================================================================
    # FDFD solve
    # ==================================================================
    freq_data = np.zeros((nf, ntr), dtype=complex)

    def _solve_one(fi):
        f_hz  = freqs[fi]
        omega = 2.0 * np.pi * f_hz
        A = build_helmholtz_2d(epsr, sigma, DH, omega, NPML,
                               a0_cfs=A0_CFS, grid_style=grid_style)
        src_amp = -(omega * MU0 * 1j) / DH**2
        Ez = solve_forward(A, src_ix, src_iz, NX, NZ,
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
    # Ricker-derivative FD2TD  (Lavoue 2014 style)
    # ==================================================================
    print("\nRicker-derivative FD2TD conversion ...")
    print(f"  Wavelet: d/dt Ricker(fc={fc_ricker/1e6:.0f} MHz)")
    sg, t_s, dt_actual = ricker_deriv_fd2td(freq_data, freqs, fc_ricker,
                                             tmax_td)
    t_ns = t_s * 1e9

    sg_max = np.max(np.abs(sg))
    if sg_max > 0:
        sg = sg / sg_max
    print(f"  Shape: {sg.shape}  t=[{t_ns[0]:.2f}, {t_ns[-1]:.2f}] ns")
    print(f"  dt={dt_actual*1e9:.3f} ns  max_before_norm={sg_max:.3e}")

    # ---- Save data -------------------------------------------------------
    npz_out = out_dir / "lavoue2014_fig215_shotgather.npz"
    np.savez(npz_out, sg=sg, offsets_m=offsets_m, t_ns=t_ns,
             freqs=freqs, fc_ricker=fc_ricker,
             dh=DH, src_ix=src_ix, src_iz=src_iz, rec_iz=rec_iz)
    print(f"  Data -> {npz_out}")

    # ==================================================================
    # Generate Figure 2.15(a) replica
    # ==================================================================
    print("\nGenerating Fig. 2.15 plots ...")

    # ---- Main figure: exact paper replica ----
    fig_path = out_dir / "lavoue2014_fig215_shotgather.png"
    plot_fig215(sg, offsets_m, t_ns, fig_path,
                tmax_ns=tmax_ns, agc_window=agc_win)
    print(f"  Fig 2.15(a)  -> {fig_path}")

    # ---- Also generate the existing-style figures for comparison ----
    from scripts.plot_shotgather import plot_shotgather_color, plot_shotgather_wiggle
    from examples.run_forward_shotgather import (
        plot_shotgather_seismic_wiggle, plot_model_with_geometry,
    )

    mn = "lavoue2014_fig215"
    x_label = "Offset (m)"

    # Color (seismic)
    f1 = out_dir / f"{mn}_color.png"
    plot_shotgather_color(
        sg, offsets_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f1, src_pos_m=0.0,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
    )
    print(f"  Color        -> {f1}")

    # Grayscale
    f2 = out_dir / f"{mn}_gray.png"
    plot_shotgather_color(
        sg, offsets_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f2, src_pos_m=0.0,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
        cmap_name="gray",
        title="Synthetic GPR Shot Gather (grayscale)",
        cbar_label="Normalized Amplitude",
    )
    print(f"  Gray         -> {f2}")

    # Wiggle
    f3 = out_dir / f"{mn}_wiggle.png"
    plot_shotgather_wiggle(
        sg, offsets_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f3, src_pos_m=0.0,
        gain=2.0, agc_window=agc_win, x_label=x_label,
        upsample=1, every_nth=2,
    )
    print(f"  Wiggle       -> {f3}")

    # Seismic wiggle (red/blue)
    f4 = out_dir / f"{mn}_seismic_wiggle.png"
    plot_shotgather_seismic_wiggle(
        sg, offsets_m, t_ns, save_path=f4,
        src_pos_m=0.0,
        agc_window=agc_win, agc_threshold=0.1,
        every_nth=2, gain=2.0, x_label=x_label,
    )
    print(f"  Seismic      -> {f4}")

    # Model with geometry
    f5 = out_dir / f"{mn}_model.png"
    plot_model_with_geometry(
        epsr, sigma, NX, NZ, NPML, DH,
        src_ix, src_iz, rec_indices, rec_iz,
        save_path=f5,
    )
    print(f"  Model        -> {f5}")

    print("\n" + "=" * 60)
    print("Done. 6 figures saved to", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
