# =============================================================================
# RFDFWI - Full-Waveform Inversion (FWI) of GPR Data  |  By Mrinal
#
# Standalone B-scan radargram for the Lavoue (2014) benchmark model.
# Loads the traced model, runs FDFD at multiple frequencies for each source
# position, converts to time domain via Ricker-derivative FD2TD, and
# produces publication-quality B-scan figures.
#
# Reference:
#   Lavoue, F. (2014). 2D full waveform inversion of ground penetrating
#   radar data. PhD thesis, Universite de Grenoble. Fig. 2.14, 2.15.
#
# Copyright (c) Mrinal Kanti Layek
# =============================================================================
"""
run_forward_bscan_lavoue2014.py
================================
GPR B-scan (radargram) for the Lavoue (2014) realistic subsurface benchmark.

A B-scan is formed by stacking A-scan traces from multiple source positions
along a survey line.  For each source, a co-located (or offset) receiver
records one trace.

FD2TD uses the time-derivative of a Ricker wavelet (fc = 100 MHz) convolved
in the frequency domain, matching the paper's methodology (Section 2.2.3.1).

Usage
-----
    cd D:/rfdfwi
    python examples/run_forward_bscan_lavoue2014.py --stag2 --ncpus 15

Output  (all in results/benchmark/bscan/)
------
    lavoue2014_bscan_color.png          colour seismic (AGC)
    lavoue2014_bscan_gray.png           grayscale (AGC)
    lavoue2014_bscan_color_noagc.png    colour seismic (no AGC)
    lavoue2014_bscan_gray_noagc.png     grayscale (no AGC)
    lavoue2014_bscan_color_sec.png      colour seismic (SEC gain)
    lavoue2014_bscan_gray_sec.png       grayscale (SEC gain)
    lavoue2014_bscan_wiggle.png         wiggle trace
    lavoue2014_bscan_model.png          model with survey geometry
    lavoue2014_bscan.npz                data file
"""
from __future__ import annotations

import argparse
import sys
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
from scripts.plot_shotgather import plot_shotgather_color, plot_shotgather_wiggle
from create_models.trace_lavoue2014_figure import build_traced_model


# ---------------------------------------------------------------------------
# Constants  (Lavoue 2014, Section 2.2.3)
# ---------------------------------------------------------------------------
NX, NZ, NPML, DH = 220, 130, 10, 0.05
N_AIR       = 10
IZ_GROUND   = NPML + N_AIR     # = 20
IZ_SRC_REC  = 18                # 2 cells above ground (z = -0.1 m)
A0_CFS      = 9.0e8
FC_RICKER   = 200e6             # Ricker central frequency
DPI         = 600


# ---------------------------------------------------------------------------
# Ricker-derivative FD2TD  (Lavoue 2014 style)
# ---------------------------------------------------------------------------
def ricker_deriv_fd2td(freq_data, freqs, fc, tmax_td):
    """
    Lavoue (2014) FD-to-TD: convolve with time-derivative of Ricker
    wavelet in frequency domain, then Hermitian IFFT.

    The Ricker derivative spectrum provides the physical wavelet shape,
    and a Hann (Tukey alpha=1) window ensures the spectrum tapers to
    zero at both band edges — preventing Gibbs ringing artifacts.
    """
    from scipy import signal

    nf, ntr = freq_data.shape

    # Ricker spectrum: R(f) = (f/fc)^2 * exp(-(f/fc)^2)
    f_ratio = freqs / fc
    ricker_spec = f_ratio**2 * np.exp(-f_ratio**2)

    # Time derivative: multiply by j*omega
    omega = 2.0 * np.pi * freqs
    wavelet_spec = 1j * omega * ricker_spec

    # Hann window to taper band edges (prevents Gibbs ringing)
    try:
        hann = signal.windows.tukey(nf, alpha=1.0)
    except AttributeError:
        hann = np.hanning(nf)

    convolved = freq_data.copy()
    for i in range(ntr):
        convolved[:, i] *= wavelet_spec * hann

    # GERMAINE-style Hermitian IFFT: [E, 0, 0, flipud(E)]
    S = np.concatenate([
        convolved,
        np.zeros((nf, ntr), dtype=complex),
        np.zeros((nf, ntr), dtype=complex),
        np.flipud(convolved),
    ], axis=0)

    td = np.real(np.fft.ifft(S, axis=0))

    df = (freqs[-1] - freqs[0]) / (nf - 1) if nf > 1 else freqs[-1]
    TmaxFD = 0.25 / df
    dt = TmaxFD / nf
    nmaxFD = min(int(tmax_td / dt), td.shape[0])

    sg = td[1:nmaxFD, :]
    t_s = np.arange(1, nmaxFD) * dt
    return sg, t_s, dt


# ---------------------------------------------------------------------------
# Model plot with B-scan survey line
# ---------------------------------------------------------------------------
def plot_model_with_bscan_geometry(
    epsr, sigma, nx, nz, npml, dh,
    src_positions, src_iz, save_path, dpi=DPI,
):
    """Save 2-panel model image with B-scan source line markers."""
    plt.rcParams.update({
        "font.family": "serif", "font.size": 14,
        "axes.labelsize": 16, "axes.titlesize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13,
    })

    int_s = np.s_[npml:nz - npml, npml:nx - npml]
    x_lo, x_hi = npml * dh, (nx - npml) * dh
    z_lo, z_hi = npml * dh, (nz - npml) * dh
    extent = [x_lo, x_hi, z_hi, z_lo]

    src_x = src_positions.astype(float) * dh
    src_z = src_iz * dh

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=dpi)

    im0 = axes[0].imshow(epsr[int_s], extent=extent, aspect="auto", cmap="jet")
    axes[0].set_title(r"(a) Relative Permittivity ($\varepsilon_r$)", fontsize=15)
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.88, pad=0.02)
    cb0.set_label(r"$\varepsilon_r$", fontsize=14)
    cb0.ax.tick_params(labelsize=12)

    im1 = axes[1].imshow(sigma[int_s] * 1e3, extent=extent, aspect="auto", cmap="jet")
    axes[1].set_title(r"(b) Conductivity ($\sigma$)", fontsize=15)
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Depth (m)")
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.88, pad=0.02)
    cb1.set_label("mS/m", fontsize=14)
    cb1.ax.tick_params(labelsize=12)

    for ax in axes:
        ax.plot(src_x, np.full_like(src_x, src_z), '-', color='lime',
                linewidth=2.5, zorder=5)
        ax.plot(src_x[0], src_z, 'v', color='lime', markersize=8,
                markeredgecolor='black', markeredgewidth=0.6, zorder=6)
        ax.plot(src_x[-1], src_z, 'v', color='lime', markersize=8,
                markeredgecolor='black', markeredgewidth=0.6, zorder=6)

    fig.tight_layout(w_pad=3.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(
        description="GPR B-scan for Lavoue (2014) benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stag1", dest="grid_style", action="store_const",
                   const="stag1", default="stag2")
    p.add_argument("--stag2", dest="grid_style", action="store_const",
                   const="stag2")
    p.add_argument("--ncpus", type=int, default=1)

    # Source survey line (paper: 41 sources, 0.25m spacing)
    p.add_argument("--src-step", type=int, default=5,
                   help="Source step in cells (5 = 0.25m at dh=0.05).")
    p.add_argument("--rx-offset", type=int, default=0,
                   help="Receiver offset from source in cells (0 = zero-offset).")

    # Frequency
    p.add_argument("--nf", type=int, default=80)
    p.add_argument("--fc-low", type=float, default=1e6)
    p.add_argument("--fc-high", type=float, default=200e6)
    p.add_argument("--fc-ricker", type=float, default=FC_RICKER,
                   help="Ricker wavelet central frequency [Hz].")

    # Display
    p.add_argument("--display-tmax-ns", type=float, default=300.0,
                   help="Display time window [ns].")
    p.add_argument("--agc-window", type=int, default=60)
    p.add_argument("--agc-threshold", type=float, default=0.0)
    p.add_argument("--sec-power", type=float, default=1.5,
                   help="SEC gain exponent: data *= (t/t0)^power.")

    # Wiggle
    p.add_argument("--wiggle-gain", type=float, default=0.8)
    p.add_argument("--wiggle-every", type=int, default=1,
                   help="Plot every Nth trace in wiggle (1 = all).")

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
    src_step   = args.src_step
    rx_offset  = args.rx_offset
    tmax_ns    = args.display_tmax_ns
    tmax_td    = tmax_ns * 1e-9
    agc_win    = 0 if args.no_agc else args.agc_window
    agc_thr    = args.agc_threshold
    sec_power  = args.sec_power

    out_dir = root / "results" / "benchmark" / "bscan"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Source survey line -----------------------------------------------
    # Paper: 41 sources, ix = 10,15,...,210 (0.25m spacing), iz = 18
    src_iz = IZ_SRC_REC
    src_start = NPML            # ix = 10
    src_end   = NX - NPML       # ix = 210
    src_positions = np.arange(src_start, src_end + 1, src_step)
    n_src = len(src_positions)

    # Receiver positions (co-located or fixed offset)
    rec_positions = np.clip(src_positions + rx_offset, 0, NX - 1)
    rec_iz = src_iz

    # X-axis: antenna position in metres (interior coordinates)
    x_axis_m = (src_positions - NPML).astype(float) * DH
    x_label = "Distance (m)"

    # ---- Load traced model ------------------------------------------------
    npz_path = root / "results" / "benchmark" / "traced_model.npz"
    npz_data = np.load(str(npz_path))
    epsr_int  = npz_data["epsr"]
    sigma_int = npz_data["sigma"]

    # ---- Build full grid with PML -----------------------------------------
    epsr, sigma = build_traced_model(epsr_int, sigma_int, NX, NZ, NPML)

    # ---- Frequency setup --------------------------------------------------
    freqs = np.linspace(fc_low, fc_high, nf)
    df = (fc_high - fc_low) / (nf - 1) if nf > 1 else fc_high
    TmaxFD = 0.25 / df
    dt_fd  = TmaxFD / nf
    total_solves = nf * n_src

    print("=" * 60)
    print("Lavoue (2014) Benchmark - B-scan Radargram")
    print("=" * 60)
    print(f"  Model      : {npz_path}")
    print(f"  Grid       : ({NZ}, {NX})  PML={NPML}  dh={DH} m")
    print(f"  B-scan     : {n_src} sources  "
          f"ix=[{src_start}..{src_positions[-1]}]  "
          f"step={src_step} ({src_step*DH:.2f} m)")
    print(f"             : iz={src_iz}  rx_offset={rx_offset} cells")
    print(f"  Freq       : {fc_low/1e6:.1f}-{fc_high/1e6:.0f} MHz  nf={nf}  "
          f"df={df/1e6:.3f} MHz")
    print(f"  Ricker fc  : {fc_ricker/1e6:.0f} MHz"
          f"  (time-derivative convolution)")
    print(f"  GERMAINE   : TmaxFD={TmaxFD*1e9:.1f} ns  dt={dt_fd*1e9:.3f} ns")
    print(f"  Display    : 0-{tmax_ns:.0f} ns")
    print(f"  AGC        : {'off' if agc_win == 0 else f'window={agc_win}'}")
    print(f"  SEC power  : {sec_power}")
    print(f"  Grid style : {grid_style}")
    print(f"  Workers    : {n_workers}")
    print(f"  Total FDFD : {total_solves} solves "
          f"({nf} freqs x {n_src} sources)")

    # ---- Save model image -------------------------------------------------
    model_png = out_dir / "lavoue2014_bscan_model.png"
    plot_model_with_bscan_geometry(
        epsr, sigma, NX, NZ, NPML, DH,
        src_positions, src_iz, save_path=model_png,
    )
    print(f"  Model   -> {model_png}")

    # ==================================================================
    # FDFD solve: nf frequencies x n_src source positions
    # Build A once per frequency, then solve for all sources
    # ==================================================================
    freq_data = np.zeros((nf, n_src), dtype=complex)

    print(f"\nSolving {total_solves} FDFD systems ...")

    for fi in range(nf):
        f_hz  = freqs[fi]
        omega = 2.0 * np.pi * f_hz
        A = build_helmholtz_2d(epsr, sigma, DH, omega, NPML,
                               a0_cfs=A0_CFS, grid_style=grid_style)
        src_amp = -(omega * MU0 * 1j) / DH**2

        def _solve_src(si, _A=A, _amp=src_amp):
            six = src_positions[si]
            rix = rec_positions[si]
            Ez = solve_forward(_A, six, src_iz, NX, NZ,
                               source_amplitude=_amp)
            return si, Ez[rec_iz, rix]

        if n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for si, val in ex.map(_solve_src, range(n_src)):
                    freq_data[fi, si] = val
        else:
            for si in range(n_src):
                _, val = _solve_src(si)
                freq_data[fi, si] = val

        print(f"  [{fi+1:3d}/{nf}]  f={f_hz/1e6:7.2f} MHz  "
              f"|E|_max={np.max(np.abs(freq_data[fi, :])):.3e}")

    # ==================================================================
    # Ricker-derivative FD2TD  (Lavoue 2014 style)
    # ==================================================================
    print("\nRicker-derivative FD2TD conversion ...")
    print(f"  Wavelet: d/dt Ricker(fc={fc_ricker/1e6:.0f} MHz)")
    bscan, t_s, dt_actual = ricker_deriv_fd2td(freq_data, freqs, fc_ricker,
                                                tmax_td)
    t_ns = t_s * 1e9

    # Global max normalization
    bscan_max = np.max(np.abs(bscan))
    if bscan_max > 0:
        bscan = bscan / bscan_max
    print(f"  Shape: {bscan.shape}  t=[{t_ns[0]:.2f}, {t_ns[-1]:.2f}] ns")
    print(f"  dt={dt_actual*1e9:.3f} ns  max_before_norm={bscan_max:.3e}")

    # ---- Save data --------------------------------------------------------
    npz_out = out_dir / "lavoue2014_bscan.npz"
    np.savez(npz_out, bscan=bscan, x_axis_m=x_axis_m, t_ns=t_ns,
             freqs=freqs, fc_ricker=fc_ricker, dh=DH,
             src_positions=src_positions, src_iz=src_iz,
             rx_offset=rx_offset)
    print(f"  Data    -> {npz_out}")

    # ==================================================================
    # Generate figures (same set as run_forward_bscan_radargram.py)
    # ==================================================================
    print("\nGenerating B-scan figures ...")
    mn = "lavoue2014_bscan"

    # ---- 1. Color B-scan (seismic, with AGC) ------------------------------
    f1 = out_dir / f"{mn}_color.png"
    plot_shotgather_color(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f1, src_pos_m=None,
        agc_window=agc_win, agc_threshold=agc_thr,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
        cmap_name="seismic", title="GPR B-scan Radargram",
    )
    print(f"  Color       -> {f1}")

    # ---- 2. Grayscale B-scan (with AGC) -----------------------------------
    f2 = out_dir / f"{mn}_gray.png"
    plot_shotgather_color(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f2, src_pos_m=None,
        agc_window=agc_win, agc_threshold=agc_thr,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
        cmap_name="gray", title="GPR B-scan Radargram (grayscale)",
        cbar_label="Normalized Amplitude",
    )
    print(f"  Gray        -> {f2}")

    # ---- 3. Color B-scan NO AGC -------------------------------------------
    f3 = out_dir / f"{mn}_color_noagc.png"
    plot_shotgather_color(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f3, src_pos_m=None,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
        cmap_name="seismic", title="GPR B-scan Radargram (no AGC)",
    )
    print(f"  NoAGC color -> {f3}")

    # ---- 4. Grayscale B-scan NO AGC ---------------------------------------
    f4 = out_dir / f"{mn}_gray_noagc.png"
    plot_shotgather_color(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f4, src_pos_m=None,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
        cmap_name="gray",
        title="GPR B-scan Radargram (grayscale, no AGC)",
        cbar_label="Normalized Amplitude",
    )
    print(f"  NoAGC gray  -> {f4}")

    # ---- 5-6. SEC (time-power) gain ---------------------------------------
    t_gain = t_ns.copy()
    t_gain[t_gain < 1.0] = 1.0
    t0 = t_gain[0]
    sec_envelope = (t_gain / t0) ** sec_power
    bscan_sec = bscan * sec_envelope[:, np.newaxis]
    sec_max = np.max(np.abs(bscan_sec))
    if sec_max > 0:
        bscan_sec = bscan_sec / sec_max

    f5 = out_dir / f"{mn}_color_sec.png"
    plot_shotgather_color(
        bscan_sec, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f5, src_pos_m=None,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
        cmap_name="seismic",
        title=f"GPR B-scan Radargram (SEC gain, t^{sec_power:.1f})",
    )
    print(f"  SEC color   -> {f5}")

    f6 = out_dir / f"{mn}_gray_sec.png"
    plot_shotgather_color(
        bscan_sec, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f6, src_pos_m=None,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label, clip_frac=None, clip_pct=99.5, upsample=1,
        cmap_name="gray",
        title=f"GPR B-scan Radargram (grayscale, SEC t^{sec_power:.1f})",
        cbar_label="Normalized Amplitude",
    )
    print(f"  SEC gray    -> {f6}")

    # ---- 7. Wiggle B-scan -------------------------------------------------
    f7 = out_dir / f"{mn}_wiggle.png"
    plot_shotgather_wiggle(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=f7, src_pos_m=None,
        gain=args.wiggle_gain, agc_window=agc_win,
        x_label=x_label, upsample=1, every_nth=args.wiggle_every,
    )
    print(f"  Wiggle      -> {f7}")

    print("\n" + "=" * 60)
    print(f"Done. 8 figures saved to {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
