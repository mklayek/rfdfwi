# =============================================================================
# RFDFWI — Full-Waveform Inversion (FWI) of GPR Data  |  By Mrinal
#
# GPR B-scan (radargram) for Marmousi-II GPR model — standalone script.
#
# A B-scan is a 2D radargram formed by stacking multiple A-scan traces
# collected along a survey line. The horizontal axis represents antenna
# position (distance), and the vertical axis represents two-way travel time,
# showing subsurface reflections and diffraction hyperbolas.
#
# Algorithm: GERMAINE FD2TD (Daniel Koehn, 2017) — identical to shot gather.
# For each source position, a co-located (zero-offset) receiver records
# one A-scan. All A-scans are stacked to form the B-scan.
#
# References:
#   Layek & Sengupta (2021) DOI: 10.1007/s00024-021-02685-3
#   Koehn, De Nil & Rabbel (2017) DOI: 10.13140/RG.2.2.29354.03523
#
# Copyright © Mrinal Kanti Layek
# =============================================================================
"""
GPR B-scan (radargram) for Marmousi-II GPR model — GERMAINE FD2TD style.

Usage:
    python examples/run_forward_bscan_marmousi.py --stag2 --ncpus 15

For each source position on the ground surface, a co-located receiver
records one A-scan trace in the frequency domain. After solving FDFD at
nf frequencies, the GERMAINE FD2TD algorithm converts to time domain.
All A-scans stacked = B-scan radargram.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import signal

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts._cli import add_common_args
from scripts.config_loader import (
    load_config, get_forward_config, get_domain, get_pml, get_freq_sweep,
)
from scripts.forward_fdfd import build_helmholtz_2d, solve_forward, MU0
from scripts.plot_shotgather import plot_shotgather_color, plot_shotgather_wiggle

DEFAULT_OUT = root / "results" / "marmousi_work" / "bscan"
BSCAN_DPI = 600


# ---------------------------------------------------------------------------
# GERMAINE FD2TD conversion (identical to shot gather — standalone copy)
# ---------------------------------------------------------------------------
def germaine_fd2td(freq_data: np.ndarray, nf: int, df: float,
                   tmax_td: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    GERMAINE-style frequency-to-time-domain conversion.

    Parameters
    ----------
    freq_data : (nf, ntr) complex — frequency-domain receiver data.
    nf        : int — number of frequencies.
    df        : float — frequency spacing [Hz].
    tmax_td   : float — desired max time [s] for extraction.

    Returns
    -------
    sg    : (nmaxFD-1, ntr) real — time-domain data.
    t_s   : (nmaxFD-1,) — time axis [s].
    dt    : float — time sample interval [s].
    """
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
    nmaxFD = int(tmax_td / dt)
    nmaxFD = min(nmaxFD, td.shape[0])

    sg = td[1:nmaxFD, :]
    t_s = np.arange(1, nmaxFD) * dt

    return sg, t_s, dt


# ---------------------------------------------------------------------------
# Model plot with source line (no legend)
# ---------------------------------------------------------------------------
def plot_model_with_bscan_geometry(
    epsr: np.ndarray,
    sigma: np.ndarray,
    nx: int, nz: int, npml: int, dh: float,
    src_positions: np.ndarray, src_iz: int,
    save_path: Path,
    dpi: int = BSCAN_DPI,
) -> None:
    """Save a 2-panel model image with B-scan source line markers."""
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
        # Mark start and end with triangles
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
        description="GPR B-scan radargram for Marmousi-II GPR model (GERMAINE FD2TD style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(p, default_kind="forward_bscan_marmousi")

    # Source survey line
    p.add_argument("--src-iz", type=int, default=None,
                   help="Source depth index (default: ground surface = npml).")
    p.add_argument("--src-start", type=int, default=None,
                   help="First source x-index (default: npml+2).")
    p.add_argument("--src-end", type=int, default=None,
                   help="Last source x-index (default: nx-npml-2).")
    p.add_argument("--src-step", type=int, default=None,
                   help="Source spacing in grid cells (auto: 0.10 m / dh).")
    p.add_argument("--rx-offset", type=int, default=0,
                   help="Receiver offset from source in grid cells (0 = zero-offset).")

    # Frequency / display
    p.add_argument("--nf", type=int, default=None)
    p.add_argument("--fc-low", type=float, default=None)
    p.add_argument("--fc-high", type=float, default=None)
    p.add_argument("--display-tmax-ns", type=float, default=250.0,
                   help="Display time window [ns].")

    # Wiggle
    p.add_argument("--wiggle-gain", type=float, default=0.8)
    p.add_argument("--wiggle-every", type=int, default=1,
                   help="Plot every Nth trace in wiggle (1 = all traces).")
    p.add_argument("--agc-window", type=int, default=60)
    p.add_argument("--agc-threshold", type=float, default=0.0)

    # SEC (time-power) gain for realistic display
    p.add_argument("--sec-power", type=float, default=1.5,
                   help="SEC gain exponent: data *= (t/t0)^power. "
                        "Compensates natural decay while preserving "
                        "lateral amplitude variations (e.g. clay shadow).")

    # Model name for filenames
    p.add_argument("--model-name", type=str, default=None,
                   help="Model name prefix for output filenames.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = _parse_args()
    out_dir = Path(args.results_dir) if args.results_dir else None

    # ---- Config ----
    if args.config:
        config_path = Path(args.config)
    else:
        marm_config = root / "input" / "input_marmousi.yaml"
        config_path = marm_config if marm_config.exists() else root / "input" / "input_forward.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return

    config = load_config(config_path)
    fwd_cfg = get_forward_config(config)
    nx, nz, dx, dz = get_domain(fwd_cfg)
    npx, npz = get_pml(fwd_cfg)
    dh = float(dx)
    grid_style = args.grid_style or fwd_cfg.get("grid_style", "stag1")
    n_workers = int(getattr(args, "ncpus", 1))
    pml_cfg = fwd_cfg.get("pml", {})
    a0_cfs = float(pml_cfg.get("a0_cfs", 9.0e8))

    if out_dir is None:
        out_dir = DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Source survey line ----
    # Auto-calculate: 0.10 m spacing across the interior domain
    trace_spacing_m = 0.10  # 10 cm step size
    src_step = args.src_step if args.src_step is not None else max(1, round(trace_spacing_m / dh))
    src_iz = args.src_iz if args.src_iz is not None else npx
    src_start = args.src_start if args.src_start is not None else (npx + 2)
    src_end = args.src_end if args.src_end is not None else (nx - npx - 2)
    rx_offset = args.rx_offset

    src_positions = np.arange(src_start, src_end + 1, src_step)
    n_src = len(src_positions)

    # Receiver positions (co-located or fixed offset)
    rec_positions = np.clip(src_positions + rx_offset, 0, nx - 1)
    rec_iz = src_iz  # same depth as source

    # X-axis in metres — first trace at 0, no blank space
    x_axis_m = (src_positions.astype(float) - src_positions[0]) * dh
    x_label = "Distance (m)"

    # ---- Frequency setup ----
    fs_cfg = get_freq_sweep(fwd_cfg)
    fc_low = args.fc_low or float(fs_cfg.get("fc_low", 1e6))
    fc_high = args.fc_high or float(fs_cfg.get("fc_high", 200e6))
    nf = args.nf or int(fs_cfg.get("nf", 80))
    freqs = np.linspace(fc_low, fc_high, nf)
    df = (fc_high - fc_low) / (nf - 1) if nf > 1 else fc_high

    display_tmax_ns = args.display_tmax_ns
    tmax_td = display_tmax_ns * 1e-9

    # GERMAINE time parameters
    TmaxFD = 0.25 / df
    dt_fd = TmaxFD / nf
    nmaxFD = int(tmax_td / dt_fd)

    # Model name
    model_cfg = fwd_cfg.get("model", {})
    model_name = args.model_name or model_cfg.get("type", "marmousi_gpr")
    mn_stem = f"{model_name}_bscan"

    agc_win = args.agc_window
    agc_thr = args.agc_threshold

    print(f"Config     : {config_path}")
    print(f"Domain     : {nx}x{nz}  dh={dh} m")
    print(f"PML        : npx={npx} npz={npz} a0={a0_cfs:.1e}")
    print(f"Grid       : {grid_style}")
    print(f"B-scan     : {n_src} sources = {n_src} receivers  "
          f"ix=[{src_start}..{src_end}] step={src_step} ({src_step*dh:.2f} m)")
    print(f"             src_iz={src_iz}  rx_offset={rx_offset} cells")
    print(f"Freq       : {fc_low/1e6:.1f}-{fc_high/1e6:.0f} MHz  nf={nf}  "
          f"df={df/1e6:.3f} MHz")
    print(f"GERMAINE   : TmaxFD={TmaxFD*1e9:.1f} ns  dt={dt_fd*1e9:.3f} ns  "
          f"nmaxFD={nmaxFD}")
    print(f"Display    : 0-{display_tmax_ns:.0f} ns")
    print(f"Workers    : {n_workers}")
    print(f"Model      : {model_name}")

    # ---- Build model ----
    if model_cfg.get("type") == "marmousi_gpr":
        from create_models.build_marmousi_gpr import (
            build_marmousi_gpr_model, print_model_summary,
        )
        print_model_summary(nx, nz, npx, dh)
        epsr, sigma = build_marmousi_gpr_model(
            nx, nz, npx, dh,
            segy_dir=model_cfg.get("segy_dir"),
            rho_matrix=float(model_cfg.get("rho_matrix", 2.65)),
            rho_fluid=float(model_cfg.get("rho_fluid", 1.00)),
            eps_matrix=float(model_cfg.get("eps_matrix", 5.0)),
            eps_water=float(model_cfg.get("eps_water", 80.0)),
            Sw=float(model_cfg.get("Sw", 1.0)),
            sigma_fluid=float(model_cfg.get("sigma_fluid", 0.05)),
            m_cementation=float(model_cfg.get("m_cementation", 1.8)),
            n_saturation=float(model_cfg.get("n_saturation", 2.0)),
            epsr_target_min=float(model_cfg.get("epsr_target_min", 4.0)),
            epsr_target_max=float(model_cfg.get("epsr_target_max", 32.0)),
            sigma_target_min=float(model_cfg.get("sigma_target_min", 0.1e-3)),
            sigma_target_max=float(model_cfg.get("sigma_target_max", 20.0e-3)),
        )
    elif model_cfg.get("type") == "layered_layek2021":
        from create_models.build_layered_layek2021 import (
            build_layered_layek2021_model, print_model_summary,
        )
        print_model_summary(nx, nz, npx, dh)
        epsr, sigma = build_layered_layek2021_model(
            nx, nz, npx, dh,
            epsr_layer1=float(model_cfg.get("epsr_layer1", 11.0)),
            sigma_layer1=float(model_cfg.get("sigma_layer1", 0.1e-3)),
            epsr_layer2=float(model_cfg.get("epsr_layer2", 4.0)),
            sigma_layer2=float(model_cfg.get("sigma_layer2", 6.0e-3)),
            epsr_layer3=float(model_cfg.get("epsr_layer3", 8.0)),
            sigma_layer3=float(model_cfg.get("sigma_layer3", 5.0e-3)),
            epsr_clay_lens=float(model_cfg.get("epsr_clay_lens", 35.0)),
            sigma_clay_lens=float(model_cfg.get("sigma_clay_lens", 20.0e-3)),
        )
    else:
        from create_models.build_models import build_model_from_config
        epsr, sigma = build_model_from_config(fwd_cfg, nx, nz)

    # ---- Save model image ----
    model_png = out_dir / f"{mn_stem}_model.png"
    plot_model_with_bscan_geometry(
        epsr, sigma, nx, nz, npx, dh,
        src_positions, src_iz,
        save_path=model_png,
    )
    print(f"  Model   -> {model_png}")

    # ==================================================================
    # FDFD solve — nf frequencies × n_src source positions
    # Build A once per frequency, solve for all source positions
    # ==================================================================
    freq_data = np.zeros((nf, n_src), dtype=complex)

    total_solves = nf * n_src
    print(f"\nSolving {nf} freqs x {n_src} sources = {total_solves} FDFD systems ...")

    for fi in range(nf):
        f_hz = freqs[fi]
        omega = 2.0 * np.pi * f_hz
        A = build_helmholtz_2d(epsr, sigma, dh, omega, npx,
                               a0_cfs=a0_cfs, grid_style=grid_style)
        src_amp = -(omega * MU0 * 1j) / dh**2

        def _solve_src(si, _A=A, _amp=src_amp):
            six = src_positions[si]
            rix = rec_positions[si]
            Ez = solve_forward(_A, six, src_iz, nx, nz, source_amplitude=_amp)
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
    # GERMAINE FD2TD conversion
    # ==================================================================
    print("\nGERMAINE FD2TD conversion ...")
    bscan, t_s, dt_actual = germaine_fd2td(freq_data, nf, df, tmax_td)
    t_ns = t_s * 1e9

    # Global max normalization
    bscan_max = np.max(np.abs(bscan))
    if bscan_max > 0:
        bscan = bscan / bscan_max
    print(f"  Shape: {bscan.shape}  t=[{t_ns[0]:.2f}, {t_ns[-1]:.2f}] ns")
    print(f"  dt={dt_actual*1e9:.3f} ns  max_before_norm={bscan_max:.3e}")

    # ---- Save data ----
    npz_path = out_dir / f"{mn_stem}.npz"
    np.savez(npz_path, bscan=bscan, x_axis_m=x_axis_m, t_ns=t_ns,
             freqs=freqs, dh=dh, src_positions=src_positions,
             src_iz=src_iz, rx_offset=rx_offset)
    print(f"  Data    -> {npz_path}")

    # ==================================================================
    # Generate figures
    # ==================================================================

    # ---- Color B-scan (seismic: red=+ve, blue=-ve, with AGC) ----
    color_png = out_dir / f"{mn_stem}_color.png"
    plot_shotgather_color(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=color_png, src_pos_m=None,
        agc_window=agc_win, agc_threshold=agc_thr,
        x_label=x_label,
        clip_frac=None, clip_pct=99.5,
        upsample=1,
        cmap_name="seismic",
        title="GPR B-scan Radargram",
    )
    print(f"  Color   -> {color_png}")

    # ---- Grayscale B-scan (with AGC) ----
    gray_png = out_dir / f"{mn_stem}_gray.png"
    plot_shotgather_color(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=gray_png, src_pos_m=None,
        agc_window=agc_win, agc_threshold=agc_thr,
        x_label=x_label,
        clip_frac=None, clip_pct=99.5,
        upsample=1,
        cmap_name="gray",
        title="GPR B-scan Radargram (grayscale)",
        cbar_label="Normalized Amplitude",
    )
    print(f"  Gray    -> {gray_png}")

    # ---- Color B-scan NO AGC (physically correct amplitudes) ----
    color_noagc_png = out_dir / f"{mn_stem}_color_noagc.png"
    plot_shotgather_color(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=color_noagc_png, src_pos_m=None,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label,
        clip_frac=None, clip_pct=99.5,
        upsample=1,
        cmap_name="seismic",
        title="GPR B-scan Radargram (no AGC)",
    )
    print(f"  NoAGC   -> {color_noagc_png}")

    # ---- Grayscale B-scan NO AGC ----
    gray_noagc_png = out_dir / f"{mn_stem}_gray_noagc.png"
    plot_shotgather_color(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=gray_noagc_png, src_pos_m=None,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label,
        clip_frac=None, clip_pct=99.5,
        upsample=1,
        cmap_name="gray",
        title="GPR B-scan Radargram (grayscale, no AGC)",
        cbar_label="Normalized Amplitude",
    )
    print(f"  NoAGC g -> {gray_noagc_png}")

    # ---- SEC (time-power) gain — realistic amplitude compensation ----
    # Applies t^power gain to compensate natural decay while preserving
    # lateral amplitude variations (e.g. clay lens attenuation shadow).
    sec_power = args.sec_power
    t_gain = t_ns.copy()
    t_gain[t_gain < 1.0] = 1.0  # avoid zero/negative times
    t0 = t_gain[0]
    sec_envelope = (t_gain / t0) ** sec_power
    bscan_sec = bscan * sec_envelope[:, np.newaxis]
    # Re-normalize
    sec_max = np.max(np.abs(bscan_sec))
    if sec_max > 0:
        bscan_sec = bscan_sec / sec_max

    color_sec_png = out_dir / f"{mn_stem}_color_sec.png"
    plot_shotgather_color(
        bscan_sec, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=color_sec_png, src_pos_m=None,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label,
        clip_frac=None, clip_pct=99.5,
        upsample=1,
        cmap_name="seismic",
        title=f"GPR B-scan Radargram (SEC gain, t^{sec_power:.1f})",
    )
    print(f"  SEC     -> {color_sec_png}")

    gray_sec_png = out_dir / f"{mn_stem}_gray_sec.png"
    plot_shotgather_color(
        bscan_sec, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=gray_sec_png, src_pos_m=None,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label,
        clip_frac=None, clip_pct=99.5,
        upsample=1,
        cmap_name="gray",
        title=f"GPR B-scan Radargram (grayscale, SEC t^{sec_power:.1f})",
        cbar_label="Normalized Amplitude",
    )
    print(f"  SEC g   -> {gray_sec_png}")

    # ---- Wiggle B-scan (with mild AGC for dipping events) ----
    wiggle_png = out_dir / f"{mn_stem}_wiggle.png"
    plot_shotgather_wiggle(
        bscan, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=wiggle_png, src_pos_m=None,
        gain=args.wiggle_gain, agc_window=agc_win,
        x_label=x_label,
        upsample=1, every_nth=args.wiggle_every,
    )
    print(f"  Wiggle  -> {wiggle_png}")

    print("Done.")


if __name__ == "__main__":
    main()
