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
#   Lavoue et al. (2014); Layek & Sengupta (2019, 2021, & 2024)
#   Koehn, D., De Nil, D. and Rabbel, W. (2017) Tutorial: Introduction to
#   frequency domain modelling and FWI of georadar data with GERMAINE.
#   DOI: 10.13140/RG.2.2.29354.03523
#   Amini, N. & Javaherian, A. (2011). Waves Random Complex Media, 21(1).
#   DOI: 10.1080/17455030.2010.537708
#
# Copyright (c) Mrinal Kanti Layek
# =============================================================================
"""
FDFD shot gather for Marmousi-II GPR model — GERMAINE FD2TD style.

Algorithm follows GERMAINE/par/visu/FD2TD.ipynb (Daniel Koehn, 2017):

  1. Solve FDFD at nf linspace frequencies (real omega)
  2. Taper frequency data with Tukey(alpha=1) = Hann window
  3. Build 4xnf Hermitian spectrum:
         S = [E,  zeros(nf,ntr),  zeros(nf,ntr),  flipud(E)]
  4. IFFT + real:   sg = Re{ IFFT(S, axis=0) }
  5. Time axis:     dt = 0.25 / (df x nf)
  6. Normalize by global max, display with fixed clip

Usage:
    python examples/run_forward_shotgather_marmousi.py --stag2 --ncpus 15
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
from scripts.plot_cmp import agc2


DEFAULT_OUT = root / "results" / "marmousi_work" / "shotgather"
SG_DPI = 600


# ---------------------------------------------------------------------------
# GERMAINE FD2TD conversion
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
    sg    : (nmaxFD-1, ntr) real — time-domain shot gather (normalized).
    t_s   : (nmaxFD-1,) — time axis [s].
    dt    : float — time sample interval [s].
    """
    ntr = freq_data.shape[1]

    # --- Tukey(alpha=1) = Hann window along frequency axis ---
    try:
        window = signal.windows.tukey(nf, alpha=1.0)
    except AttributeError:
        window = np.hanning(nf)
    tapered = freq_data.copy()
    for i in range(ntr):
        tapered[:, i] *= window

    # --- GERMAINE Hermitian assembly: [E, zeros, zeros, flipud(E)] ---
    S = np.concatenate([
        tapered,
        np.zeros((nf, ntr), dtype=complex),
        np.zeros((nf, ntr), dtype=complex),
        np.flipud(tapered),
    ], axis=0)

    # --- IFFT + real (NO ifftshift) ---
    td = np.real(np.fft.ifft(S, axis=0))

    # --- Time axis ---
    TmaxFD = 0.25 / df
    dt = TmaxFD / nf
    nmaxFD = int(tmax_td / dt)
    nmaxFD = min(nmaxFD, td.shape[0])

    sg = td[1:nmaxFD, :]
    t_s = np.arange(1, nmaxFD) * dt

    return sg, t_s, dt


# ---------------------------------------------------------------------------
# Model plot with source & receiver markers
# ---------------------------------------------------------------------------
def plot_model_with_geometry(
    epsr: np.ndarray,
    sigma: np.ndarray,
    nx: int, nz: int, npml: int, dh: float,
    src_ix: int, src_iz: int,
    rec_indices: np.ndarray, rec_iz: int,
    save_path: Path,
    dpi: int = SG_DPI,
) -> None:
    """Save a publishable 2-panel model image (epsr + sigma) with geometry."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 11,
    })

    int_s = np.s_[npml:nz - npml, npml:nx - npml]
    x_lo, x_hi = npml * dh, (nx - npml) * dh
    z_lo, z_hi = npml * dh, (nz - npml) * dh
    extent = [x_lo, x_hi, z_hi, z_lo]

    src_x = src_ix * dh
    src_z = src_iz * dh
    rec_x = rec_indices.astype(float) * dh
    rec_z = np.full_like(rec_x, rec_iz * dh)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=dpi)

    # ---- (a) Relative permittivity ----
    im0 = axes[0].imshow(epsr[int_s], extent=extent, aspect="auto", cmap="jet")
    axes[0].set_title(r"(a) Relative Permittivity ($\varepsilon_r$)", fontsize=15)
    axes[0].set_xlabel("Distance (m)")
    axes[0].set_ylabel("Depth (m)")
    cb0 = fig.colorbar(im0, ax=axes[0], shrink=0.88, pad=0.02)
    cb0.set_label(r"$\varepsilon_r$", fontsize=14)
    cb0.ax.tick_params(labelsize=12)

    # ---- (b) Conductivity (mS/m) ----
    im1 = axes[1].imshow(sigma[int_s] * 1e3, extent=extent, aspect="auto", cmap="jet")
    axes[1].set_title(r"(b) Conductivity ($\sigma$)", fontsize=15)
    axes[1].set_xlabel("Distance (m)")
    axes[1].set_ylabel("Depth (m)")
    cb1 = fig.colorbar(im1, ax=axes[1], shrink=0.88, pad=0.02)
    cb1.set_label("mS/m", fontsize=14)
    cb1.ax.tick_params(labelsize=12)

    # ---- Source (star) & receiver (triangle) markers — no legend ----
    for ax in axes:
        rec_thin = np.arange(0, len(rec_x), max(1, len(rec_x) // 20))
        ax.plot(rec_x[rec_thin], rec_z[rec_thin], 'v', color='lime',
                markersize=6, markeredgecolor='black', markeredgewidth=0.6,
                zorder=5)
        ax.plot(src_x, src_z, '*', color='red', markersize=16,
                markeredgecolor='black', markeredgewidth=0.8,
                zorder=6)

    fig.tight_layout(w_pad=3.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Seismic-style wiggle plot (Amini & Javaherian 2011, Fig. 10 style)
# ---------------------------------------------------------------------------
def plot_shotgather_seismic_wiggle(
    sg: np.ndarray,
    offsets_m: np.ndarray,
    t_ns: np.ndarray,
    save_path: Path,
    src_pos_m: float | None = None,
    agc_window: int = 0,
    agc_threshold: float = 0.0,
    every_nth: int = 3,
    gain: float = 0.8,
    clip_percentile: float = 99.5,
    dpi: int = SG_DPI,
    x_label: str = "Distance (m)",
) -> None:
    """
    Publishable variable-area wiggle plot — seismic style
    (Amini & Javaherian 2011, Fig. 10).
    Clean thin traces matching grayscale shot gather visualization.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    nt, ntr_in = sg.shape
    data = sg.copy().astype(float)

    if agc_window > 0:
        data = -agc2(data, window=agc_window, threshold=agc_threshold)

    # Clip amplitudes at percentile to prevent thick saturation
    clip_val = float(np.percentile(np.abs(data), clip_percentile))
    if clip_val > 0:
        data = np.clip(data, -clip_val, clip_val)

    trace_idx = np.arange(0, ntr_in, every_nth)
    x_sel = offsets_m[trace_idx]
    data_sel = data[:, trace_idx]
    ntr = len(trace_idx)

    dx = float(x_sel[1] - x_sel[0]) if ntr > 1 else 1.0
    wiggle_scale = gain * dx

    # Per-trace normalization for uniform visual weight
    for j in range(ntr):
        tmax = np.max(np.abs(data_sel[:, j]))
        if tmax > 0:
            data_sel[:, j] /= tmax

    fig, ax = plt.subplots(figsize=(12, 8), dpi=dpi)

    for j in range(ntr):
        x0 = x_sel[j]
        trace = data_sel[:, j]
        wiggle = x0 + wiggle_scale * trace

        ax.fill_betweenx(t_ns, x0, wiggle, where=(trace >= 0),
                         color='red', alpha=0.7, linewidth=0)
        ax.fill_betweenx(t_ns, x0, wiggle, where=(trace < 0),
                         color='blue', alpha=0.4, linewidth=0)
        ax.plot(wiggle, t_ns, color='black', linewidth=0.25)

    if "Distance" in x_label and src_pos_m is not None:
        ax.axvline(src_pos_m, color='green', linewidth=1.2, linestyle='--',
                   alpha=0.8)
    ax.set_xlim(x_sel[0] - dx * 0.5, x_sel[-1] + dx * 0.5)
    ax.set_ylim(t_ns[-1], t_ns[0])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time (ns)")
    ax.set_title("Synthetic GPR Shot Gather (wiggle)", pad=8)
    ax.grid(True, linewidth=0.2, color="gray", alpha=0.25)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    p = argparse.ArgumentParser(
        description="FDFD shot gather (GERMAINE FD2TD style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(p, default_kind="forward_shotgather")

    p.add_argument("--src-ix",  type=int, default=None)
    p.add_argument("--src-iz",  type=int, default=20)
    p.add_argument("--rec-iz",  type=int, default=None)
    p.add_argument("--rec-step", type=int, default=1)
    p.add_argument("--rec-ix-start", type=int, default=None)
    p.add_argument("--rec-ix-end",   type=int, default=None)

    p.add_argument("--nf",        type=int,   default=None, help="Number of frequencies.")
    p.add_argument("--fc-low",    type=float, default=None, help="Low freq [Hz].")
    p.add_argument("--fc-high",   type=float, default=None, help="High freq [Hz].")

    p.add_argument("--x-axis", default="distance", choices=["offset", "distance"])
    p.add_argument("--display-tmax-ns", type=float, default=250.0,
                   help="Display time window [ns].")
    p.add_argument("--clip", type=float, default=None,
                   help="Display clip value (after global-max normalization).")

    # AGC: mild by default (window=40 samples, threshold=0.1)
    p.add_argument("--no-agc", action="store_true", default=False,
                   help="Disable AGC.")
    p.add_argument("--agc-window", type=int, default=40,
                   help="AGC window [samples]. 0 disables AGC.")
    p.add_argument("--agc-threshold", type=float, default=0.1,
                   help="AGC threshold (fraction of per-trace max RMS).")
    p.add_argument("--wiggle-gain", type=float, default=2.0)
    p.add_argument("--wiggle-every", type=int, default=2,
                   help="Plot every Nth trace in seismic wiggle.")
    p.add_argument("--model-name", type=str, default=None,
                   help="Model name prefix for output filenames (e.g. 'clay_lens').")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args    = _parse_args()
    out_dir = Path(args.results_dir) if args.results_dir else None

    # ---- Config ----
    if args.config:
        config_path = Path(args.config)
    else:
        marmousi_config = root / "input" / "input_marmousi.yaml"
        config_path = marmousi_config if marmousi_config.exists() else root / "input" / "input_forward.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return

    config     = load_config(config_path)
    fwd_cfg    = get_forward_config(config)
    nx, nz, dx, dz = get_domain(fwd_cfg)
    npx, npz   = get_pml(fwd_cfg)
    dh         = float(dx)
    grid_style = args.grid_style or fwd_cfg.get("grid_style", "stag1")
    n_workers  = int(getattr(args, "ncpus", 1))
    pml_cfg    = fwd_cfg.get("pml", {})
    a0_cfs     = float(pml_cfg.get("a0_cfs", 9.0e8))

    if out_dir is None:
        cfg_out = fwd_cfg.get("output", {}).get("dir")
        out_dir = Path(root / cfg_out) if cfg_out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Source ----
    src_cfg = fwd_cfg.get("source", {})
    src_ix  = args.src_ix if args.src_ix is not None else int(src_cfg.get("ix", nx // 2))
    src_iz  = args.src_iz if args.src_iz != 20 else int(src_cfg.get("iz", 20))

    # ---- Receivers ----
    rcfg      = fwd_cfg.get("receivers", {})
    rec_iz    = args.rec_iz if args.rec_iz is not None else int(rcfg.get("iz", src_iz))
    rec_start = (args.rec_ix_start if args.rec_ix_start is not None
                 else int(rcfg.get("ix_start", npx + 1)))
    rec_end   = (args.rec_ix_end if args.rec_ix_end is not None
                 else int(rcfg.get("ix_end", nx - npx - 1)))
    rec_step  = (int(rcfg.get("ix_step", rcfg.get("step", 1)))
                 if args.rec_step == 1 and rcfg else args.rec_step)
    rec_end     = min(rec_end, nx - 1)
    rec_indices = np.arange(rec_start, rec_end + 1, rec_step)
    ntr         = len(rec_indices)

    if args.x_axis == "distance":
        x_axis_m = rec_indices.astype(float) * dh
        x_label  = "Distance (m)"
    else:
        x_axis_m = (rec_indices - src_ix).astype(float) * dh
        x_label  = "Offset [m]"

    # ---- Frequency setup: linspace (GERMAINE style) ----
    fs_cfg      = get_freq_sweep(fwd_cfg)
    display_cfg = fwd_cfg.get("display", {})

    fc_low  = args.fc_low  or float(fs_cfg.get("fc_low", 1e6))
    fc_high = args.fc_high or float(fs_cfg.get("fc_high", 200e6))
    nf      = args.nf      or int(fs_cfg.get("nf", 80))
    freqs   = np.linspace(fc_low, fc_high, nf)
    df      = (fc_high - fc_low) / (nf - 1) if nf > 1 else fc_high

    # Display time (default 100 ns)
    display_tmax_ns = args.display_tmax_ns
    tmax_td = display_tmax_ns * 1e-9

    # Clip value (after global-max normalization)
    clip_val = args.clip
    if clip_val is None:
        clip_val = float(fs_cfg.get("clip", 2.5e-3))

    # AGC settings
    agc_win = 0 if args.no_agc else args.agc_window
    agc_thr = args.agc_threshold

    # GERMAINE time parameters
    TmaxFD = 0.25 / df
    dt_fd  = TmaxFD / nf
    nmaxFD = int(tmax_td / dt_fd)

    stem = f"sg_{grid_style}_src{src_ix}-{src_iz}_{int(fc_high/1e6)}MHz"

    print(f"Config     : {config_path}")
    print(f"Domain     : {nx}x{nz}  dh={dh} m")
    print(f"PML        : npx={npx} npz={npz} a0={a0_cfs:.1e}")
    print(f"Grid       : {grid_style}")
    print(f"Source     : ix={src_ix} ({src_ix*dh:.2f}m)  iz={src_iz} ({src_iz*dh:.2f}m)")
    print(f"Receivers  : iz={rec_iz}  ix=[{rec_start}..{rec_end}] step={rec_step}  ntr={ntr}")
    print(f"Freq       : {fc_low/1e6:.1f}-{fc_high/1e6:.0f} MHz  nf={nf}  df={df/1e6:.3f} MHz")
    print(f"GERMAINE   : TmaxFD={TmaxFD*1e9:.1f} ns  dt={dt_fd*1e9:.3f} ns  "
          f"nmaxFD={nmaxFD}")
    print(f"Display    : 0-{display_tmax_ns:.0f} ns  clip={clip_val:.4f}")
    print(f"AGC        : {'off' if agc_win == 0 else f'window={agc_win} threshold={agc_thr}'}")
    print(f"Workers    : {n_workers}")

    # ---- Build model ----
    model_cfg = fwd_cfg.get("model", {})
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

    # ---- Save model image with source & receiver geometry ----
    model_png = out_dir / f"model_{stem}.png"
    plot_model_with_geometry(
        epsr, sigma, nx, nz, npx, dh,
        src_ix, src_iz, rec_indices, rec_iz,
        save_path=model_png,
    )
    print(f"  Model  -> {model_png}")

    # ==================================================================
    # FDFD solve — nf frequencies, real omega, TE source amplitude
    # ==================================================================
    freq_data = np.zeros((nf, ntr), dtype=complex)

    def _solve_one(fi):
        f_hz  = freqs[fi]
        omega = 2.0 * np.pi * f_hz
        A = build_helmholtz_2d(epsr, sigma, dh, omega, npx,
                               a0_cfs=a0_cfs, grid_style=grid_style)
        src_amp = -(omega * MU0 * 1j) / dh**2
        Ez = solve_forward(A, src_ix, src_iz, nx, nz,
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

    # Global max normalization (GERMAINE style)
    sg_max = np.max(np.abs(sg))
    if sg_max > 0:
        sg = sg / sg_max
    print(f"  Shape: {sg.shape}  t=[{t_ns[0]:.2f}, {t_ns[-1]:.2f}] ns")
    print(f"  dt={dt_actual*1e9:.3f} ns  max_before_norm={sg_max:.3e}")

    # ---- Save data ----
    npz_path = out_dir / f"{stem}.npz"
    np.savez(npz_path, sg=sg, x_axis_m=x_axis_m, t_ns=t_ns,
             freqs=freqs, dh=dh, src_ix=src_ix, src_iz=src_iz,
             rec_iz=rec_iz, clip=clip_val)
    print(f"  Data -> {npz_path}")

    src_pos_m = float(src_ix * dh)

    # ==================================================================
    # Generate all figures — single set with model-name prefix
    # ==================================================================
    if args.model_name:
        model_name = args.model_name
    else:
        model_name = model_cfg.get("type", "marmousi_gpr")
    mn_stem = f"{model_name}_shotgather"

    # ---- Color plot (seismic_r, thin reflections) ----
    mn_color = out_dir / f"{mn_stem}_color.png"
    plot_shotgather_color(
        sg, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=mn_color, src_pos_m=src_pos_m,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label,
        clip_frac=None, clip_pct=99.5,
        upsample=1,
    )
    print(f"  Color   -> {mn_color}")

    # ---- Grayscale plot ----
    mn_gray = out_dir / f"{mn_stem}_gray.png"
    plot_shotgather_color(
        sg, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=mn_gray, src_pos_m=src_pos_m,
        agc_window=0, agc_threshold=0.0,
        x_label=x_label,
        clip_frac=None, clip_pct=99.5,
        upsample=1,
        cmap_name="gray",
        title="Synthetic GPR Shot Gather (grayscale)",
        cbar_label="Normalized Amplitude",
    )
    print(f"  Gray    -> {mn_gray}")

    # ---- Standard wiggle plot (mild AGC to reveal dipping events) ----
    mn_wiggle = out_dir / f"{mn_stem}_wiggle.png"
    plot_shotgather_wiggle(
        sg, x_axis_m, t_ns, grid_style=grid_style, freqs_hz=freqs,
        save_path=mn_wiggle, src_pos_m=src_pos_m,
        gain=args.wiggle_gain, agc_window=agc_win, x_label=x_label,
        upsample=1, every_nth=args.wiggle_every,
    )
    print(f"  Wiggle  -> {mn_wiggle}")

    # ---- Seismic-style wiggle (red/blue, mild AGC) ----
    mn_seis = out_dir / f"{mn_stem}_seismic_wiggle.png"
    plot_shotgather_seismic_wiggle(
        sg, x_axis_m, t_ns, save_path=mn_seis,
        src_pos_m=src_pos_m,
        agc_window=agc_win, agc_threshold=agc_thr,
        every_nth=args.wiggle_every, gain=args.wiggle_gain,
        x_label=x_label,
    )
    print(f"  Seismic -> {mn_seis}")

    # ---- Model geometry ----
    mn_model = out_dir / f"{mn_stem}_model.png"
    plot_model_with_geometry(
        epsr, sigma, nx, nz, npx, dh,
        src_ix, src_iz, rec_indices, rec_iz,
        save_path=mn_model,
    )
    print(f"  Model   -> {mn_model}")

    print("Done.")


if __name__ == "__main__":
    main()
