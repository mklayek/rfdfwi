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
Run FDFD forward modelling — time-domain shot gather.

For a single source, the FDFD wavefield is solved at multiple frequencies.
All receivers along a surface line are extracted and the full frequency
matrix is converted to a time-domain shot gather via the GPRFM Hermitian
IFFT with a Blackman-Harris spectral window.

WHY the spectral window is essential
-------------------------------------
The Python FDFD solver uses a flat-spectrum (unit-amplitude) point source.
MATLAB's RHS_TE1 embeds a frequency-dependent Ricker/Blackman-Harris source
amplitude inside the FDFD, so MATLAB's precrnew already carries the wavelet
signature.  Applying a Blackman-Harris window here compensates for the flat
source and produces the same clean, bandlimited pulse as MATLAB.

Without the window: IFFT of a flat-spectrum signal → sinc-like waveform with
many equal-amplitude sidelobes (horizontal stripes in the gather).
With the window: IFFT → clean single Blackman-Harris pulse per arrival.

Algorithm (matches GPRFM.m shot gather section):

    1. FDFD solve at each frequency → Ez full wavefield
    2. precrnew[fi, :] = Ez[rec_iz, rec_indices]           (flat spectrum)
    3. precrnew *= blackman_harris_spectrum(freqs)          (wavelet shaping)
    4. Hermitian IFFT → time-domain shot gather
    5. 10× spline upsample + AGC2 + polarity flip (GPRFM post-processing)
    6. Colour (seismic colormap) + wiggle plots

Default frequencies: linspace 50–200 MHz, nf=50 (50 solves).
  → ns = 99 time samples, dt ≈ 3.3 ns, Tmax ≈ 325 ns
Override with --gprfm-freqs (10 discrete) or --fc-low/--fc-high/--nf.

X-axis: signed offset from source [m]  (negative left, positive right)
Y-axis: two-way travel time [ns]

Output
------
results/forward/shotgather/sg_<tags>.npz          Shot gather + metadata.
results/forward/shotgather/sg_<tags>.png          Colour image.
results/forward/shotgather/sg_wiggle_<tags>.png   Wiggle trace image.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts._cli import add_common_args
from scripts.config_loader import (
    load_config, get_forward_config, get_domain, get_pml, get_freq_sweep,
)
from scripts.forward_fdfd import run_forward_single_source
from scripts.plot_cmp import (
    GPRFM_FREQS_HZ,
    freq_to_timedomain,
    blackman_harris_spectrum,
)
from scripts.plot_shotgather import plot_shotgather_color, plot_shotgather_wiggle

DEFAULT_OUT = root / "results" / "forward" / "shotgather"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDFD shot gather — time-domain, colour + wiggle (GPRFM style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_kind="forward_shotgather")

    # Source
    parser.add_argument(
        "--src-ix", type=int, default=None, metavar="IX",
        help="Source x grid index (default: nx//2).",
    )
    parser.add_argument(
        "--src-iz", type=int, default=20, metavar="IZ",
        help="Source depth grid index.",
    )

    # Receivers
    parser.add_argument(
        "--rec-iz", type=int, default=None, metavar="IZ",
        help="Receiver depth grid index (default: same as --src-iz).",
    )
    parser.add_argument(
        "--rec-step", type=int, default=1, metavar="N",
        help="Step between receivers [grid cells]. 1 = every cell.",
    )
    parser.add_argument(
        "--rec-ix-start", type=int, default=None, metavar="IX",
        help="First receiver x index (default: npx+1).",
    )
    parser.add_argument(
        "--rec-ix-end", type=int, default=None, metavar="IX",
        help="Last receiver x index inclusive (default: nx-npx-1).",
    )

    # Frequency mode
    freq_grp = parser.add_mutually_exclusive_group()
    freq_grp.add_argument(
        "--gprfm-freqs", action="store_true", default=False,
        help="Use GPRFM 10 discrete frequencies [50,60,...,200] MHz instead of "
             "the default linspace sweep.  Matches GPRFM.m exactly but gives "
             "coarser time resolution (ns=19, dt≈5.3 ns).",
    )
    freq_grp.add_argument(
        "--fc-low", type=float, default=None, metavar="HZ",
        help="Start frequency for custom linspace sweep [Hz].",
    )
    parser.add_argument(
        "--fc-high", type=float, default=None, metavar="HZ",
        help="End frequency for custom linspace sweep [Hz].",
    )
    parser.add_argument(
        "--nf", type=int, default=None, metavar="N",
        help="Number of linspace frequencies (default 50).",
    )

    # Spectral window
    parser.add_argument(
        "--no-window", action="store_true", default=False,
        help="Skip the Blackman-Harris spectral window (raw flat-spectrum IFFT; "
             "produces sinc-like waveforms instead of clean pulses).",
    )

    # IFFT / display
    parser.add_argument(
        "--tmax-ns", type=float, default=150.0, metavar="NS",
        help="Maximum display time [ns].",
    )
    parser.add_argument(
        "--pad", type=int, default=0, metavar="N",
        help="Zero-samples added on each side of Hermitian spectrum "
             "(ns = (nf+pad)*2-1). Default 0.",
    )
    parser.add_argument(
        "--wiggle-gain", type=float, default=1.5, metavar="G",
        help="Wiggle amplitude scale relative to trace spacing.",
    )

    return parser.parse_args()


def main() -> None:
    args    = _parse_args()
    out_dir = Path(args.results_dir) if args.results_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolve config ----
    if args.config:
        config_path = Path(args.config)
    else:
        model_eps   = root / "inputmodel" / "model_epsr.npy"
        config_path = (root / "input" / "input_forward_with_model.yaml"
                       if model_eps.exists() else
                       root / "input" / "input_forward.yaml")
        if not config_path.exists():
            config_path = root / "input" / "input_forward.yaml"

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

    # ---- Source position ----
    src_ix = args.src_ix if args.src_ix is not None else nx // 2
    src_iz = args.src_iz

    # ---- Receiver line ----
    rec_iz    = args.rec_iz if args.rec_iz is not None else src_iz
    rec_start = args.rec_ix_start if args.rec_ix_start is not None else npx + 1
    rec_end   = args.rec_ix_end   if args.rec_ix_end   is not None else nx - npx - 1
    rec_step  = args.rec_step

    rec_indices = np.arange(rec_start, rec_end + 1, rec_step)
    ntr         = len(rec_indices)

    # Signed offset from source [m]: negative = left, positive = right
    offsets_m = (rec_indices - src_ix).astype(float) * dh

    # ---- Frequency array ----
    # Default: linspace 50 frequencies (50–200 MHz) for good time resolution.
    # ns = 99 time samples, dt ≈ 3.3 ns, Tmax ≈ 325 ns.
    fs_cfg = get_freq_sweep(fwd_cfg)

    if args.gprfm_freqs:
        freqs      = np.array(GPRFM_FREQS_HZ, dtype=float)
        freq_label = (f"GPRFM  {freqs[0]/1e6:.0f}–{freqs[-1]/1e6:.0f} MHz"
                      f"  ({len(freqs)} discrete)")
        freq_tag   = "gprfm10"
    elif args.fc_low is not None or args.fc_high is not None or args.nf is not None:
        fc_low  = args.fc_low  if args.fc_low  is not None else fs_cfg["fc_low"]
        fc_high = args.fc_high if args.fc_high is not None else fs_cfg["fc_high"]
        nf      = args.nf      if args.nf      is not None else 50
        freqs   = np.linspace(fc_low, fc_high, nf)
        df      = (fc_high - fc_low) / max(nf - 1, 1)
        freq_label = f"linspace {fc_low/1e6:.0f}–{fc_high/1e6:.0f} MHz  nf={nf}  df={df/1e6:.3f} MHz"
        freq_tag   = f"{int(fc_low/1e6)}-{int(fc_high/1e6)}MHz_nf{nf}"
    else:
        # Default: linspace 50 freqs, 50–200 MHz
        fc_low  = fs_cfg.get("fc_low",  50e6)
        fc_high = fs_cfg.get("fc_high", 200e6)
        nf      = 50
        freqs   = np.linspace(fc_low, fc_high, nf)
        df      = (fc_high - fc_low) / max(nf - 1, 1)
        freq_label = (f"linspace {fc_low/1e6:.0f}–{fc_high/1e6:.0f} MHz"
                      f"  nf={nf}  df={df/1e6:.3f} MHz  [default]")
        freq_tag   = f"{int(fc_low/1e6)}-{int(fc_high/1e6)}MHz_nf{nf}"

    nf_    = len(freqs)
    pad    = args.pad
    ns     = (nf_ + pad) * 2 - 1
    df_min = float(np.min(np.diff(freqs))) if nf_ > 1 else float(freqs[0])
    dt_ns  = 1e9 / (ns * df_min)
    it_max = min(int(args.tmax_ns / dt_ns) + 1, ns)

    apply_window = not args.no_window
    window_label = "Blackman-Harris" if apply_window else "none (raw flat-spectrum)"

    # Filename stem
    stem = (f"sg_{grid_style}"
            f"_src{src_ix}-{src_iz}"
            f"_reciz{rec_iz}_step{rec_step}"
            f"_{freq_tag}")

    print(f"Config       : {config_path}")
    print(f"Domain       : {nx} x {nz}  (dh={dh} m)")
    print(f"PML          : npx={npx}, npz={npz}")
    print(f"Grid style   : {grid_style}")
    print(f"Source       : ix={src_ix}  ({src_ix*dh:.2f} m),  iz={src_iz}  ({src_iz*dh:.2f} m)")
    print(f"Receivers    : iz={rec_iz}  ({rec_iz*dh:.2f} m),  "
          f"ix={rec_start}..{rec_end} step={rec_step}  → {ntr} traces")
    print(f"Offset range : [{offsets_m[0]:.2f}, {offsets_m[-1]:.2f}] m")
    print(f"Frequencies  : {freq_label}")
    print(f"Spectral win : {window_label}")
    print(f"IFFT         : Hermitian ns={ns} (nf={nf_}+pad={pad})"
          f"  dt={dt_ns:.3f} ns  Tmax={ns*dt_ns:.0f} ns"
          f"  (display 0–{args.tmax_ns:.0f} ns)")
    print(f"Workers      : {n_workers}")
    print(f"Output stem  : {stem}")

    # ---- Frequency-domain data assembly ----
    # precrnew[fi, rec_idx] = Ez[rec_iz, rec_ix] at frequency freqs[fi]
    precrnew = np.zeros((nf_, ntr), dtype=complex)

    cfg_run = dict(fwd_cfg)
    cfg_run.setdefault("output", {})["save_fields"] = True
    cfg_run["source"] = {"ix": src_ix, "iz": src_iz}

    def _solve_freq(fi: int):
        cfg_fc            = dict(cfg_run)
        cfg_fc["freq_hz"] = float(freqs[fi])
        _, Ez, _          = run_forward_single_source(
            cfg_fc, use_gpu=args.use_gpu, grid_style=grid_style,
        )
        if Ez is None:
            return fi, np.zeros(ntr, dtype=complex)
        return fi, Ez[rec_iz, rec_indices].astype(complex)

    print(f"Solving {nf_} FDFD problems (1 source × {nf_} frequencies) ...")
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for fi, row in ex.map(_solve_freq, range(nf_)):
                precrnew[fi, :] = row
                print(f"  f={freqs[fi]/1e6:.0f} MHz  |E|_max={np.max(np.abs(row)):.3e}")
    else:
        for fi in range(nf_):
            _, row = _solve_freq(fi)
            precrnew[fi, :] = row
            print(f"  [{fi+1:2d}/{nf_}]  f={freqs[fi]/1e6:.0f} MHz"
                  f"  |E|_max={np.max(np.abs(row)):.3e}")

    # ---- Blackman-Harris spectral window (wavelet shaping) ----
    # Compensates for the flat-spectrum unit source used by the Python FDFD
    # solver.  MATLAB's RHS_TE1 embeds a frequency-dependent source amplitude;
    # this window replicates that effect.
    if apply_window:
        bh_win   = blackman_harris_spectrum(freqs)          # (nf,)
        precrnew = precrnew * bh_win[:, np.newaxis]         # broadcast over traces

    # ---- Hermitian IFFT → time domain (GPRFM algorithm) ----
    print("Applying Hermitian IFFT (MATLAB GPRFM style) ...")
    sg_full, t_ns_full = freq_to_timedomain(precrnew, freqs, pad=pad)

    sg_matrix = sg_full[:it_max, :]
    t_ns_disp = t_ns_full[:it_max]
    print(f"  Shot gather shape: {sg_matrix.shape}  "
          f"t = [0, {t_ns_disp[-1]:.1f}] ns")

    # ---- Save ----
    npz_path = out_dir / f"{stem}.npz"
    np.savez(npz_path,
             sg=sg_matrix,
             offsets_m=offsets_m,
             rec_indices=rec_indices,
             t_ns=t_ns_disp,
             dh=dh,
             freqs=freqs,
             precrnew=precrnew,
             src_ix=src_ix, src_iz=src_iz, rec_iz=rec_iz)
    print(f"  Data     -> {npz_path}")

    src_pos_m = float(src_ix * dh)

    # ---- Colour shot gather (GPRFM: spline upsample + AGC2 + polarity flip) ----
    png_path = out_dir / f"{stem}.png"
    plot_shotgather_color(
        sg_matrix, offsets_m, t_ns_disp,
        grid_style=grid_style,
        freqs_hz=freqs,
        save_path=png_path,
        src_pos_m=src_pos_m,
    )
    print(f"  Color    -> {png_path}")

    # ---- Wiggle shot gather ----
    wig_stem = (f"sg_wiggle_{grid_style}"
                f"_src{src_ix}-{src_iz}"
                f"_reciz{rec_iz}_step{rec_step}_{freq_tag}")
    wig_path = out_dir / f"{wig_stem}.png"
    plot_shotgather_wiggle(
        sg_matrix, offsets_m, t_ns_disp,
        grid_style=grid_style,
        freqs_hz=freqs,
        save_path=wig_path,
        src_pos_m=src_pos_m,
        gain=args.wiggle_gain,
    )
    print(f"  Wiggle   -> {wig_path}")


if __name__ == "__main__":
    main()
