"""
FDFD forward modelling — center-shot time-domain shot gather.

Replicates MATLAB GPRFM workflow with adaptive receiver geometry:
1. Generate FD shot gather (1 source, multiple receivers, multiple frequencies)
2. Apply Blackman-Harris spectral window (wavelet shaping)
3. Convert to TD via Hermitian IFFT
4. Reorganize for center-shot display:
   - For MATLAB geometry (ntr=117): exact tmp1-tmp4 split/flip replication
   - For other receiver counts: adaptive split/flip around center offset
5. Generate PNG visualizations and save .npz with both FD and reorganized TD data

This is the center-shot variant of run_forward_shotgather.py, with receiver
reorganization that creates a symmetric center-shot layout (negative offsets on
left/flipped, positive offsets on right).

Command-line usage example:
    python examples/run_forward_shotgather_center.py \\
        --gprfm-freqs \\
        --src-ix 100 --src-iz 20 \\
        --rec-iz 20 \\
        --ncpus 15

Output
------
results/forward/shotgather_center/sg_center_<tags>.npz       Data + metadata
results/forward/shotgather_center/sg_center_<tags>.png       Colour image
results/forward/shotgather_center/sg_center_wiggle_<tags>.png Wiggle image
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
from scripts.td_shotgather import reorganize_center_shot, verify_matlab_geometry

DEFAULT_OUT = root / "results" / "forward" / "shotgather_center"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDFD shot gather — center-shot TD, colour + wiggle (MATLAB GPRFM exact replica).",
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
        help="Use GPRFM 10 discrete frequencies [50,60,...,200] MHz (MATLAB exact). "
             "Gives nf=10, ns=19, dt≈5.3 ns.",
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
        help="Skip Blackman-Harris spectral window (raw flat-spectrum IFFT).",
    )

    # IFFT / display
    parser.add_argument(
        "--tmax-ns", type=float, default=150.0, metavar="NS",
        help="Maximum display time [ns].",
    )
    parser.add_argument(
        "--pad", type=int, default=0, metavar="N",
        help="Zero-padding on each side (ns = (nf+pad)*2-1). Default 0.",
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

    # Check MATLAB geometry
    is_matlab_geom = verify_matlab_geometry(ntr, 10)
    if not is_matlab_geom:
        print(f"[INFO] ntr={ntr} (MATLAB geometry has ntr=117)")
        print(f"       Using adaptive center-shot reorganization for this receiver count.")

    # Signed offset from source [m]: negative = left, positive = right
    offsets_m = (rec_indices - src_ix).astype(float) * dh

    # ---- Frequency array ----
    fs_cfg = get_freq_sweep(fwd_cfg)

    if args.gprfm_freqs:
        freqs      = np.array(GPRFM_FREQS_HZ, dtype=float)
        freq_label = (f"GPRFM  {freqs[0]/1e6:.0f}-{freqs[-1]/1e6:.0f} MHz"
                      f"  ({len(freqs)} discrete)")
        freq_tag   = "gprfm10"
    elif args.fc_low is not None or args.fc_high is not None or args.nf is not None:
        fc_low  = args.fc_low  if args.fc_low  is not None else fs_cfg["fc_low"]
        fc_high = args.fc_high if args.fc_high is not None else fs_cfg["fc_high"]
        nf      = args.nf      if args.nf      is not None else 50
        freqs   = np.linspace(fc_low, fc_high, nf)
        df      = (fc_high - fc_low) / max(nf - 1, 1)
        freq_label = f"linspace {fc_low/1e6:.0f}-{fc_high/1e6:.0f} MHz  nf={nf}  df={df/1e6:.3f} MHz"
        freq_tag   = f"{int(fc_low/1e6)}-{int(fc_high/1e6)}MHz_nf{nf}"
    else:
        # Default: linspace 50 freqs, 50-200 MHz
        fc_low  = fs_cfg.get("fc_low",  50e6)
        fc_high = fs_cfg.get("fc_high", 200e6)
        nf      = 50
        freqs   = np.linspace(fc_low, fc_high, nf)
        df      = (fc_high - fc_low) / max(nf - 1, 1)
        freq_label = (f"linspace {fc_low/1e6:.0f}-{fc_high/1e6:.0f} MHz"
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
    stem = (f"sg_center_{grid_style}"
            f"_src{src_ix}-{src_iz}"
            f"_reciz{rec_iz}_step{rec_step}"
            f"_{freq_tag}")

    print("\n" + "="*70)
    print("FDFD CENTER-SHOT GATHER (MATLAB GPRFM REPLICA)")
    print("="*70)
    print(f"Config       : {config_path}")
    print(f"Domain       : {nx} x {nz}  (dh={dh} m)")
    print(f"PML          : npx={npx}, npz={npz}")
    print(f"Grid style   : {grid_style}")
    print(f"Source       : ix={src_ix}  ({src_ix*dh:.2f} m),  iz={src_iz}  ({src_iz*dh:.2f} m)")
    print(f"Receivers    : iz={rec_iz}  ({rec_iz*dh:.2f} m),  "
          f"ix={rec_start}..{rec_end} step={rec_step}  ({ntr} traces)")
    print(f"Offset range : [{offsets_m[0]:.2f}, {offsets_m[-1]:.2f}] m")
    print(f"Frequencies  : {freq_label}")
    print(f"Spectral win : {window_label}")
    print(f"IFFT         : Hermitian ns={ns} (nf={nf_}+pad={pad})"
          f"  dt={dt_ns:.3f} ns  Tmax={ns*dt_ns:.0f} ns"
          f"  (display 0 to {args.tmax_ns:.0f} ns)")
    print(f"Workers      : {n_workers}")
    print(f"Output dir   : {out_dir}")
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

    print(f"\nSolving {nf_} FDFD problems (1 source × {nf_} frequencies) ...")
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
    # Compensates for the flat-spectrum unit source used by the Python FDFD solver.
    # MATLAB's RHS_TE1 embeds a frequency-dependent source amplitude;
    # this window replicates that effect.
    if apply_window:
        bh_win   = blackman_harris_spectrum(freqs)          # (nf,)
        precrnew = precrnew * bh_win[:, np.newaxis]         # broadcast over traces

    # ---- Hermitian IFFT → time domain (GPRFM algorithm) ----
    print("Applying Hermitian IFFT (MATLAB GPRFM style) ...")
    sg_full, t_ns_full = freq_to_timedomain(precrnew, freqs, pad=pad)

    sg_matrix = sg_full[:it_max, :]
    t_ns_disp = t_ns_full[:it_max]
    print(f"  Raw shot gather shape: {sg_matrix.shape}  "
          f"t = [0, {t_ns_disp[-1]:.1f}] ns")

    # ---- Spatial reorganization (CENTER-SHOT KEY) ----
    print("Applying center-shot spatial reorganization (MATLAB tmp1-tmp4) ...")
    sg_reorg, offsets_reorg = reorganize_center_shot(
        sg_matrix, src_ix, rec_indices, dh
    )
    print(f"  Reorganized shot gather shape: {sg_reorg.shape}")
    print(f"  Offset range: [{offsets_reorg[0]:.2f}, {offsets_reorg[-1]:.2f}] m")

    # ---- Save ----
    npz_path = out_dir / f"{stem}.npz"
    np.savez(npz_path,
             sg_raw=sg_matrix,
             offsets_raw=offsets_m,
             sg_reorg=sg_reorg,
             offsets_reorg=offsets_reorg,
             rec_indices=rec_indices,
             t_ns=t_ns_disp,
             dh=dh,
             freqs=freqs,
             precrnew=precrnew,
             src_ix=src_ix, src_iz=src_iz, rec_iz=rec_iz)
    print(f"Data saved -> {npz_path}")

    src_pos_m = float(src_ix * dh)

    # ---- Colour shot gather (GPRFM: spline upsample + AGC2 + polarity flip) ----
    print("Generating colour plot ...")
    png_path = out_dir / f"{stem}.png"
    plot_shotgather_color(
        sg_reorg, offsets_reorg, t_ns_disp,
        grid_style=grid_style,
        freqs_hz=freqs,
        save_path=png_path,
        src_pos_m=0.0,  # Source at zero offset after reorganization
    )
    print(f"Color plot  -> {png_path}")

    # ---- Wiggle shot gather ----
    print("Generating wiggle plot ...")
    wig_stem = (f"sg_center_wiggle_{grid_style}"
                f"_src{src_ix}-{src_iz}"
                f"_reciz{rec_iz}_step{rec_step}_{freq_tag}")
    wig_path = out_dir / f"{wig_stem}.png"
    plot_shotgather_wiggle(
        sg_reorg, offsets_reorg, t_ns_disp,
        grid_style=grid_style,
        freqs_hz=freqs,
        save_path=wig_path,
        src_pos_m=0.0,  # Source at zero offset after reorganization
        gain=args.wiggle_gain,
    )
    print(f"Wiggle plot -> {wig_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
