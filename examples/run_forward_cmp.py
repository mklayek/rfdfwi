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
Run FDFD CMP (Common Mid-Point) gather — time-domain display.

Default mode uses the 10 GPRFM discrete frequencies [50,60,...,200] MHz,
matching GPRFM.m exactly.  Override with --fc-low / --fc-high / --nf for
a linspace frequency sweep instead.

Processing pipeline (matches GPRFM.m):
    1. FDFD solve at each frequency × offset → complex E[src_iz, rec_ix]
    2. Hermitian IFFT → time-domain CMP matrix
    3. 10× spline upsample  (GPRFM: interp2 'spline')
    4. AGC2 + polarity flip (GPRFM: outData = -agc2(outData, 60, 10))
    5. Colour plot (seismic colormap) + wiggle plot (positive fill)

Output
------
results/forward/cmp/cmp_<tags>.npz           CMP matrix + metadata.
results/forward/cmp/cmp_<tags>.png           Colour image CMP.
results/forward/cmp/cmp_wiggle_<tags>.png    Wiggle trace CMP.
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
    plot_cmp_color,
    plot_cmp_wiggle,
)

DEFAULT_OUT = root / "results" / "forward" / "cmp"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDFD CMP gather — GPRFM-style time-domain, colour + wiggle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_kind="forward_cmp")
    parser.add_argument(
        "--n-offsets", type=int, default=25, metavar="N",
        help="Number of half-offset positions.",
    )
    parser.add_argument(
        "--offset-min", type=float, default=0.1, metavar="M",
        help="Minimum half-offset [m].",
    )
    parser.add_argument(
        "--offset-max", type=float, default=2.0, metavar="M",
        help="Maximum half-offset [m].",
    )
    parser.add_argument(
        "--mid-ix", type=int, default=None, metavar="IX",
        help="Mid-point x grid index (default: nx//2).",
    )
    parser.add_argument(
        "--src-iz", type=int, default=20, metavar="IZ",
        help="Source/receiver depth grid index.",
    )
    parser.add_argument(
        "--fc-low", type=float, default=None, metavar="HZ",
        help="Start frequency [Hz]. If set, switches to linspace sweep "
             "(overrides GPRFM default frequencies).",
    )
    parser.add_argument(
        "--fc-high", type=float, default=None, metavar="HZ",
        help="End frequency [Hz]. If set, switches to linspace sweep.",
    )
    parser.add_argument(
        "--nf", type=int, default=None, metavar="N",
        help="Number of linspace frequencies. If set, switches to linspace sweep.",
    )
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
        "--no-window", action="store_true", default=False,
        help="Skip the Blackman-Harris spectral window (raw flat-spectrum IFFT; "
             "produces sinc-like waveforms with visible sidelobes).",
    )
    parser.add_argument(
        "--wiggle-gain", type=float, default=1.5, metavar="G",
        help="Wiggle amplitude scale relative to offset spacing.",
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

    # ---- Frequency array ----
    # Default: 10 GPRFM discrete frequencies matching GPRFM.m exactly.
    # If any of --fc-low / --fc-high / --nf is given → linspace sweep instead.
    use_linspace = (args.fc_low is not None
                    or args.fc_high is not None
                    or args.nf is not None)
    if use_linspace:
        fs_cfg  = get_freq_sweep(fwd_cfg)
        fc_low  = args.fc_low  if args.fc_low  is not None else fs_cfg["fc_low"]
        fc_high = args.fc_high if args.fc_high is not None else fs_cfg["fc_high"]
        nf      = args.nf      if args.nf      is not None else fs_cfg["nf"]
        freqs   = np.linspace(fc_low, fc_high, nf)
        freq_label = (f"linspace {fc_low/1e6:.0f}–{fc_high/1e6:.0f} MHz  nf={nf}"
                      f"  df={( fc_high-fc_low)/max(nf-1,1)/1e6:.3f} MHz")
        freq_tag   = f"{int(fc_low/1e6)}-{int(fc_high/1e6)}MHz_nf{nf}"
    else:
        freqs      = np.array(GPRFM_FREQS_HZ, dtype=float)
        freq_label = (f"GPRFM  {freqs[0]/1e6:.0f}–{freqs[-1]/1e6:.0f} MHz"
                      f"  ({len(freqs)} discrete)")
        freq_tag   = "gprfm10"

    nf_    = len(freqs)
    pad    = args.pad
    ns     = (nf_ + pad) * 2 - 1
    df_min = float(np.min(np.diff(freqs))) if nf_ > 1 else float(freqs[0])
    dt_ns  = 1e9 / (ns * df_min)
    it_max = min(int(args.tmax_ns / dt_ns) + 1, ns)

    # ---- CMP geometry ----
    mid          = args.mid_ix if args.mid_ix is not None else nx // 2
    src_iz       = args.src_iz
    half_offsets = np.linspace(args.offset_min, args.offset_max, args.n_offsets)
    n_off        = len(half_offsets)
    off_tag      = f"h{args.offset_min:.2f}-{args.offset_max:.2f}"
    stem         = f"cmp_{grid_style}_noff{n_off}_iz{src_iz}_{off_tag}_{freq_tag}"

    print(f"Config       : {config_path}")
    print(f"Domain       : {nx} x {nz}  (dh={dh} m)")
    print(f"PML          : npx={npx}, npz={npz}")
    print(f"Grid style   : {grid_style}")
    print(f"Mid-point    : ix={mid}  ({mid * dh:.2f} m)")
    print(f"Source/rec iz: {src_iz}  ({src_iz * dh:.2f} m)")
    print(f"Half-offsets : {n_off} in [{args.offset_min:.2f}, {args.offset_max:.2f}] m")
    print(f"Frequencies  : {freq_label}")
    print(f"Spectral win : {'Blackman-Harris' if not args.no_window else 'none (raw)'}")
    print(f"IFFT         : Hermitian ns={ns} (nf={nf_}+pad={pad})"
          f"  dt={dt_ns:.3f} ns  Tmax={ns*dt_ns:.0f} ns"
          f"  (display 0–{args.tmax_ns:.0f} ns)")
    print(f"Workers      : {n_workers}")
    print(f"Output stem  : {stem}")

    # ---- CMP assembly: freq_matrix[fi, col] = E(freqs[fi], half_offsets[col]) ----
    freq_matrix = np.zeros((nf_, n_off), dtype=complex)

    cfg_run = dict(fwd_cfg)
    cfg_run.setdefault("output", {})["save_fields"] = True

    def _solve(fi_col: tuple[int, int]):
        fi, col  = fi_col
        h        = half_offsets[col]
        h_cells  = int(round(h / dh))
        src_ix   = max(npx + 1, mid - h_cells)
        rec_ix   = min(nx - npx - 2, mid + h_cells)

        cfg_fc            = dict(cfg_run)
        cfg_fc["freq_hz"] = float(freqs[fi])
        cfg_fc["source"]  = {"ix": src_ix, "iz": src_iz}

        _, Ez, _ = run_forward_single_source(
            cfg_fc, use_gpu=args.use_gpu, grid_style=grid_style,
        )
        val = complex(Ez[src_iz, rec_ix]) if Ez is not None else 0j
        return fi, col, val

    tasks   = [(fi, col) for col in range(n_off) for fi in range(nf_)]
    n_tasks = len(tasks)
    print(f"Solving {n_tasks} FDFD problems ({nf_} freq × {n_off} offsets) ...")

    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for k, (fi, col, val) in enumerate(ex.map(_solve, tasks)):
                freq_matrix[fi, col] = val
                if (k + 1) % max(1, n_tasks // 20) == 0 or k == n_tasks - 1:
                    print(f"  [{k+1:4d}/{n_tasks}]")
    else:
        for k, (fi, col) in enumerate(tasks):
            _, _, val = _solve((fi, col))
            freq_matrix[fi, col] = val
            if (k + 1) % max(1, n_tasks // 20) == 0 or k == n_tasks - 1:
                print(f"  [{k+1:4d}/{n_tasks}]  h={half_offsets[col]:.2f}m"
                      f"  f={freqs[fi]/1e6:.0f} MHz  |E|={abs(val):.3e}")

    # ---- Blackman-Harris spectral window (wavelet shaping) ----
    # Compensates for the flat-spectrum unit source in the Python FDFD solver.
    if not args.no_window:
        bh_win     = blackman_harris_spectrum(freqs)        # (nf,)
        freq_matrix = freq_matrix * bh_win[:, np.newaxis]  # broadcast over offsets

    # ---- Hermitian IFFT → time domain (MATLAB GPRFM algorithm) ----
    print("Applying Hermitian IFFT (MATLAB GPRFM style) ...")
    cmp_full, t_ns_full = freq_to_timedomain(freq_matrix, freqs, pad=pad)

    cmp_matrix = cmp_full[:it_max, :]
    t_ns_disp  = t_ns_full[:it_max]
    print(f"  CMP shape: {cmp_matrix.shape}  t = [0, {t_ns_disp[-1]:.1f}] ns")

    # ---- Save raw CMP array ----
    npz_path = out_dir / f"{stem}.npz"
    np.savez(npz_path,
             cmp=cmp_matrix, half_offsets=half_offsets,
             t_ns=t_ns_disp, dh=dh,
             freqs=freqs, freq_matrix=freq_matrix, mid_ix=mid)
    print(f"  Traces   -> {npz_path}")

    # ---- Colour CMP (GPRFM: spline upsample + AGC2 + polarity flip) ----
    png_path = out_dir / f"{stem}.png"
    plot_cmp_color(
        cmp_matrix, half_offsets, t_ns_disp,
        grid_style=grid_style,
        freqs_hz=freqs,
        save_path=png_path,
    )
    print(f"  Color    -> {png_path}")

    # ---- Wiggle CMP ----
    wig_path = out_dir / f"cmp_wiggle_{grid_style}_noff{n_off}_iz{src_iz}_{off_tag}_{freq_tag}.png"
    plot_cmp_wiggle(
        cmp_matrix, half_offsets, t_ns_disp,
        grid_style=grid_style,
        freqs_hz=freqs,
        save_path=wig_path,
        gain=args.wiggle_gain,
    )
    print(f"  Wiggle   -> {wig_path}")


if __name__ == "__main__":
    main()
