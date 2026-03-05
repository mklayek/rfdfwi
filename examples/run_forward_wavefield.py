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
Run FDFD forward modelling over a frequency range and plot the 2-D wavefield.

Replicates the MATLAB-style figure "Wavefield_9pCFSPML_para_stag_FM":
  - Real part of Ez on the full computational grid (including PML region).
  - seismic colormap, symmetric colour limits (+/- max|Ey|).
  - White dashed PML inner-boundary rectangle with 'x' markers.
  - Axes labelled "Distance [m]" (x) and "Depth [m]" (y, increasing downward).
  - Title: "Real part of Ez wavefield <grid_style>".
  - Saved as high-resolution PNG and TIFF.

Frequencies are solved in parallel across --ncpus workers.
The wavefield shown is for the highest frequency (or last in the list).

Output
------
results/forward/wavefield/wavefield_real.png    PNG figure.
results/forward/wavefield/wavefield_real.tiff   TIFF figure (MATLAB-compatible).
results/forward/wavefield/impedance_matrix.npz  (only when --impedance-matrix is set)
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts._cli import add_common_args
from scripts.config_loader import load_config, get_forward_config, get_domain, get_pml
from scripts.forward_fdfd import run_forward_single_source

DEFAULT_OUT = root / "results" / "forward" / "wavefield"

# Figure DPI — matches MATLAB exportgraphics(...,'Resolution',600)
WAVEFIELD_DPI = 600


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDFD multi-frequency wavefield plot (MATLAB-style).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_kind="forward_wavefield")
    parser.add_argument(
        "--source-ix", type=int, default=None, metavar="IX",
        help="Source x grid index (overrides config value).",
    )
    parser.add_argument(
        "--source-iz", type=int, default=None, metavar="IZ",
        help="Source z grid index (overrides config value).",
    )
    parser.add_argument(
        "--freq-min", type=float, default=50e6, metavar="HZ",
        help="Start frequency [Hz] for the sweep (MATLAB: FC_low).",
    )
    parser.add_argument(
        "--freq-max", type=float, default=200e6, metavar="HZ",
        help="End frequency [Hz] for the sweep (MATLAB: FC_high).",
    )
    parser.add_argument(
        "--nf", type=int, default=50, metavar="N",
        help="Number of frequencies (MATLAB: nf=50). "
             "df=(freq_max-freq_min)/(nf-1). Overrides --freq-step.",
    )
    parser.add_argument(
        "--freq-step", type=float, default=None, metavar="HZ",
        help="Explicit frequency step [Hz] (overrides --nf when set).",
    )
    parser.add_argument(
        "--clip", type=float, default=2.5e-3, metavar="V",
        help="Blackman-Harris amplitude clip (MATLAB: clip=2.5e-3).",
    )
    parser.add_argument(
        "--clip1", type=float, default=1.0e-2, metavar="V",
        help="Secondary amplitude clip (MATLAB: clip1=1e-2).",
    )
    parser.add_argument(
        "--caxis", type=float, default=10.0, metavar="V",
        help="Fixed symmetric colour limits +-caxis [V/m] (MATLAB: cx1=10). "
             "Set 0 to use data maximum instead.",
    )
    parser.add_argument(
        "--no-tiff", action="store_true", default=False,
        help="Skip saving the TIFF output (save PNG only).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Wavefield figure
# ---------------------------------------------------------------------------

def plot_wavefield(
    Ez: np.ndarray,
    dx: float,
    dz: float,
    grid_style: str,
    freq_hz: float,
    caxis_max: float = 10.0,
    save_png: Path | None = None,
    save_tiff: Path | None = None,
    dpi: int = WAVEFIELD_DPI,
) -> plt.Figure:
    """
    MATLAB-style wavefield figure matching Wavefield_9pCFSPML_para_stag_FM.

    Replicates:
      imagesc(model1.x, model1.y, real(tew1))   -- FULL extended grid (includes PML)
      caxis([-cx1 cx1])  % cx1=10
      colormap(flipud(seismic)); grid on; axis ij; axis equal; axis tight
      FontSize=20; exportgraphics(...,'Resolution',600)

    Parameters
    ----------
    Ez        : ndarray (nz, nx), complex — full extended wavefield (includes PML)
    dx, dz    : float  Grid spacing [m].
    grid_style: str    Stencil label for title.
    freq_hz   : float  Frequency [Hz] shown in title.
    caxis_max : float  Symmetric colour limit [V/m]  (MATLAB: cx1=10).
                       Pass 0 to use max(|Ez|) instead.
    """
    nz, nx = Ez.shape
    ez_real = np.real(Ez)

    # MATLAB: imagesc(model1.x, model1.y, tew1) on the FULL extended grid
    # model1.x = dh*(1:nx_full) = [0.05, ..., 10.0 m]  (200 cells including PML)
    x = np.arange(1, nx + 1) * dx   # 0.05 → 10.0 m
    z = np.arange(1, nz + 1) * dz
    extent = [x[0], x[-1], z[-1], z[0]]

    # Colour limits — MATLAB: cx1=10; caxis([-cx1 cx1])
    vmax = float(caxis_max) if caxis_max > 0 else float(np.max(np.abs(ez_real))) or 1.0

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)

    # MATLAB: imagesc; colormap(flipud(seismic))  ->  seismic_r
    im = ax.imshow(
        ez_real,
        extent=extent,
        aspect="equal",
        cmap="seismic",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
    )

    # MATLAB: axis ij; axis equal; axis tight
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(z[-1], z[0])   # depth increases downward

    # MATLAB: grid on
    ax.grid(True, linewidth=0.5, color="gray", alpha=0.5)

    # MATLAB: FontSize=20
    ax.tick_params(labelsize=20)
    ax.set_xlabel("Distance [m]", fontsize=20)
    ax.set_ylabel("Depth [m]",    fontsize=20)
    freq_str = f"{freq_hz / 1e6:.0f} MHz" if freq_hz >= 1e6 else f"{freq_hz:.0f} Hz"
    ax.set_title(
        f"Real part of Ey wavefield  {grid_style}  ({freq_str})",
        fontsize=20,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Re(Ey) (V/m)", fontsize=20)
    cbar.ax.tick_params(labelsize=20)

    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=dpi, bbox_inches="tight")
        print(f"  PNG  -> {save_png}")

    if save_tiff is not None:
        save_tiff.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_tiff, dpi=dpi, bbox_inches="tight", format="tiff")
        print(f"  TIFF -> {save_tiff}")

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    out_dir = Path(args.results_dir) if args.results_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve config
    if args.config:
        config_path = Path(args.config)
    else:
        model_eps = root / "inputmodel" / "model_epsr.npy"
        config_path = (root / "input" / "input_forward_with_model.yaml"
                       if model_eps.exists() else
                       root / "input" / "input_forward.yaml")
        if not config_path.exists():
            config_path = root / "input" / "input_forward.yaml"

    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        print("  Run 'python examples/run_build_model.py' first, or supply --config.")
        return

    config  = load_config(config_path)
    fwd_cfg = get_forward_config(config)
    nx, nz, dx, dz = get_domain(fwd_cfg)
    npx, npz = get_pml(fwd_cfg)
    grid_style = args.grid_style or fwd_cfg.get("grid_style", "stag1")
    n_workers  = int(getattr(args, "ncpus", 1))

    # Source position (CLI overrides config)
    if args.source_ix is not None:
        source_ix = args.source_ix
    else:
        src_cfg = fwd_cfg.get("source", fwd_cfg.get("sources", [{}]))
        if isinstance(src_cfg, list):
            src_cfg = src_cfg[0]
        source_ix = int(src_cfg.get("ix", nx // 2))

    if args.source_iz is not None:
        source_iz = args.source_iz
    else:
        src_cfg = fwd_cfg.get("source", fwd_cfg.get("sources", [{}]))
        if isinstance(src_cfg, list):
            src_cfg = src_cfg[0]
        source_iz = int(src_cfg.get("iz", npz + 2))

    # Frequency sweep: nf logarithmically-spaced or explicit step
    # MATLAB: df = (FC_high - FC_low) / (nf-1), freqs = linspace(FC_low, FC_high, nf)
    if args.freq_step is not None:
        freqs = np.arange(args.freq_min, args.freq_max + args.freq_step * 0.5, args.freq_step)
        df_mhz = args.freq_step / 1e6
        mode_str = f"step {df_mhz:.3f} MHz"
    else:
        freqs = np.linspace(args.freq_min, args.freq_max, args.nf)
        df = (args.freq_max - args.freq_min) / max(args.nf - 1, 1)
        df_mhz = df / 1e6
        mode_str = f"nf={args.nf}, df={df_mhz:.3f} MHz"
    n_freqs = len(freqs)

    print(f"Config     : {config_path}")
    print(f"Domain     : {nx} x {nz}  (dx={dx} m, dz={dz} m)")
    print(f"PML        : npx={npx}, npz={npz} cells")
    print(f"Source     : ix={source_ix}, iz={source_iz}  "
          f"({source_ix * dx:.2f} m, {source_iz * dz:.2f} m)")
    print(f"Grid style : {grid_style}")
    print(f"Frequencies: {args.freq_min/1e6:.0f}-{args.freq_max/1e6:.0f} MHz "
          f"({mode_str}, {n_freqs} total)")
    print(f"Clip       : {args.clip:.2e}  clip1={args.clip1:.2e}")
    print(f"Workers    : {n_workers}")

    # Build per-frequency config (source fixed, frequency varies)
    cfg_run = dict(fwd_cfg)
    cfg_run.setdefault("output", {})["save_fields"] = True
    cfg_run["source"] = {"ix": source_ix, "iz": source_iz}

    def _solve_freq(f: float):
        cfg_f = dict(cfg_run)
        cfg_f["freq_hz"] = float(f)
        _, Ez, _ = run_forward_single_source(
            cfg_f, use_gpu=args.use_gpu, grid_style=grid_style,
        )
        return f, Ez

    print(f"Solving {n_freqs} frequencies ...")
    if n_workers > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            results = list(ex.map(_solve_freq, freqs))
    else:
        results = []
        for idx, f in enumerate(freqs):
            fmhz = f / 1e6
            _, Ez = _solve_freq(f)  # returns (freq, Ez)
            results.append((f, Ez))
            if (idx + 1) % max(1, n_freqs // 10) == 0 or idx == n_freqs - 1:
                vmax = float(np.max(np.abs(np.real(Ez)))) if Ez is not None else 0.0
                print(f"  [{idx+1:3d}/{n_freqs}]  {fmhz:.0f} MHz  "
                      f"|Re(Ey)|_max = {vmax:.4e} V/m")

    # Plot wavefield at the HIGHEST frequency (freq_max = 200 MHz) — matches MATLAB.
    # High frequency = short wavelength = many thin rings (like MATLAB figure).
    freq_plot, Ez_plot = results[-1]

    if Ez_plot is None:
        print("[ERROR] Wavefield not returned. Check save_fields is enabled.")
        return

    if args.verbose:
        vmax = float(np.max(np.abs(np.real(Ez_plot))))
        print(f"  Ez shape : {Ez_plot.shape}  |Re(Ey)|_max = {vmax:.4e} V/m")

    # Build filename encoding key CLI options, e.g.:
    #   wavefield_stag2_src99-20_200MHz_cx10.png
    stencil_tag = "stag2" if getattr(args, "grid_style", "stag1") == "stag2" else grid_style
    src_tag  = f"src{source_ix}-{source_iz}"
    freq_tag = f"{int(freq_plot/1e6)}MHz"
    cx_tag   = f"cx{int(args.caxis)}" if args.caxis > 0 else "cxauto"
    stem = f"wavefield_{stencil_tag}_{src_tag}_{freq_tag}_{cx_tag}"

    save_png  = out_dir / f"{stem}.png"
    save_tiff = None if args.no_tiff else out_dir / f"{stem}.tiff"

    vmax_full = float(np.max(np.abs(np.real(Ez_plot))))
    print(f"  |Re(Ey)|_max (full grid) = {vmax_full:.4e} V/m  (caxis +/-{args.caxis})")
    print("Saving wavefield figure ...")
    plot_wavefield(
        Ez_plot,
        dx=dx,
        dz=dz,
        grid_style=grid_style,
        freq_hz=freq_plot,
        caxis_max=args.caxis,
        save_png=save_png,
        save_tiff=save_tiff,
        dpi=WAVEFIELD_DPI,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
