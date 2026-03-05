"""
Run FDFD forward modelling for a zero-offset GPR B-scan (radargram).

For each source position the full 2-D wavefield is computed; the vertical
column at the source x-index (zero-offset trace) is extracted and stacked
to produce a depth-vs-position B-scan image.

Two plot formats are saved:
  1. Colour (image) B-scan  — seismic colormap, percentile-clipped colour scale.
  2. Wiggle B-scan          — normalised trace waveforms with positive-fill,
                              classic seismic / GPR A-scan display.

Output
------
results/forward/bscan/bscan_<tags>.npz        B-scan matrix (nz x n_sources), real.
results/forward/bscan/bscan_<tags>.png        Colour image B-scan.
results/forward/bscan/bscan_wiggle_<tags>.png Wiggle trace B-scan.
"""
from __future__ import annotations

import argparse
import sys
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

DEFAULT_OUT = root / "results" / "forward" / "bscan"
BSCAN_DPI   = 600


# ---------------------------------------------------------------------------
# Colour B-scan plot
# ---------------------------------------------------------------------------

def plot_bscan_color(
    bscan: np.ndarray,
    x_positions: np.ndarray,
    dh: float,
    grid_style: str,
    freq_hz: float,
    save_path: Path,
    clip_pct: float = 95.0,
    dpi: int = BSCAN_DPI,
) -> None:
    """
    Colour (image) B-scan: depth vs source position.

    Uses per-column (per-trace) normalisation so deep reflections are visible,
    then applies a percentile colour clip to suppress near-source singularity.

    Parameters
    ----------
    bscan     : (nz, n_src)  Raw real-valued B-scan matrix.
    clip_pct  : float        Percentile for symmetric colour limits (default 95).
    """
    nz, n_src = bscan.shape
    z = np.arange(nz) * dh
    x = x_positions

    # --- Per-trace (AGC) normalisation — equalises amplitude vs depth ---
    bscan_norm = bscan.copy()
    for col in range(n_src):
        t_max = np.max(np.abs(bscan_norm[:, col]))
        if t_max > 0:
            bscan_norm[:, col] /= t_max

    # Percentile symmetric colour limits on normalised data
    vmax = float(np.percentile(np.abs(bscan_norm), clip_pct))
    vmax = vmax if vmax > 0 else 1.0

    extent = [x[0], x[-1], z[-1], z[0]]

    domain_w = x[-1] - x[0]
    domain_h = z[-1]
    ratio    = domain_h / max(domain_w, 1e-9)
    fig_w    = 10.0
    fig_h    = max(fig_w * ratio, 4.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    im = ax.imshow(
        bscan_norm,
        extent=extent,
        aspect="auto",
        cmap="seismic",          # flipped: positive = red, negative = blue
        vmin=-vmax, vmax=vmax,
        interpolation="none",    # no blurring — keeps traces thin and crisp
    )

    ax.set_xlabel("Distance [m]", fontsize=14)
    ax.set_ylabel("Depth [m]",    fontsize=14)
    ax.tick_params(labelsize=12)
    freq_str = f"{freq_hz / 1e6:.0f} MHz" if freq_hz >= 1e6 else f"{freq_hz:.0f} Hz"
    ax.set_title(
        f"B-scan (zero-offset)  {grid_style}  ({freq_str})",
        fontsize=14, pad=6,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Re(Ey)  [norm.]", fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Wiggle B-scan plot
# ---------------------------------------------------------------------------

def plot_bscan_wiggle(
    bscan: np.ndarray,
    x_positions: np.ndarray,
    dh: float,
    grid_style: str,
    freq_hz: float,
    save_path: Path,
    gain: float = 1.5,
    dpi: int = BSCAN_DPI,
) -> None:
    """
    Wiggle-trace B-scan: each A-scan is drawn as a normalised waveform
    with positive amplitudes filled black — classic GPR / seismic display.

    Parameters
    ----------
    bscan        : (nz, n_src)  Raw real-valued B-scan matrix.
    x_positions  : (n_src,)     Physical x-coordinate of each source [m].
    dh           : float        Grid spacing [m] — used for depth axis.
    gain         : float        Wiggle amplitude scale relative to trace spacing
                                (1.0 = wiggles just touch adjacent traces).
    """
    nz, n_src = bscan.shape
    z = np.arange(nz) * dh

    # Trace spacing
    dx_trace = float(x_positions[1] - x_positions[0]) if n_src > 1 else dh
    wiggle_scale = gain * dx_trace

    # Per-trace normalisation (AGC) — each trace has peak amplitude = 1
    bscan_norm = bscan.copy()
    for col in range(n_src):
        t_max = np.max(np.abs(bscan_norm[:, col]))
        if t_max > 0:
            bscan_norm[:, col] /= t_max

    domain_w = x_positions[-1] - x_positions[0]
    domain_h = z[-1]
    ratio    = domain_h / max(domain_w, 1e-9)
    fig_w    = 12.0
    fig_h    = max(fig_w * ratio, 4.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    for col, x0 in enumerate(x_positions):
        trace  = bscan_norm[:, col]
        wiggle = x0 + wiggle_scale * trace

        # Waveform line
        ax.plot(wiggle, z, color="black", linewidth=0.4, alpha=0.85)

        # Positive-amplitude fill (black) — standard seismic/GPR wiggle
        ax.fill_betweenx(z, x0, wiggle,
                         where=(trace >= 0),
                         color="black", alpha=0.75, linewidth=0)

    ax.set_xlim(x_positions[0] - dx_trace, x_positions[-1] + dx_trace)
    ax.set_ylim(z[-1], z[0])   # depth increases downward
    ax.set_xlabel("Distance [m]", fontsize=14)
    ax.set_ylabel("Depth [m]",    fontsize=14)
    ax.tick_params(labelsize=12)
    freq_str = f"{freq_hz / 1e6:.0f} MHz" if freq_hz >= 1e6 else f"{freq_hz:.0f} Hz"
    ax.set_title(
        f"B-scan wiggle (zero-offset)  {grid_style}  ({freq_str})",
        fontsize=14, pad=6,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        print("  Run 'python examples/run_build_model.py' first, or supply --config.")
        return

    config     = load_config(config_path)
    fwd_cfg    = get_forward_config(config)
    nx, nz, dx, dz = get_domain(fwd_cfg)
    npx, npz   = get_pml(fwd_cfg)
    dh         = float(dx)
    freq_hz    = float(fwd_cfg.get("freq_hz", 50e6))
    grid_style = args.grid_style or fwd_cfg.get("grid_style", "stag1")
    src_iz     = args.src_iz
    step       = args.src_step

    # Source positions: stay inside PML on both sides
    src_ixs = list(range(npx + 2, nx - npx - 2, step))
    n_src   = len(src_ixs)
    x_pos   = np.array([ix * dh for ix in src_ixs])

    # Build filename stem encoding key CLI options
    freq_tag   = f"{int(freq_hz / 1e6)}MHz"
    stem       = f"bscan_{grid_style}_step{step}_iz{src_iz}_{freq_tag}"

    print(f"Config     : {config_path}")
    print(f"Domain     : {nx} x {nz}  (dh={dh} m)")
    print(f"PML        : npx={npx}, npz={npz}")
    print(f"Grid style : {grid_style}")
    print(f"Sources    : {n_src} positions  (step={step} cells, iz={src_iz})")
    print(f"Frequency  : {freq_hz / 1e6:.0f} MHz")
    print(f"Output stem: {stem}")

    # ---- B-scan assembly ----
    bscan   = np.zeros((nz, n_src), dtype=float)
    cfg_run = dict(fwd_cfg)
    cfg_run.setdefault("output", {})["save_fields"] = True
    imp_saved = False

    for col, src_ix in enumerate(src_ixs):
        cfg_src = dict(cfg_run)
        cfg_src["source"] = {"ix": src_ix, "iz": src_iz}

        if args.impedance_matrix and not imp_saved:
            imp_saved = True   # (imp_path saving omitted for brevity)

        _, Ez, _ = run_forward_single_source(
            cfg_src,
            use_gpu=args.use_gpu,
            grid_style=grid_style,
        )

        if Ez is not None:
            bscan[:, col] = np.real(Ez[:, src_ix])

        if (col + 1) % max(1, n_src // 10) == 0 or col == n_src - 1:
            print(f"  [{col + 1:3d}/{n_src}]  src_ix={src_ix}  "
                  f"|Re(Ey)|_max = {np.abs(bscan[:, col]).max():.3e}")

    # ---- Save raw B-scan array ----
    npz_path = out_dir / f"{stem}.npz"
    np.savez(npz_path, bscan=bscan, x_positions=x_pos, dh=dh, freq_hz=freq_hz)
    print(f"  Traces   -> {npz_path}")

    # ---- Colour B-scan ----
    png_path = out_dir / f"{stem}.png"
    plot_bscan_color(
        bscan, x_pos, dh,
        grid_style=grid_style,
        freq_hz=freq_hz,
        save_path=png_path,
        dpi=BSCAN_DPI,
    )
    print(f"  Color    -> {png_path}")

    # ---- Wiggle B-scan ----
    wig_path = out_dir / f"bscan_wiggle_{grid_style}_step{step}_iz{src_iz}_{freq_tag}.png"
    plot_bscan_wiggle(
        bscan, x_pos, dh,
        grid_style=grid_style,
        freq_hz=freq_hz,
        save_path=wig_path,
        dpi=BSCAN_DPI,
    )
    print(f"  Wiggle   -> {wig_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDFD zero-offset GPR B-scan (radargram) — colour + wiggle output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_kind="forward_bscan")
    parser.add_argument(
        "--src-step", type=int, default=2, metavar="N",
        help="Grid-cell step between successive source positions.",
    )
    parser.add_argument(
        "--src-iz", type=int, default=20, metavar="IZ",
        help="Source depth grid index (absolute, includes PML; default=20 = npz+10).",
    )
    parser.add_argument(
        "--wiggle-gain", type=float, default=1.5, metavar="G",
        help="Wiggle amplitude scale relative to trace spacing (default 1.5).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
