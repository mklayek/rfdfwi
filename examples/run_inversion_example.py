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
FDFD Full Waveform Inversion (FWI) — synthetic-data example.

Matches MATLAB RFDFWI.m workflow:

    1. Build true model from YAML config (mkl_two_cross by default).
    2. Generate synthetic observed data at GPRFM 10 discrete frequencies
       for all sources in the 4-sided acquisition.
    3. Build initial model (Gaussian-smoothed true model, or homogeneous
       from config initial_model section).
    4. Run FWI iteration loop (adjoint-state gradient + Tikhonov + line search).
    5. Save per-iteration images for all intermediate arrays + misfit curve + log.

Output
------
results/inversion/obs/d_obs.npz                      Observed data + metadata.
results/inversion/models/iter_<N>_epsr.png            Recovered εᵣ per iteration.
results/inversion/models/iter_<N>_sigma.png           Recovered σ per iteration.
results/inversion/gradient/iter_<N>_grad_epsr.png     Raw gradient εᵣ.
results/inversion/gradient/iter_<N>_grad_sigma.png    Raw gradient σ.
results/inversion/hessian/iter_<N>_hess_epsr.png      Pseudo-Hessian εᵣ.
results/inversion/hessian/iter_<N>_hess_sigma.png     Pseudo-Hessian σ.
results/inversion/search_direction/iter_<N>_dir_*.png Search direction εᵣ/σ.
results/inversion/tikhonov/iter_<N>_tikh_*.png        Tikhonov regularisation term.
results/inversion/models/final_result.npz             Final epsr + sigma arrays.
results/inversion/misfit/misfit_curve.png             L2 vs iteration plot.
results/inversion/misfit/misfit_history.npz           Misfit array.
results/inversion/logs/run_log.txt                    Run metadata and per-iter L2.
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts._cli import add_common_args
from scripts.config_loader import (
    load_config, get_forward_config, get_domain, get_pml,
)
from scripts.inversion_fwi import (
    GPRFM_FREQS_HZ, compute_forward_data, run_inversion,
)
from create_models.build_models import (
    build_model_from_config, build_4sided_acquisition,
)

DEFAULT_BASE = root / "results" / "inversion"

_matplotlib_imported = False


class _TeeLogger:
    """Duplicate stdout to a log file, prepending timestamps to file lines."""
    def __init__(self, path: Path, timestamps: bool = False) -> None:
        self._file = open(path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self._timestamps = timestamps
        self._at_sol = True   # True when next write is at start of a new line
        sys.stdout = self

    def write(self, msg: str) -> None:
        if not msg:
            return
        # Only prepend timestamp at the very start of a new line
        if self._at_sol and msg != "\n":
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"[{ts}] "
            self._file.write(prefix + msg)
            if self._timestamps:
                self._stdout.write(prefix + msg)
            else:
                self._stdout.write(msg)
        else:
            self._stdout.write(msg)
            self._file.write(msg)
        self._at_sol = msg.endswith("\n")
        self._file.flush()

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()


def _import_matplotlib():
    global _matplotlib_imported
    if not _matplotlib_imported:
        import matplotlib
        matplotlib.use("Agg")
        _matplotlib_imported = True


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FDFD FWI — synthetic observed data, GPRFM-style (matches RFDFWI.m).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(parser, default_kind="inversion")

    # True model override (for quick homogeneous tests; normally leave unset)
    parser.add_argument(
        "--true-epsr", type=float, default=None, metavar="V",
        help="Override: homogeneous true epsr.  Default: load inputmodel/model_epsr.npy "
             "(built by run_build_model.py), or fall back to YAML model section.",
    )
    parser.add_argument(
        "--true-sigma", type=float, default=None, metavar="V",
        help="Override: homogeneous true sigma [S/m].  Default: load inputmodel/model_sigma.npy.",
    )

    # Initial model override
    parser.add_argument(
        "--init-smooth", type=float, default=6.0, metavar="PX",
        help="Gaussian sigma [pixels] for smoothing true model → initial model. "
             "Set 0 to use config initial_model (no smoothing).",
    )
    parser.add_argument(
        "--init-epsr",  type=float, default=None, metavar="V",
        help="Override: homogeneous initial epsr (overrides --init-smooth).",
    )
    parser.add_argument(
        "--init-sigma", type=float, default=None, metavar="V",
        help="Override: homogeneous initial sigma [S/m] (overrides --init-smooth).",
    )

    # Frequency override
    parser.add_argument(
        "--gprfm-freqs", action="store_true", default=True,
        help="Use GPRFM 10 discrete frequencies [50,60,...,200] MHz (default).",
    )
    parser.add_argument(
        "--fc-low",  type=float, default=None, metavar="HZ",
        help="Start frequency for custom linspace sweep [Hz].",
    )
    parser.add_argument(
        "--fc-high", type=float, default=None, metavar="HZ",
        help="End frequency for custom linspace sweep [Hz].",
    )
    parser.add_argument(
        "--nf", type=int, default=None, metavar="N",
        help="Number of frequencies for custom linspace sweep.",
    )

    # Inversion parameter overrides
    parser.add_argument(
        "--max-iter", type=int, default=None, metavar="N",
        help="Override inversion.max_iter from config.",
    )
    parser.add_argument(
        "--lambda-sigma", type=float, default=None, metavar="V",
        help="Override Tikhonov LAMBDA_1 for sigma (MATLAB default: 2e-4).",
    )
    parser.add_argument(
        "--step-init", type=float, default=None, metavar="V",
        help="Override initial step size (<=0 for auto-scale, default: auto).",
    )

    # Output
    parser.add_argument(
        "--save-obs", action="store_true", default=True,
        help="Save observed data to d_obs.npz.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_model_image(
    arr: np.ndarray,
    dh: float,
    path: Path,
    title: str,
    cmap: str = "seismic",
    label: str = "",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    _import_matplotlib()
    import matplotlib.pyplot as plt

    nz, nx = arr.shape
    x = np.arange(nx) * dh
    z = np.arange(nz) * dh

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(
        arr,
        extent=[x[0], x[-1], z[-1], z[0]],
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Distance [m]", fontsize=12)
    ax.set_ylabel("Depth [m]",    fontsize=12)
    ax.set_title(title,           fontsize=13)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_diverging_image(
    arr: np.ndarray,
    dh: float,
    path: Path,
    title: str,
    cmap: str = "seismic",
    label: str = "",
) -> None:
    """Save a 2-D array with a symmetric (zero-centred) diverging colormap."""
    _import_matplotlib()
    import matplotlib.pyplot as plt

    nz, nx = arr.shape
    x = np.arange(nx) * dh
    z = np.arange(nz) * dh
    vmax = float(np.max(np.abs(arr)))
    if vmax == 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    im = ax.imshow(
        arr,
        extent=[x[0], x[-1], z[-1], z[0]],
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("Distance [m]", fontsize=12)
    ax.set_ylabel("Depth [m]",    fontsize=12)
    ax.set_title(title,           fontsize=13)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    if label:
        cbar.set_label(label, fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_misfit_plot(misfit: list[float], path: Path,
                      max_iter: int = None) -> None:
    _import_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    iters = np.arange(1, len(misfit) + 1)
    ax.semilogy(iters, misfit, "ko-", linewidth=1.5, markersize=5)
    # Highlight the last (most recent) point in red
    ax.plot(iters[-1], misfit[-1], "ro", markersize=10, zorder=5)
    # Annotate the last point with its value
    ax.annotate(f"{misfit[-1]:.3e}", xy=(iters[-1], misfit[-1]),
                xytext=(5, 5), textcoords="offset points", fontsize=9)
    # Faint vertical dashed line at expected total iterations
    if max_iter is not None:
        ax.axvline(x=max_iter, color="gray", linestyle="--", linewidth=0.8,
                   alpha=0.4)
    # Summary text box in the upper-right corner
    if len(misfit) > 1:
        ratio = misfit[-1] / misfit[0] if misfit[0] != 0 else float("nan")
        text = (
            f"L2[0]  = {misfit[0]:.3e}\n"
            f"L2[-1] = {misfit[-1]:.3e}\n"
            f"ratio  = {ratio:.3e}\n"
            f"iter   = {len(misfit)}"
        )
        ax.text(0.97, 0.97, text, transform=ax.transAxes, va="top", ha="right",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat",
                                      alpha=0.5))
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("L2 misfit (log scale)", fontsize=12)
    ax.set_title("FWI convergence curve", fontsize=13)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = _parse_args()
    t_start = datetime.datetime.now()

    # ---- Output directories ----
    base      = Path(args.results_dir) if args.results_dir else DEFAULT_BASE
    obs_dir   = base / "obs"
    models_dir = base / "models"
    grad_dir  = base / "gradient"
    hess_dir  = base / "hessian"
    sdir_dir  = base / "search_direction"
    tikh_dir  = base / "tikhonov"
    misfit_dir = base / "misfit"
    logs_dir  = base / "logs"
    for d in (obs_dir, models_dir, grad_dir, hess_dir, sdir_dir, tikh_dir,
              misfit_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # -- Progress log (captures all print output with timestamps) --
    _ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _progress_log = logs_dir / f"progress_{_ts}.txt"
    _logger = _TeeLogger(_progress_log, timestamps=True)
    print(f"Progress log : {_progress_log}")

    # ---- Config ----
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = root / "input" / "input_inversion.yaml"
        if not config_path.exists():
            config_path = root / "input" / "input_forward.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return

    config     = load_config(config_path)
    fwd_cfg    = get_forward_config(config)
    nx, nz, dx, _  = get_domain(fwd_cfg)
    npml, _    = get_pml(fwd_cfg)
    dh         = float(dx)
    a0_cfs     = float(fwd_cfg.get("pml", {}).get("a0_cfs", 9e8))
    grid_style = args.grid_style or fwd_cfg.get("grid_style", "stag1")
    n_workers  = int(getattr(args, "ncpus", 1))

    # ---- Apply CLI overrides to config ----
    inv_cfg = config.setdefault("inversion", {})
    if args.max_iter is not None:
        inv_cfg["max_iter"] = args.max_iter
    if args.lambda_sigma is not None:
        inv_cfg.setdefault("regularization", {})["lambda_sigma"] = args.lambda_sigma
    if args.step_init is not None:
        inv_cfg["step_init"] = args.step_init
    if args.patience is not None:
        inv_cfg["patience"] = args.patience
    if args.warmup is not None:
        inv_cfg["warmup_iters"] = args.warmup
    if args.step_epsr is not None:
        inv_cfg["step_init_epsr"] = args.step_epsr
    if args.step_sigma is not None:
        inv_cfg["step_init_sigma"] = args.step_sigma
    if args.c2_wolfe is not None:
        inv_cfg["c2_wolfe"] = args.c2_wolfe
    if args.nlbfgs is not None:
        inv_cfg["nlbfgs"] = args.nlbfgs
    if args.no_lbfgs:
        inv_cfg["use_lbfgs"] = False

    # ---- Frequencies ----
    if args.fc_low is not None or args.fc_high is not None or args.nf is not None:
        fs_cfg  = fwd_cfg.get("freq_sweep", {})
        fc_low  = args.fc_low  if args.fc_low  is not None else float(fs_cfg.get("fc_low",  50e6))
        fc_high = args.fc_high if args.fc_high is not None else float(fs_cfg.get("fc_high", 200e6))
        nf      = args.nf      if args.nf      is not None else 10
        freqs   = np.linspace(fc_low, fc_high, nf)
        inv_cfg["freqs_hz"] = freqs.tolist()
        freq_label = f"linspace {fc_low/1e6:.0f}-{fc_high/1e6:.0f} MHz  nf={nf}"
    else:
        freqs = np.array(GPRFM_FREQS_HZ, dtype=float)
        freq_label = f"GPRFM 10 discrete ({freqs[0]/1e6:.0f}-{freqs[-1]/1e6:.0f} MHz)"

    # ---- Acquisition ----
    acq_cfg = config.get("acquisition", {})
    if acq_cfg.get("mode") == "4sided":
        npml_acq = int(acq_cfg.get("npml", npml))
        nsrc_ps  = int(acq_cfg.get("nsrc_per_side", 20))
        nrec_ps  = int(acq_cfg.get("nrec_per_side", 40))
        src_list, rec_list = build_4sided_acquisition(npml_acq, nrec_ps, nsrc_ps)
        sources   = [(int(s["ix"]), int(s["iz"])) for s in src_list]
        receivers = [(int(r["ix"]), int(r["iz"])) for r in rec_list]
    else:
        src_list = acq_cfg.get("sources", [{"ix": 99, "iz": 20}])
        sources  = [(int(s["ix"]), int(s["iz"])) for s in src_list]
        rec_cfg  = acq_cfg.get("receivers", {})
        if isinstance(rec_cfg, dict) and rec_cfg.get("mode") == "line":
            iz_r = int(rec_cfg.get("iz", 20))
            xs   = int(rec_cfg.get("ix_start", 20))
            xe   = int(rec_cfg.get("ix_end", 179))
            receivers = [(ix, iz_r) for ix in range(xs, xe + 1)]
        else:
            receivers = [(int(r["ix"]), int(r["iz"])) for r in (rec_cfg or [])]

    n_src = len(sources)
    n_rec = len(receivers)
    n_freq = len(freqs)

    # ---- True model ----
    inputmodel_dir  = root / "inputmodel"
    saved_epsr_path = inputmodel_dir / "model_epsr.npy"
    saved_sig_path  = inputmodel_dir / "model_sigma.npy"

    if args.true_epsr is not None and args.true_sigma is not None:
        # Explicit homogeneous override (quick tests)
        true_epsr  = np.full((nz, nx), args.true_epsr)
        true_sigma = np.full((nz, nx), args.true_sigma)
        true_label = f"homogeneous epsr={args.true_epsr}, sigma={args.true_sigma} S/m"
    elif saved_epsr_path.exists() and saved_sig_path.exists():
        # Primary: load the mkl_two_cross model built by run_build_model.py
        true_epsr  = np.load(saved_epsr_path)
        true_sigma = np.load(saved_sig_path)
        true_label = f"inputmodel/model_epsr.npy  (shape {true_epsr.shape})"
        print(f"Loaded true model from {saved_epsr_path}")
    else:
        # Fallback: build from YAML config model section
        print(
            f"[WARN] inputmodel/model_epsr.npy not found.\n"
            f"       Run  python examples/run_build_model.py  first to build the\n"
            f"       mkl_two_cross model.  Falling back to YAML model section."
        )
        true_epsr, true_sigma = build_model_from_config(fwd_cfg, nx, nz)
        true_label = f"built from config ({fwd_cfg.get('model', {}).get('type', 'config')})"

    # ---- Save true model images ----
    _save_model_image(true_epsr, dh, models_dir / "true_model_epsr.png",
                      "True model — εᵣ", cmap="seismic", vmin=0.0, vmax=10.0,
                      label="Relative permittivity")
    _save_model_image(true_sigma * 1e3, dh, models_dir / "true_model_sigma.png",
                      "True model — σ [mS/m]", cmap="seismic", vmin=0.0, vmax=10.0,
                      label="Conductivity [mS/m]")
    print(f"True model : {true_label}")
    print(f"  -> {models_dir / 'true_model_epsr.png'}")

    # ---- Generate observed data ----
    print(f"\nGenerating observed data ({n_src} sources × {n_freq} freqs × {n_rec} receivers) ...")
    d_obs = compute_forward_data(
        true_epsr, true_sigma, dh, npml, a0_cfs, freqs,
        sources, receivers, grid_style=grid_style, n_workers=n_workers,
    )
    print(f"  d_obs shape: {d_obs.shape}  |max|={np.max(np.abs(d_obs)):.3e}")

    if args.save_obs:
        obs_path = obs_dir / "d_obs.npz"
        np.savez(obs_path, d_obs=d_obs, freqs=freqs,
                 sources=np.array(sources), receivers=np.array(receivers))
        print(f"  Saved -> {obs_path}")

    # ---- Initial model ----
    if args.init_epsr is not None and args.init_sigma is not None:
        epsr_init  = np.full((nz, nx), args.init_epsr)
        sigma_init = np.full((nz, nx), args.init_sigma)
        init_label = f"homogeneous epsr={args.init_epsr}, sigma={args.init_sigma}"
    elif args.init_smooth > 0.0:
        from scipy.ndimage import gaussian_filter
        epsr_init  = gaussian_filter(true_epsr,  sigma=args.init_smooth)
        sigma_init = gaussian_filter(true_sigma, sigma=args.init_smooth)
        init_label = f"Gaussian-smoothed true model (sigma={args.init_smooth} px)"
    else:
        init_cfg = config.get("initial_model", {})
        if isinstance(init_cfg, dict) and init_cfg.get("type") == "homogeneous":
            epsr_init  = np.full((nz, nx), float(init_cfg.get("epsr",  4.0)))
            sigma_init = np.full((nz, nx), float(init_cfg.get("sigma", 3e-3)))
            init_label = f"config initial_model (epsr={init_cfg.get('epsr',4)}, sigma={init_cfg.get('sigma',3e-3)})"
        else:
            epsr_init, sigma_init = build_model_from_config(
                config.get("initial_model", fwd_cfg), nx, nz
            )
            init_label = "config initial_model"

    _save_model_image(epsr_init, dh, models_dir / "#0initial_model_epsr.png",
                      "Initial model — εᵣ", cmap="seismic", vmin=0.0, vmax=10.0,
                      label="Relative permittivity")
    _save_model_image(sigma_init * 1e3, dh, models_dir / "#0initial_model_sigma.png",
                      "Initial model — σ [mS/m]", cmap="seismic", vmin=0.0, vmax=10.0,
                      label="Conductivity [mS/m]")
    print(f"\nInitial model: {init_label}")

    # ---- Per-iteration callback: save all intermediate images ----
    _live_misfit: list[float] = []

    def _callback(it: int, epsr_it: np.ndarray, sigma_it: np.ndarray,
                  extras: dict) -> None:
        L2 = extras.get("L2", float("nan"))
        _live_misfit.append(L2)
        _save_misfit_plot(_live_misfit, misfit_dir / "#Output_L2_ratio_curve.png",
                          max_iter=inv_cfg.get("max_iter", 20))

        # -- Updated models --
        _save_model_image(
            epsr_it, dh, models_dir / f"000Output_model_at_iteration={it:04d}_epsr.png",
            f"Recovered εᵣ — iter {it}  L2={L2:.3e}",
            cmap="seismic", vmin=0.0, vmax=10.0, label="Relative permittivity")
        _save_model_image(
            sigma_it * 1e3, dh, models_dir / f"000Output_model_at_iteration={it:04d}_sigma.png",
            f"Recovered σ [mS/m] — iter {it}  L2={L2:.3e}",
            cmap="seismic", vmin=0.0, vmax=10.0, label="Conductivity [mS/m]")

        # -- Raw gradient (diverging: positive=increase, negative=decrease) --
        g_e = extras.get("grad_epsr")
        g_s = extras.get("grad_sigma")
        if g_e is not None:
            _save_diverging_image(
                g_e, dh, grad_dir / f"02grad_iteration_{n_freq}_iter={it:04d}_epsr.png",
                f"Gradient εᵣ — iter {it}", label="∂L/∂εᵣ")
        if g_s is not None:
            _save_diverging_image(
                g_s, dh, grad_dir / f"02grad_iteration_{n_freq}_iter={it:04d}_sigma.png",
                f"Gradient σ — iter {it}", label="∂L/∂σ")

        # -- Pseudo-Hessian diagonal (non-negative → sequential cmap) --
        h_e = extras.get("hess_epsr")
        h_s = extras.get("hess_sigma")
        if h_e is not None:
            _save_model_image(
                h_e, dh, hess_dir / f"03HESS_iteration_{n_freq}_{it:04d}_epsr.png",
                f"Pseudo-Hessian εᵣ — iter {it}",
                cmap="inferno", label="|u|²·ω⁴")
        if h_s is not None:
            _save_model_image(
                h_s, dh, hess_dir / f"03HESS_iteration_{n_freq}_{it:04d}_sigma.png",
                f"Pseudo-Hessian σ — iter {it}",
                cmap="inferno", label="|u|²·ω²")

        # -- Search direction (diverging) --
        d_e = extras.get("dir_epsr")
        d_s = extras.get("dir_sigma")
        if d_e is not None:
            _save_diverging_image(
                d_e, dh, sdir_dir / f"04Hgrad_iteration_{it:04d}_epsr.png",
                f"Search direction εᵣ — iter {it}", label="p_εᵣ")
        if d_s is not None:
            _save_diverging_image(
                d_s, dh, sdir_dir / f"04Hgrad_iteration_{it:04d}_sigma.png",
                f"Search direction σ — iter {it}", label="p_σ")

        # -- Tikhonov regularisation term (diverging; epsr term is usually zero) --
        t_e = extras.get("tikh_epsr")
        t_s = extras.get("tikh_sigma")
        if t_e is not None and np.any(t_e != 0):
            _save_diverging_image(
                t_e, dh, tikh_dir / f"05Tikhonov_iter={it:04d}_epsr.png",
                f"Tikhonov εᵣ — iter {it}", label="Tikh εᵣ")
        if t_s is not None:
            _save_diverging_image(
                t_s, dh, tikh_dir / f"05Tikhonov_iter={it:04d}_sigma.png",
                f"Tikhonov σ — iter {it}", label="Tikh σ")

        print(f"  [Callback] iter {it:04d}: saved model/grad/hess/dir/tikh images"
              f"  (L2={L2:.4e})")

    # ---- Print run summary ----
    print(f"\nConfig       : {config_path}")
    print(f"Grid style   : {grid_style}")
    print(f"Domain       : {nx}×{nz}  dh={dh} m")
    print(f"PML          : npml={npml}  a0_cfs={a0_cfs:.2e}")
    print(f"Acquisition  : {n_src} sources, {n_rec} receivers")
    print(f"Frequencies  : {freq_label}")
    print(f"Workers      : {n_workers}")
    print(f"Max iter     : {inv_cfg.get('max_iter', 20)}")
    print(f"\nStarting FWI ...")

    # ---- Run inversion ----
    epsr_rec, sigma_rec, history = run_inversion(
        config, d_obs,
        epsr_init=epsr_init,
        sigma_init=sigma_init,
        use_gpu=args.use_gpu,
        n_workers=n_workers,
        grid_style=grid_style,
        iter_callback=_callback,
    )
    n_iters = len(history["misfit"])

    # ---- Save final model ----
    final_path = models_dir / "final_result.npz"
    np.savez(final_path, epsr=epsr_rec, sigma=sigma_rec,
             epsr_init=epsr_init, sigma_init=sigma_init,
             true_epsr=true_epsr, true_sigma=true_sigma,
             misfit=np.array(history["misfit"]))
    print(f"\nFinal model  -> {final_path}")

    _save_model_image(epsr_rec, dh,
                      models_dir / f"#Output_FINAL_Converged_Models_at_iteration={n_iters:04d}_epsr.png",
                      "Recovered εᵣ (final)", cmap="seismic", vmin=0.0, vmax=10.0,
                      label="Relative permittivity")
    _save_model_image(sigma_rec * 1e3, dh,
                      models_dir / f"#Output_FINAL_Converged_Models_at_iteration={n_iters:04d}_sigma.png",
                      "Recovered σ [mS/m] (final)", cmap="seismic", vmin=0.0, vmax=10.0,
                      label="Conductivity [mS/m]")

    # ---- Misfit plot ----
    misfit_path = misfit_dir / "#Output_L2_ratio_curve.png"
    np.savez(misfit_dir / "misfit_history.npz",
             misfit=np.array(history["misfit"]),
             step=np.array(history["step"]))
    print(f"Misfit curve -> {misfit_path}")

    # ---- Log ----
    t_end    = datetime.datetime.now()
    log_path = logs_dir / "run_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("rfdfwi FWI run log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Progress log : {_progress_log}\n")
        f.write(f"Date/time    : {t_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config       : {config_path}\n")
        f.write(f"Grid style   : {grid_style}\n")
        f.write(f"Workers      : {n_workers}\n")
        f.write(f"True model   : {true_label}\n")
        f.write(f"Init model   : {init_label}\n")
        f.write(f"Frequencies  : {freq_label}\n")
        f.write(f"Sources      : {n_src}  Receivers: {n_rec}\n")
        f.write(f"Iterations   : {n_iters}\n")
        if history["misfit"]:
            f.write(f"L2 initial   : {history['misfit'][0]:.6e}\n")
            f.write(f"L2 final     : {history['misfit'][-1]:.6e}\n")
            ratio = history["misfit"][-1] / history["misfit"][0] if history["misfit"][0] > 0 else 1
            f.write(f"L2 ratio     : {ratio:.3e}\n")
        f.write(f"Elapsed      : {(t_end - t_start).total_seconds():.1f} s\n")
        f.write("\nPer-iteration L2:\n")
        for i, m in enumerate(history["misfit"]):
            step_str = f"  step={history['step'][i]:.3e}" if i < len(history["step"]) else ""
            f.write(f"  iter {i+1:>4d}: {m:.6e}{step_str}\n")
    print(f"Log          -> {log_path}")

    if history["misfit"]:
        print(f"\nFinal L2 : {history['misfit'][-1]:.6e}")
        if len(history["misfit"]) > 1:
            print(f"L2 ratio : {history['misfit'][-1]/history['misfit'][0]:.3e}")

    _logger.close()


if __name__ == "__main__":
    main()
