"""
Build a 2-D GPR model (permittivity and conductivity) from config and save
to inputmodel/ as NumPy arrays and PNG figures.

Step 1 of the rfdfwi workflow::

    python examples/run_build_model.py
    python examples/run_forward_bscan.py
    python examples/run_inversion_example.py

Output
------
inputmodel/model_epsr.npy     Relative permittivity grid (nz, nx).
inputmodel/model_sigma.npy    Conductivity grid (nz, nx) [S/m].
inputmodel/model_eps_sig.png  Two-panel model figure (eps + sigma).
inputmodel/model_epsr.png     Permittivity-only figure.
inputmodel/model_sigma.png    Conductivity-only figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts.config_loader import load_config, get_forward_config, get_domain
from create_models.build_models import build_model_from_config
from scripts.plot_utils import setup_figure, save_figure, draw_pml_boundary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and save a 2-D GPR model from a YAML config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None, metavar="FILE",
        help="Path to YAML config. Defaults to input/input_forward.yaml.",
    )
    return parser.parse_args()


def save_model_images(
    epsr: np.ndarray,
    sigma: np.ndarray,
    out_dir: Path,
    dx: float,
    dz: float,
    npx: int = 10,
    npz: int = 10,
) -> None:
    """Save combined two-panel figure and individual PNG files."""
    nz, nx = epsr.shape
    extent    = [0.0, (nx - 1) * dx, (nz - 1) * dz, 0.0]
    clim_eps  = (0.0, 10.0)
    clim_mS   = (0.0, 10.0)
    sigma_mS  = sigma * 1000.0

    # PML inner-boundary coordinates
    xi_min = npx * dx
    xi_max = (nx - npx - 1) * dx
    zi_min = npz * dz
    zi_max = (nz - npz - 1) * dz

    # ---------- two-panel combined figure ----------
    fig, axes = setup_figure(1, 2, figsize=(12, 5))
    for ax, data, clim, label, ttl in [
        (axes[0], epsr,     clim_eps, "Relative permittivity", "eps model para"),
        (axes[1], sigma_mS, clim_mS,  "Conductivity (mS/m)",  "sig model para"),
    ]:
        im = ax.imshow(data, extent=extent, aspect="equal", cmap="seismic", clim=clim)
        ax.set_xlabel("Distance [m]", fontsize=11)
        ax.set_ylabel("Depth [m]",    fontsize=11)
        ax.set_title(ttl, fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.8, label=label)
        draw_pml_boundary(ax, xi_min, xi_max, zi_min, zi_max, color="black")
    save_figure(fig, out_dir / "model_eps_sig.png")

    # ---------- individual figures ----------
    for data, clim, label, fname in [
        (epsr,     clim_eps, "Relative permittivity", "model_epsr.png"),
        (sigma_mS, clim_mS,  "Conductivity (mS/m)",  "model_sigma.png"),
    ]:
        f, a = setup_figure(1, 1, figsize=(6, 5))
        im = a[0].imshow(data, extent=extent, aspect="equal", cmap="seismic", clim=clim)
        a[0].set_xlabel("Distance [m]"); a[0].set_ylabel("Depth [m]")
        a[0].set_title(label)
        f.colorbar(im, ax=a[0], shrink=0.8, label=label)
        draw_pml_boundary(a[0], xi_min, xi_max, zi_min, zi_max, color="black")
        save_figure(f, out_dir / fname)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config) if args.config else root / "input" / "input_forward.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config not found: {config_path}")
        return

    config  = load_config(config_path)
    fwd_cfg = get_forward_config(config)
    nx, nz, dx, dz = get_domain(fwd_cfg)
    npx = int(fwd_cfg.get("pml", {}).get("npx", 10))
    npz = int(fwd_cfg.get("pml", {}).get("npz", 10))

    print(f"Config : {config_path}")
    print(f"Domain : {nx} x {nz}  (dx={dx} m, dz={dz} m)  PML: {npx}/{npz} cells")

    epsr, sigma = build_model_from_config(fwd_cfg, nx, nz)
    print(f"Model  : epsr [{epsr.min():.2f}, {epsr.max():.2f}]  "
          f"sigma [{sigma.min():.4f}, {sigma.max():.4f}] S/m")

    out_dir = root / "inputmodel"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "model_epsr.npy",  epsr)
    np.save(out_dir / "model_sigma.npy", sigma)
    print(f"\nArrays -> {out_dir}/model_epsr.npy  (shape {epsr.shape})")
    print(f"         {out_dir}/model_sigma.npy")

    save_model_images(epsr, sigma, out_dir, dx, dz, npx, npz)
    print(f"Figs   -> model_eps_sig.png  model_epsr.png  model_sigma.png")
    print(f"\nNext: python examples/run_forward_bscan.py")


if __name__ == "__main__":
    main()
