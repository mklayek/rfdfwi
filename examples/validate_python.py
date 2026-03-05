"""
Validation tests for the forward solver and inversion.

Tests
-----
test_forward_residual   Verify ``||A u - b|| < 1e-10`` (direct-solve accuracy).
test_misfit_decrease    Verify misfit does not increase over 3 FWI iterations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from scripts.forward_fdfd import build_helmholtz_2d, solve_forward, run_forward
from scripts.inversion_fwi import compute_misfit, run_inversion
from create_models.build_models import homogeneous_model
from scripts.config_loader import load_config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_forward_residual() -> None:
    """Forward solution must satisfy A @ u = b to machine precision (< 1e-10)."""
    nx, nz = 31, 21
    dx, dz = 0.02, 0.02
    epsr, sigma = homogeneous_model(nx, nz, 9.0, 0.01)
    omega = 2.0 * np.pi * 500e6

    A = build_helmholtz_2d(epsr, sigma, dx, dz, omega, 5, 5, use_gpu=False)
    sx, sz = nx // 2, 2
    u = solve_forward(A, sx, sz, nx, nz, use_gpu=False)

    b = np.zeros(nx * nz, dtype=np.complex128)
    b[sz * nx + sx] = 1.0
    residual_norm = np.linalg.norm(A @ u.ravel() - b)

    assert residual_norm < 1e-10, f"Forward residual too large: {residual_norm:.3e}"
    print(f"  [PASS] Forward residual = {residual_norm:.3e}  (< 1e-10)")


def test_misfit_decrease() -> None:
    """FWI misfit must not increase significantly after 3 iterations."""
    config_path = root / "input" / "input_inversion.yaml"
    if not config_path.exists():
        print("  [SKIP] No input/input_inversion.yaml found.")
        return

    config  = load_config(config_path)
    fwd_cfg = dict(config.get("forward", config))
    acq     = config.get("acquisition", {})

    fwd_cfg["sources"]   = acq.get("sources", [{"ix": 50, "iz": 2}])
    fwd_cfg["receivers"] = acq.get(
        "receivers", {"mode": "line", "iz": 2, "ix_start": 0, "ix_end": 100}
    )
    fwd_cfg["model"] = {"type": "homogeneous", "epsr": 9.0, "sigma": 0.01}
    d_obs, _ = run_forward(fwd_cfg)

    config["inversion"] = config.get("inversion", {})
    config["inversion"]["max_iter"] = 3
    _, _, history = run_inversion(config, d_obs, use_gpu=False)

    misfit = history["misfit"]
    assert misfit[-1] <= misfit[0] + 1e-6, (
        f"Misfit increased: {misfit[0]:.4e} → {misfit[-1]:.4e}"
    )
    print(f"  [PASS] Misfit: {misfit[0]:.4e} → {misfit[-1]:.4e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Running validation …\n")
    print("1. Forward residual check")
    test_forward_residual()
    print("\n2. Inversion misfit-decrease check")
    test_misfit_decrease()
    print("\nAll validation checks passed.")
    print(
        "\nTip: to compare with MATLAB, export observed data to obs/ and run forward "
        "with the same config; compare traces numerically."
    )


if __name__ == "__main__":
    main()
