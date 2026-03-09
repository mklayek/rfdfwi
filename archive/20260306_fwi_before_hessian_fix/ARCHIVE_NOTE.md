# Archive — 2026-03-06 — Before Hessian Preconditioning Fix

## Reason for archiving
The FWI code produced a flat misfit curve (L2 ratio = 1.000 across 16 iterations).
Root cause: pseudo-Hessian was computed but never applied to precondition the gradient.
Raw gradient magnitudes (~1e+20) caused auto-scaled step ~1e-42 → model never updated.

## Files archived
- `inversion_fwi.py`         — FWI core (compute_gradient, run_inversion)
- `run_inversion_example.py` — FWI example driver script

## Fix applied (in live scripts)
Hessian preconditioning added to search direction in `run_inversion`:

    H_max    = max(hess_epsr.max(), hess_sigma.max())
    H_eps    = 1e-5 * H_max
    dir_epsr  = -g_epsr  / (hess_epsr  + H_eps)
    dir_sigma = -g_sigma / (hess_sigma + H_eps)

Step initialisation changed from `L2 / ||g_raw||²` → fixed `1.0`
(H-preconditioned direction is already O(1), so step=1 is correct starting point).

Armijo descent slope corrected to use `g·d` (proper inner product):

    descent_slope = sum(g_epsr * dir_epsr + g_sigma * dir_sigma)   # < 0
    armijo_rhs    = -c1 * step * descent_slope                     # > 0
