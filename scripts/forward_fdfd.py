"""
2D FDFD forward solver for GPR (TM mode, Ez).

Implements two 9-point CFS-PML staggered-grid Helmholtz formulations:

  stag1 — Hustedt et al. (2004) parallel staggered grid
           (port of imp_A_TE_9p_cfs_PML_stag1_para.m)
  stag2 — Layek & Sengupta (2023) new staggered grid
           (port of imp_A_TE_9p_cfs_PML_stag2_para.m)

PML coefficients follow Kuzuoglu & Mittra (1996) CFS-PML with cosine/sine
damping profiles (port of PML_9p_CFS_stag_para.m).

Copyright of the original MATLAB formulations: Mrinal Kanti Layek.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sp_linalg

# Optional GPU: CuPy for future GPU sparse/dense solve; default remains CPU
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

# Physical constants
MU0: float = 4e-7 * np.pi       # H/m
EPS0: float = 8.854187817e-12   # F/m

# Project root for cross-package imports
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _idx(ix: int, iz: int, nx: int) -> int:
    """Lexicographic (row-major) flat index for grid point (ix, iz)."""
    return iz * nx + ix


# ---------------------------------------------------------------------------
# CFS-PML coefficient arrays
# ---------------------------------------------------------------------------

def _build_cfspml_coeffs(
    nx: int,
    nz: int,
    npml: int,
    dh: float,
    omega: float,
    a0_cfs: float,
    free_surface: bool = False,
) -> dict[str, np.ndarray]:
    """
    Compute CFS-PML coefficient arrays for the 9-point staggered-grid schemes.

    Based on Kuzuoglu & Mittra (1996) cosine/sine damping profiles.
    Direct port of ``PML_9p_CFS_stag_para.m``.

    Parameters
    ----------
    nx, nz : int
        Grid dimensions (x = horizontal columns, z = depth rows).
    npml : int
        PML thickness in grid cells (same on all sides).
    dh : float
        Uniform grid spacing [m].
    omega : float
        Angular frequency [rad/s].
    a0_cfs : float
        CFS-PML maximum conductivity sigma_max.
    free_surface : bool
        If True the top-boundary PML is suppressed (free surface BC).

    Returns
    -------
    dict with keys P, Q, Px, Qx, Qy, Px1, Qx1 — all shape (nz, nx), complex128.
    """
    lPML = npml * dh
    pih = np.pi / 2.0
    alpha_max = 0.2
    kappax = 2.0
    kappay = 2.0

    sigma_x = np.zeros(nx)
    alpha_x = np.zeros(nx)
    sigma_y = np.zeros(nz)
    alpha_y = np.zeros(nz)

    # ---- Left PML: MATLAB i = 1..npml-1  ->  Python ix = 0..npml-2 ----
    if npml > 1:
        ix_l = np.arange(npml - 1)
        arg = (npml - 1 - ix_l) * dh * pih / lPML
        sigma_x[ix_l] = a0_cfs * np.cos(arg)
        alpha_x[ix_l] = -alpha_max * np.sin(arg)

    # ---- Right PML: MATLAB i = nx-npml..nx  ->  Python ix = nx-npml-1..nx-1 ----
    if npml > 0:
        ix_r = np.arange(nx - npml - 1, nx)
        arg = (nx - npml - ix_r) * dh * pih / lPML
        sigma_x[ix_r] = a0_cfs * np.cos(arg)
        alpha_x[ix_r] = -alpha_max * np.sin(arg)

    # ---- Top PML (skipped if free surface): MATLAB j = 1..npml-1 -> Python iz = 0..npml-2 ----
    if npml > 1 and not free_surface:
        iz_t = np.arange(npml - 1)
        arg = (npml - 1 - iz_t) * dh * pih / lPML
        sigma_y[iz_t] = a0_cfs * np.cos(arg)
        alpha_y[iz_t] = -alpha_max * np.sin(arg)

    # ---- Bottom PML: MATLAB j = ny-npml..ny  ->  Python iz = nz-npml-1..nz-1 ----
    if npml > 0:
        iz_b = np.arange(nz - npml - 1, nz)
        arg = (nz - npml - iz_b) * dh * pih / lPML
        sigma_y[iz_b] = a0_cfs * np.cos(arg)
        alpha_y[iz_b] = -alpha_max * np.sin(arg)

    # Complex stretch factors (1-D profiles broadcast to full 2-D grids)
    # sx varies only in x; sy varies only in z — expand both to (nz, nx)
    # MATLAB uses eps01=1 (not eps0=8.85e-12), so denominator is (alpha + omega).
    sx_1d = kappax + sigma_x * 1j / (alpha_x + omega)   # shape (nx,)
    sy_1d = kappay + sigma_y * 1j / (alpha_y + omega)   # shape (nz,)

    sx = np.broadcast_to(sx_1d[np.newaxis, :], (nz, nx)).copy()   # (nz, nx)
    sy = np.broadcast_to(sy_1d[:, np.newaxis], (nz, nx)).copy()   # (nz, nx)

    P = 1.0 / sx   # shape (nz, nx)
    Q = 1.0 / sy   # shape (nz, nx)

    # Staggered harmonic-mean averages (initialised to P/Q, then interior updated)
    # MATLAB loop: j = 2..ny-1, i = 2..nx-1  ->  Python: iz = 1..nz-2, ix = 1..nx-2
    Px = P.copy()
    Qx = Q.copy()
    Qy = Q.copy()
    Px1 = np.zeros_like(P)
    Qx1 = np.zeros_like(Q)

    s = (slice(1, nz - 1), slice(1, nx - 1))   # interior slice

    # Px[iz, ix]  = harmonic mean of P[iz, ix] and P[iz, ix+1]   (staggered in x)
    Px[s] = 2.0 / (1.0 / P[s] + 1.0 / P[1:nz-1, 2:nx])

    # Qx[iz, ix]  = harmonic mean of Q[iz, ix] and Q[iz, ix+1]   (staggered in x)
    Qx[s] = 2.0 / (1.0 / Q[s] + 1.0 / Q[1:nz-1, 2:nx])

    # Qy[iz, ix]  = harmonic mean of Q[iz, ix] and Q[iz+1, ix]   (staggered in z)
    Qy[s] = 2.0 / (1.0 / Q[s] + 1.0 / Q[2:nz, 1:nx-1])

    # Px1[iz, ix] = harmonic mean of P[iz, ix] and P[iz+1, ix+1] (diagonal)
    Px1[s] = 2.0 / (1.0 / P[s] + 1.0 / P[2:nz, 2:nx])

    # Qx1[iz, ix] = harmonic mean of Q[iz, ix] and Q[iz+1, ix+1] (diagonal)
    Qx1[s] = 2.0 / (1.0 / Q[s] + 1.0 / Q[2:nz, 2:nx])

    return {"P": P, "Q": Q, "Px": Px, "Qx": Qx, "Qy": Qy, "Px1": Px1, "Qx1": Qx1}


# ---------------------------------------------------------------------------
# 9-point stencil assembly — stag1 (Hustedt et al. 2004)
# ---------------------------------------------------------------------------

def _build_helmholtz_stag1(
    epsr: np.ndarray,
    sigma: np.ndarray,
    dh: float,
    omega: float,
    pml: dict[str, np.ndarray],
) -> sparse.csr_matrix:
    """
    Assemble the 9-point Helmholtz impedance matrix using the stag1 scheme.

    Port of ``imp_A_TE_9p_cfs_PML_stag1_para.m`` (Hustedt et al. 2004).
    Stencil coefficients: a=0.5461, c=0.6248, d=0.9381e-1,
    e=(1-c-4d)/4.  MATLAB convention: eps_c = eps0*epsr + j*sigma/omega.
    """
    nz, nx = epsr.shape
    N = nx * nz
    omega2 = omega ** 2
    idh2 = 1.0 / dh ** 2
    a = 0.5461
    c = 0.6248
    d = 0.9381e-1
    e = (1.0 - c - 4.0 * d) / 4.0

    P, Q = pml["P"], pml["Q"]
    Px, Qx = pml["Px"], pml["Qx"]
    Qy = pml["Qy"]
    Px1, Qx1 = pml["Px1"], pml["Qx1"]

    # Complex permittivity (MATLAB convention: +j*sigma/omega)
    eps_c = EPS0 * epsr + 1j * sigma / omega

    # Flat row-index grid: K[iz, ix] = iz*nx + ix
    iz2d, ix2d = np.mgrid[0:nz, 0:nx]
    K = (iz2d * nx + ix2d).astype(np.intp)

    rows_l, cols_l, vals_l = [], [], []

    def _push(iz_sl, ix_sl, d_iz, d_ix, v):
        r = K[iz_sl, ix_sl].ravel()
        rows_l.append(r)
        cols_l.append(r + d_iz * nx + d_ix)
        vals_l.append(v.ravel())

    # ---- NW: iz>0, ix>0  |  neighbour (iz-1, ix-1) ----
    # coef = (1-a)*idh2/4 * Qx1[iz-1, ix-1]*Q[iz, ix] + e*MU0*omega2*eps_c[iz-1, ix-1]
    _push(slice(1, nz), slice(1, nx), -1, -1,
          ((1-a)*idh2/4) * Qx1[:-1, :-1] * Q[1:, 1:]
          + e * MU0 * omega2 * eps_c[:-1, :-1])

    # ---- N: iz>0 (stag1: only iz condition)  |  neighbour (iz-1, ix) ----
    # coef = a*idh2 * Qx[iz-1, ix]*Q[iz, ix] + d*MU0*omega2*eps_c[iz-1, ix]
    _push(slice(1, nz), slice(None), -1, 0,
          a * idh2 * Qx[:-1, :] * Q[1:, :]
          + d * MU0 * omega2 * eps_c[:-1, :])

    # ---- NE: iz>0, ix<nx-1  |  neighbour (iz-1, ix+1) ----
    # coef = (1-a)*idh2/4 * Px1[iz-1, ix]*P[iz, ix] + e*MU0*omega2*eps_c[iz-1, ix+1]
    _push(slice(1, nz), slice(0, nx-1), -1, +1,
          ((1-a)*idh2/4) * Px1[:-1, :-1] * P[1:, :-1]
          + e * MU0 * omega2 * eps_c[:-1, 1:])

    # ---- W: ix>0 (stag1: only ix condition)  |  neighbour (iz, ix-1) ----
    # coef = a*idh2 * Px[iz, ix-1]*P[iz, ix] + d*MU0*omega2*eps_c[iz, ix-1]
    _push(slice(None), slice(1, nx), 0, -1,
          a * idh2 * Px[:, :-1] * P[:, 1:]
          + d * MU0 * omega2 * eps_c[:, :-1])

    # ---- Center: iz>0, ix>0, iz<nz-1, ix<nx-1 ----
    ir, ic = slice(1, nz-1), slice(1, nx-1)
    axis_sum = (
        Px[1:nz-1, 0:nx-2] * P[ir, ic]   # Px[iz, ix-1]
        + Px[ir, ic]        * P[ir, ic]   # Px[iz, ix]
        + Qy[ir, ic]        * Q[ir, ic]   # Qy[iz, ix]
        + Qy[0:nz-2, ic]    * Q[ir, ic]   # Qy[iz-1, ix]
    )
    diag_sum = (
        Px1[0:nz-2, ic]     * P[ir, ic]   # Px1[iz-1, ix]
        + Px1[ir, 0:nx-2]   * P[ir, ic]   # Px1[iz, ix-1]
        + Qx1[ir, ic]       * Q[ir, ic]   # Qx1[iz, ix]
        + Qx1[0:nz-2, 0:nx-2] * Q[ir, ic] # Qx1[iz-1, ix-1]
    )
    _push(ir, ic, 0, 0,
          c * MU0 * omega2 * eps_c[ir, ic]
          - a * idh2 * axis_sum
          - ((1-a)*idh2/4) * diag_sum)

    # ---- E: ix<nx-1 (stag1: only ix condition)  |  neighbour (iz, ix+1) ----
    # coef = a*idh2 * Px[iz, ix]*P[iz, ix] + d*MU0*omega2*eps_c[iz, ix+1]
    _push(slice(None), slice(0, nx-1), 0, +1,
          a * idh2 * Px[:, :-1] * P[:, :-1]
          + d * MU0 * omega2 * eps_c[:, 1:])

    # ---- SW: iz<nz-1, ix>0  |  neighbour (iz+1, ix-1) ----
    # coef = (1-a)*idh2/4 * Px1[iz, ix-1]*P[iz, ix] + e*MU0*omega2*eps_c[iz+1, ix-1]
    _push(slice(0, nz-1), slice(1, nx), +1, -1,
          ((1-a)*idh2/4) * Px1[:-1, :-1] * P[:-1, 1:]
          + e * MU0 * omega2 * eps_c[1:, :-1])

    # ---- S: iz<nz-1 (stag1: only iz condition)  |  neighbour (iz+1, ix) ----
    # coef = a*idh2 * Qy[iz, ix]*Q[iz, ix] + d*MU0*omega2*eps_c[iz+1, ix]
    _push(slice(0, nz-1), slice(None), +1, 0,
          a * idh2 * Qy[:-1, :] * Q[:-1, :]
          + d * MU0 * omega2 * eps_c[1:, :])

    # ---- SE: iz<nz-1, ix<nx-1  |  neighbour (iz+1, ix+1) ----
    # coef = (1-a)*idh2/4 * Qx1[iz, ix]*Q[iz, ix] + e*MU0*omega2*eps_c[iz+1, ix+1]
    _push(slice(0, nz-1), slice(0, nx-1), +1, +1,
          ((1-a)*idh2/4) * Qx1[:-1, :-1] * Q[:-1, :-1]
          + e * MU0 * omega2 * eps_c[1:, 1:])

    row_arr = np.concatenate(rows_l)
    col_arr = np.concatenate(cols_l)
    val_arr = np.concatenate(vals_l)
    return sparse.csr_matrix((val_arr, (row_arr, col_arr)), shape=(N, N))


# ---------------------------------------------------------------------------
# 9-point stencil assembly — stag2 (Layek & Sengupta 2023)
# ---------------------------------------------------------------------------

def _build_helmholtz_stag2(
    epsr: np.ndarray,
    sigma: np.ndarray,
    dh: float,
    omega: float,
    pml: dict[str, np.ndarray],
) -> sparse.csr_matrix:
    """
    Assemble the 9-point Helmholtz impedance matrix using the stag2 scheme.

    Port of ``imp_A_TE_9p_cfs_PML_stag2_para.m`` (Layek & Sengupta 2023).
    Same stencil coefficients as stag1.  Key difference: off-axis (N/S/E/W)
    entries mix Qy and Px cross-products with a (1-a)/4 correction term;
    diagonal entries (NW/NE/SW/SE) use Qy*Q + Px*P products rather than the
    Qx1/Px1 diagonal averages used by stag1.
    """
    nz, nx = epsr.shape
    N = nx * nz
    omega2 = omega ** 2
    idh2 = 1.0 / dh ** 2
    a = 0.5461
    c = 0.6248
    d = 0.9381e-1
    e = (1.0 - c - 4.0 * d) / 4.0

    P, Q = pml["P"], pml["Q"]
    Px = pml["Px"]
    Qy = pml["Qy"]

    eps_c = EPS0 * epsr + 1j * sigma / omega

    iz2d, ix2d = np.mgrid[0:nz, 0:nx]
    K = (iz2d * nx + ix2d).astype(np.intp)

    rows_l, cols_l, vals_l = [], [], []

    def _push(iz_sl, ix_sl, d_iz, d_ix, v):
        r = K[iz_sl, ix_sl].ravel()
        rows_l.append(r)
        cols_l.append(r + d_iz * nx + d_ix)
        vals_l.append(v.ravel())

    # ---- NW: iz>0, ix>0  |  neighbour (iz-1, ix-1) ----
    # stag2: (1-a)*idh2/4 * (Qy[iz-1,ix]*Q[iz,ix] + Px[iz,ix-1]*P[iz,ix])
    #        + e*MU0*omega2*eps_c[iz-1, ix-1]
    _push(slice(1, nz), slice(1, nx), -1, -1,
          ((1-a)*idh2/4) * (Qy[:-1, 1:] * Q[1:, 1:] + Px[1:, :-1] * P[1:, 1:])
          + e * MU0 * omega2 * eps_c[:-1, :-1])

    # ---- N: iz>0, ix>0 (stag2: both conditions)  |  neighbour (iz-1, ix) ----
    # stag2: a*idh2*Qy[iz-1,ix]*Q[iz,ix]
    #        + (1-a)*idh2/4*(2*Qy[iz-1,ix]*Q - Px[iz,ix]*P - Px[iz,ix-1]*P)
    #        + d*MU0*omega2*eps_c[iz-1, ix]
    _push(slice(1, nz), slice(1, nx), -1, 0,
          a * idh2 * Qy[:-1, 1:] * Q[1:, 1:]
          + ((1-a)*idh2/4) * (2 * Qy[:-1, 1:] * Q[1:, 1:]
                               - Px[1:, 1:] * P[1:, 1:]
                               - Px[1:, :-1] * P[1:, 1:])
          + d * MU0 * omega2 * eps_c[:-1, 1:])

    # ---- NE: iz>0, ix<nx-1  |  neighbour (iz-1, ix+1) ----
    # stag2: (1-a)*idh2/4 * (Px[iz,ix]*P[iz,ix] + Qy[iz-1,ix]*Q[iz,ix])
    #        + e*MU0*omega2*eps_c[iz-1, ix+1]
    _push(slice(1, nz), slice(0, nx-1), -1, +1,
          ((1-a)*idh2/4) * (Px[1:, :-1] * P[1:, :-1] + Qy[:-1, :-1] * Q[1:, :-1])
          + e * MU0 * omega2 * eps_c[:-1, 1:])

    # ---- W: iz>0, ix>0 (stag2: both conditions)  |  neighbour (iz, ix-1) ----
    # stag2: a*idh2*Px[iz,ix-1]*P[iz,ix]
    #        + (1-a)*idh2/4*(2*Px[iz,ix-1]*P - Qy[iz,ix]*Q - Qy[iz-1,ix]*Q)
    #        + d*MU0*omega2*eps_c[iz, ix-1]
    _push(slice(1, nz), slice(1, nx), 0, -1,
          a * idh2 * Px[1:, :-1] * P[1:, 1:]
          + ((1-a)*idh2/4) * (2 * Px[1:, :-1] * P[1:, 1:]
                               - Qy[1:, 1:] * Q[1:, 1:]
                               - Qy[:-1, 1:] * Q[1:, 1:])
          + d * MU0 * omega2 * eps_c[1:, :-1])

    # ---- Center: iz>0, ix>0, iz<nz-1, ix<nx-1 ----
    ir, ic = slice(1, nz-1), slice(1, nx-1)
    axis_sum = (
        Px[1:nz-1, 0:nx-2] * P[ir, ic]   # Px[iz, ix-1]
        + Px[ir, ic]        * P[ir, ic]   # Px[iz, ix]
        + Qy[ir, ic]        * Q[ir, ic]   # Qy[iz, ix]
        + Qy[0:nz-2, ic]    * Q[ir, ic]   # Qy[iz-1, ix]
    )
    _push(ir, ic, 0, 0,
          c * MU0 * omega2 * eps_c[ir, ic]
          - a * idh2 * axis_sum
          - ((1-a)*idh2/2) * axis_sum)    # stag2 uses (1-a)/2 * axis (not diag)

    # ---- E: iz>0, ix<nx-1 (stag2: iz>0 and ix<nx-1)  |  neighbour (iz, ix+1) ----
    # stag2: a*idh2*Px[iz,ix]*P + (1-a)*idh2/4*(2*Px*P - Qy[iz,ix]*Q - Qy[iz-1,ix]*Q)
    #        + d*MU0*omega2*eps_c[iz, ix+1]
    _push(slice(1, nz), slice(0, nx-1), 0, +1,
          a * idh2 * Px[1:, :-1] * P[1:, :-1]
          + ((1-a)*idh2/4) * (2 * Px[1:, :-1] * P[1:, :-1]
                               - Qy[1:, :-1] * Q[1:, :-1]
                               - Qy[:-1, :-1] * Q[1:, :-1])
          + d * MU0 * omega2 * eps_c[1:, 1:])

    # ---- SW: iz<nz-1, ix>0  |  neighbour (iz+1, ix-1) ----
    # stag2: (1-a)*idh2/4 * (Qy[iz,ix]*Q + Px[iz,ix-1]*P) + e*MU0*omega2*eps_c[iz+1, ix-1]
    _push(slice(0, nz-1), slice(1, nx), +1, -1,
          ((1-a)*idh2/4) * (Qy[:-1, 1:] * Q[:-1, 1:] + Px[:-1, :-1] * P[:-1, 1:])
          + e * MU0 * omega2 * eps_c[1:, :-1])

    # ---- S: iz<nz-1, ix>0 (stag2: both conditions)  |  neighbour (iz+1, ix) ----
    # stag2: a*idh2*Qy[iz,ix]*Q
    #        + (1-a)*idh2/4*(2*Qy*Q - Px[iz,ix]*P - Px[iz,ix-1]*P)
    #        + d*MU0*omega2*eps_c[iz+1, ix]
    _push(slice(0, nz-1), slice(1, nx), +1, 0,
          a * idh2 * Qy[:-1, 1:] * Q[:-1, 1:]
          + ((1-a)*idh2/4) * (2 * Qy[:-1, 1:] * Q[:-1, 1:]
                               - Px[:-1, 1:] * P[:-1, 1:]
                               - Px[:-1, :-1] * P[:-1, 1:])
          + d * MU0 * omega2 * eps_c[1:, 1:])

    # ---- SE: iz<nz-1, ix<nx-1  |  neighbour (iz+1, ix+1) ----
    # stag2: (1-a)*idh2/4 * (Px[iz,ix]*P + Qy[iz,ix]*Q) + e*MU0*omega2*eps_c[iz+1, ix+1]
    _push(slice(0, nz-1), slice(0, nx-1), +1, +1,
          ((1-a)*idh2/4) * (Px[:-1, :-1] * P[:-1, :-1] + Qy[:-1, :-1] * Q[:-1, :-1])
          + e * MU0 * omega2 * eps_c[1:, 1:])

    row_arr = np.concatenate(rows_l)
    col_arr = np.concatenate(cols_l)
    val_arr = np.concatenate(vals_l)
    return sparse.csr_matrix((val_arr, (row_arr, col_arr)), shape=(N, N))


# ---------------------------------------------------------------------------
# Matrix assembly dispatcher
# ---------------------------------------------------------------------------

def build_helmholtz_2d(
    epsr: np.ndarray,
    sigma: np.ndarray,
    dh: float,
    omega: float,
    npml: int,
    a0_cfs: float = 100.0,
    free_surface: bool = False,
    grid_style: str = "stag1",
    use_gpu: bool = False,
) -> Any:
    """
    Assemble the 2D Helmholtz (impedance) matrix for TM-mode Ey.

    Dispatches to the CFS-PML 9-point stencil implementation selected by
    *grid_style* ("stag1" or "stag2").  Assumes a uniform grid spacing *dh*
    (dx = dz = dh) as required by the 9-point staggered-grid formulations.

    Parameters
    ----------
    epsr : ndarray, shape (nz, nx)
        Relative permittivity grid.
    sigma : ndarray, shape (nz, nx)
        Electrical conductivity grid [S/m].
    dh : float
        Uniform grid spacing [m].
    omega : float
        Angular frequency [rad/s].
    npml : int
        PML thickness in grid cells.
    a0_cfs : float
        CFS-PML maximum conductivity (sigma_max).  Default 100.0.
    free_surface : bool
        If True suppress the top-side PML (free surface BC).
    grid_style : str
        Discretisation variant: "stag1" (Hustedt 2004) or "stag2"
        (Layek & Sengupta 2023).
    use_gpu : bool
        Reserved for future CuPy GPU acceleration.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (nx*nz, nx*nz)
        Assembled sparse Helmholtz system matrix.
    """
    nz, nx = epsr.shape

    pml = _build_cfspml_coeffs(nx, nz, npml, dh, omega, a0_cfs,
                                free_surface=free_surface)

    if grid_style == "stag2":
        return _build_helmholtz_stag2(epsr, sigma, dh, omega, pml)
    return _build_helmholtz_stag1(epsr, sigma, dh, omega, pml)


# ---------------------------------------------------------------------------
# Forward solve
# ---------------------------------------------------------------------------

def solve_forward(
    A: Any,
    source_ix: int,
    source_iz: int,
    nx: int,
    nz: int,
    use_gpu: bool = False,
    source_amplitude: complex = 1.0,
) -> np.ndarray:
    """
    Solve the system A @ Ey = b for a single point source.

    Parameters
    ----------
    A : sparse matrix, shape (nx*nz, nx*nz)
        Helmholtz system matrix from :func:`build_helmholtz_2d`.
    source_ix, source_iz : int
        Grid-index position of the point source.
    nx, nz : int
        Grid dimensions.
    use_gpu : bool
        Reserved for future GPU acceleration.
    source_amplitude : complex
        Amplitude of the point source.

    Returns
    -------
    Ez : ndarray, shape (nz, nx)
        Complex electric-field solution.
    """
    N = nx * nz
    b = np.zeros(N, dtype=np.complex128)
    b[_idx(source_ix, source_iz, nx)] = source_amplitude

    A_cpu = A if sparse.issparse(A) else sparse.csr_matrix(A)
    x = sp_linalg.spsolve(A_cpu, b)
    return x.reshape(nz, nx)


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

def run_forward(
    config: dict[str, Any],
    epsr: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    sources: list[tuple[int, int]] | None = None,
    receivers_list: list[list[tuple[int, int]]] | None = None,
    use_gpu: bool = False,
    n_workers: int = 1,
    grid_style: str = "stag1",
    save_impedance_path: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Run FDFD forward modelling for one or more sources.

    The Helmholtz matrix is assembled once and shared across all source solves.
    Multi-source parallelism is controlled by *n_workers*.

    Parameters
    ----------
    config : dict
        Forward configuration (domain, pml, freq_hz, source/sources,
        receivers, model, output).
    epsr : ndarray or None
        Relative permittivity (nz, nx).  Built from *config* if ``None``.
    sigma : ndarray or None
        Conductivity in S/m (nz, nx).  Built from *config* if ``None``.
    sources : list of (ix, iz) or None
        Source positions.  Read from *config* if ``None``.
    receivers_list : list of lists or None
        Receiver lists per source.  Read from *config* if ``None``.
    use_gpu : bool
        Use GPU acceleration if CuPy is available (future feature).
    n_workers : int
        Number of parallel workers for multi-source solves (``--ncpus``).
    grid_style : str
        Discretisation variant passed to :func:`build_helmholtz_2d`
        ("stag1" or "stag2").
    save_impedance_path : str or Path or None
        If provided, the assembled Helmholtz matrix is saved to this path
        as a SciPy sparse NPZ file (``--impedance-matrix``).

    Returns
    -------
    traces : ndarray, shape (n_sources, n_receivers), complex
        Recorded field values at receiver positions.
    info : dict
        Metadata: nx, nz, dh, freq_hz, receivers, and optionally 'fields'.
    """
    from create_models.build_models import build_model_from_config

    domain = config.get("domain", config)
    nx = int(domain.get("nx", 101))
    nz = int(domain.get("nz", 81))
    dh = float(domain.get("dx", domain.get("dh", 0.1)))
    pml_cfg = config.get("pml", {})
    npml = int(pml_cfg.get("npx", pml_cfg.get("npml", 10)))
    a0_cfs = float(pml_cfg.get("a0_cfs", 100.0))
    free_surface = bool(pml_cfg.get("free_surface", False))
    grid_style = config.get("grid_style", grid_style)
    freq = float(config.get("freq_hz", 900e6))
    omega = 2.0 * np.pi * freq

    if epsr is None or sigma is None:
        epsr, sigma = build_model_from_config(config, nx, nz)

    if sources is None:
        src = config.get("source", config.get("sources", [{"ix": 50, "iz": 2}]))
        if isinstance(src, list):
            sources = [(int(s["ix"]), int(s["iz"])) for s in src]
        else:
            sources = [(int(src["ix"]), int(src["iz"]))]

    if receivers_list is None:
        rcfg = config.get("receivers", {})
        if isinstance(rcfg, dict) and rcfg.get("mode") == "line":
            iz_r = int(rcfg.get("iz", 2))
            i_start = int(rcfg.get("ix_start", 0))
            i_end = int(rcfg.get("ix_end", nx - 1))
            recs = [(ix, iz_r) for ix in range(i_start, i_end + 1)]
        else:
            recs = [(50, 2)]
        receivers_list = [recs] * len(sources)

    # Assemble impedance matrix (built once, reused for all sources)
    A = build_helmholtz_2d(epsr, sigma, dh, omega, npml,
                           a0_cfs=a0_cfs, free_surface=free_surface,
                           grid_style=grid_style, use_gpu=use_gpu)

    if save_impedance_path is not None:
        _imp_path = Path(save_impedance_path)
        _imp_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(str(_imp_path), A)
        print(f"  Impedance matrix saved -> {_imp_path}")

    save_fields = config.get("output", {}).get("save_fields", False)

    # Source amplitude matches MATLAB RHS_TE1.m: amp = -(omega * mu0 * j) / dh^2
    src_amp = -(omega * MU0 * 1j) / dh ** 2

    def _solve_source(src: tuple[int, int]) -> np.ndarray:
        sx, sz = src
        return solve_forward(A, sx, sz, nx, nz, use_gpu=use_gpu, source_amplitude=src_amp)

    if n_workers > 1 and len(sources) > 1:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            fields_all = list(ex.map(_solve_source, sources))
    else:
        fields_all = [_solve_source(src) for src in sources]

    out_traces: list[np.ndarray] = []
    fields_list: list[np.ndarray] = []
    for Ez, recs in zip(fields_all, receivers_list):
        trace = np.array([Ez[riz, rix] for rix, riz in recs], dtype=np.complex128)
        out_traces.append(trace)
        if save_fields:
            fields_list.append(Ez.copy())

    traces = np.array(out_traces)
    info: dict[str, Any] = {
        "nx": nx, "nz": nz, "dh": dh,
        "freq_hz": freq, "receivers": receivers_list[0],
    }
    if save_fields:
        info["fields"] = fields_list
    return traces, info


def run_forward_single_source(
    config: dict[str, Any],
    epsr: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    use_gpu: bool = False,
    grid_style: str = "stag1",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Convenience wrapper: run forward for one source and return (trace_1d, field_2d, info).
    """
    cfg = dict(config)
    cfg.setdefault("output", {})["save_fields"] = True
    traces, info = run_forward(cfg, epsr=epsr, sigma=sigma,
                               use_gpu=use_gpu, grid_style=grid_style)
    field = info.get("fields", [None])[0]
    return traces[0], field, info
