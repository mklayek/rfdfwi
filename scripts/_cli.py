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
Shared argparse helpers for rfdfwi example scripts.
"""
from __future__ import annotations

import argparse


def add_common_args(
    parser: argparse.ArgumentParser,
    default_kind: str = "forward",
) -> argparse.ArgumentParser:
    """
    Add arguments common to all forward / inversion example scripts.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend in-place.
    default_kind : str
        Logical category used in help text (e.g. "forward_bscan", "forward_wavefield",
        "forward_cmp", "inversion").

    Added arguments
    ---------------
    --config FILE          Path to YAML configuration file.
    --results-dir DIR      Override the default output directory.
    --impedance-matrix     Save the assembled Helmholtz matrix.
    --ncpus N              Parallel CPU workers.
    --use-gpu              Enable GPU acceleration (requires CuPy).
    --stag1                Use stag1 (Hustedt 2004) 9-point CFS-PML stencil (default).
    --stag2                Use stag2 (Layek & Sengupta 2024) 9-point CFS-PML stencil.
    -v / --verbose         Verbose console output.
    --timestamps           Prepend [YYYY-MM-DD HH:MM:SS] to every console line.
    --patience N           Early-stop FWI after N non-decreasing misfit iterations.
    """
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=f"Override the default output directory for {default_kind} results.",
    )
    parser.add_argument(
        "--impedance-matrix",
        action="store_true",
        default=False,
        help=(
            "Assemble and save the Helmholtz (impedance) matrix to the results "
            "directory as impedance_matrix.npz (SciPy sparse NPZ format)."
        ),
    )
    parser.add_argument(
        "--ncpus",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel CPU workers for multi-source forward solves.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=False,
        help="Use GPU acceleration via CuPy if available (experimental).",
    )
    _grid = parser.add_mutually_exclusive_group()
    _grid.add_argument(
        "--stag1",
        dest="grid_style",
        action="store_const",
        const="stag1",
        help="Use stag1 (Hustedt et al. 2004) 9-point CFS-PML stencil (default).",
    )
    _grid.add_argument(
        "--stag2",
        dest="grid_style",
        action="store_const",
        const="stag2",
        help="Use stag2 (Layek & Sengupta 2024) 9-point CFS-PML stencil.",
    )
    parser.set_defaults(grid_style="stag1")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print extra diagnostic information during the run.",
    )
    parser.add_argument(
        "--timestamps",
        action="store_true",
        default=False,
        help="Prepend [YYYY-MM-DD HH:MM:SS] timestamp to every console output line.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        metavar="N",
        help="Early-stop if L2 misfit does not decrease for N consecutive iterations "
             "(default: read from config inversion.patience, fallback=5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        metavar="N",
        help="Number of initial iterations to skip before applying early-stop check "
             "(default: read from config inversion.warmup_iters, fallback=3).",
    )
    parser.add_argument(
        "--step-epsr", type=float, default=None, metavar="S",
        help="Initial line-search step for εᵣ update (overrides config step_init_epsr).",
    )
    parser.add_argument(
        "--step-sigma", type=float, default=None, metavar="S",
        help="Initial line-search step for σ update (overrides config step_init_sigma).",
    )
    parser.add_argument(
        "--c2-wolfe", type=float, default=None, metavar="V",
        help="Wolfe C2 curvature constant (default: 0.9, MATLAB: 0.9).",
    )
    parser.add_argument(
        "--nlbfgs", type=int, default=None, metavar="N",
        help="L-BFGS memory length (default: 5, MATLAB: 5).",
    )
    parser.add_argument(
        "--no-lbfgs", action="store_true", default=False,
        help="Disable L-BFGS, use steepest descent with Hessian preconditioning.",
    )
    return parser
