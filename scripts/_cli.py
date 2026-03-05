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
    --stag2                Use stag2 (Layek & Sengupta 2023) 9-point CFS-PML stencil.
    -v / --verbose         Verbose console output.
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
        help="Use stag2 (Layek & Sengupta 2023) 9-point CFS-PML stencil.",
    )
    parser.set_defaults(grid_style="stag1")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Print extra diagnostic information during the run.",
    )
    return parser
