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
Time-domain shot gather generation with center-shot spatial reorganization.

This module implements the MATLAB GPRFM center-shot gather workflow:
1. Convert FD shot gather to TD via Hermitian IFFT
2. Reorganize receiver order for center-shot display (tmp1-tmp4 rearrangement)
3. Generate symmetric offset array

Theory
------
The MATLAB GPRFM code reorganizes receivers into 4 groups for a center-shot display:
  - tmp1 = receivers [1:29]       (29 receivers, center)
  - tmp2 = receivers [30:57]      (28 receivers, left)    → flip
  - tmp3 = receivers [58:87]      (30 receivers, left)    → flip
  - tmp4 = receivers [88:117]     (30 receivers, right)

After flipping tmp2 and tmp3 and concatenating as [tmp2_flip, tmp3_flip, tmp1, tmp4],
the result is a symmetric center-shot gather where:
  - Left half: flipped receivers (offset < 0)
  - Center: tmp1 (offset near 0)
  - Right half: tmp4 (offset > 0)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple


def reorganize_center_shot(
    sg_matrix: np.ndarray,
    src_ix: int,
    rec_indices: np.ndarray,
    dh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorganize shot gather for center-shot display (MATLAB tmp1-tmp4 style).

    For MATLAB 82-source / 162-receiver geometry (ntr=117), implements exact 4-group
    split/flip logic. For other receiver counts, generalizes the concept by:
      - Finding center position (closest to zero offset)
      - Splitting into left (negative offset) and right (positive offset)
      - Flipping left side horizontally
      - Concatenating as [left_flipped, center, right]

    Parameters
    ----------
    sg_matrix : np.ndarray
        Shape (nt, ntr) — raw time-domain shot gather
    src_ix : int
        Source x-index (for offset calculation)
    rec_indices : np.ndarray
        Shape (ntr,) — receiver x-indices
    dh : float
        Grid spacing [m]

    Returns
    -------
    sg_reorg : np.ndarray
        Shape (nt, ntr_reorg) — spatially reorganized shot gather
    offsets_reorg : np.ndarray
        Shape (ntr_reorg,) — reorganized offset array [m], symmetric around zero
    """
    ntr = len(rec_indices)
    nt = sg_matrix.shape[0]

    # Compute receiver offsets (signed distance from source)
    offsets_orig = (rec_indices - src_ix).astype(float) * dh

    # Special case: MATLAB geometry (ntr=117)
    if ntr == 117:
        # MATLAB group splits: [29, 28, 30, 30]
        # Groups in original indexing: [0:29], [29:57], [57:87], [87:117]
        g1_start, g1_end = 0, 29      # tmp1: 29 receivers
        g2_start, g2_end = 29, 57     # tmp2: 28 receivers (to flip)
        g3_start, g3_end = 57, 87     # tmp3: 30 receivers (to flip)
        g4_start, g4_end = 87, 117    # tmp4: 30 receivers

        # Extract groups from raw shot gather
        tmp1 = sg_matrix[:, g1_start:g1_end]         # (nt, 29)
        tmp2 = sg_matrix[:, g2_start:g2_end]         # (nt, 28)
        tmp3 = sg_matrix[:, g3_start:g3_end]         # (nt, 30)
        tmp4 = sg_matrix[:, g4_start:g4_end]         # (nt, 30)

        # Flip tmp2 and tmp3 horizontally (reverse along receiver axis = axis 1)
        tmp2_flip = np.fliplr(tmp2)  # (nt, 28)
        tmp3_flip = np.fliplr(tmp3)  # (nt, 30)

        # Concatenate in MATLAB order: [tmp2_flip, tmp3_flip, tmp1, tmp4]
        sg_reorg = np.concatenate([tmp2_flip, tmp3_flip, tmp1, tmp4], axis=1)  # (nt, 117)

        # Extract offsets for each group and apply same flip logic
        off1 = offsets_orig[g1_start:g1_end]  # (29,)
        off2 = offsets_orig[g2_start:g2_end]  # (28,)
        off3 = offsets_orig[g3_start:g3_end]  # (30,)
        off4 = offsets_orig[g4_start:g4_end]  # (30,)

        # Flip offset arrays (reverse order)
        off2_flip = off2[::-1]  # (28,)
        off3_flip = off3[::-1]  # (30,)

        # Concatenate offsets in same order
        offsets_reorg = np.concatenate([off2_flip, off3_flip, off1, off4])  # (117,)
    else:
        # Adaptive reorganization for other receiver counts:
        # Split into left (negative offset) and right (positive offset)
        # Flip left, then concatenate as [left_flip, right]

        # Find split point: closest receiver to zero offset
        split_idx = np.argmin(np.abs(offsets_orig))

        # Left side (negative or near-zero offsets): to be flipped
        left_data = sg_matrix[:, :split_idx]  # (nt, split_idx)
        left_offsets = offsets_orig[:split_idx]

        # Center and right (positive or near-zero offsets): keep as is
        center_right_data = sg_matrix[:, split_idx:]  # (nt, ntr-split_idx)
        center_right_offsets = offsets_orig[split_idx:]

        # Flip left side horizontally
        left_flip = np.fliplr(left_data)  # (nt, split_idx)
        left_flip_offsets = left_offsets[::-1]

        # Concatenate: [left_flip, center_right]
        sg_reorg = np.concatenate([left_flip, center_right_data], axis=1)
        offsets_reorg = np.concatenate([left_flip_offsets, center_right_offsets])

    return sg_reorg, offsets_reorg


def verify_matlab_geometry(ntr: int, nf: int) -> bool:
    """
    Verify that the data dimensions match MATLAB 82-source / 162-receiver geometry.

    Parameters
    ----------
    ntr : int
        Number of traces (receivers per shot)
    nf : int
        Number of frequencies

    Returns
    -------
    is_valid : bool
        True if dimensions are MATLAB-compatible
    """
    # MATLAB uses 81-receiver boundary (4-sided acquisition, 162 total)
    # Single-source shot gather typically has ~117 receivers (half boundary)
    valid_ntr = ntr == 117
    valid_nf = nf > 0

    return valid_ntr and valid_nf
