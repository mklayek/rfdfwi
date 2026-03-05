"""
CMP (Common Midpoint) processing and visualization — MATLAB GPRFM style.

FD→TD algorithm matches GPRFM.m exactly:

    precN_pf  = [conj(E[nw-1:0:-1]),  E[0:nw]]      ← Hermitian extension
    precN_pad = [zeros(pad), precN_pf, zeros(pad)]   ← zero-pad both sides
    FDFD_pt   = real(ifft(ifftshift(precN_pad)))      ← MATLAB-style IFFT

Post-processing matches GPRFM.m:

    outData = interp2(X, Y, data, X2, Y2, 'spline')  ← 10× spline upsample
    outData = -agc2(outData, 60, 10)                  ← AGC + polarity flip
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

CMP_DPI = 600

# GPRFM.m default discrete frequencies (Hz)
# f = [50e6 60e6 70e6 80e6 90e6 100e6 125e6 150e6 175e6 200e6]
GPRFM_FREQS_HZ: list[float] = [
    50e6, 60e6, 70e6, 80e6, 90e6, 100e6, 125e6, 150e6, 175e6, 200e6,
]


# ---------------------------------------------------------------------------
# Source wavelet spectrum (Blackman-Harris)
# ---------------------------------------------------------------------------

def blackman_harris_spectrum(freqs: np.ndarray) -> np.ndarray:
    """
    4-term Blackman-Harris window evaluated at the actual frequency positions.

    Use as a multiplicative spectral taper applied to frequency-domain data
    **before** the Hermitian IFFT.  This compensates for the flat-spectrum
    (unit-amplitude) point source used by the Python FDFD solver and gives
    a clean, bandlimited Blackman-Harris wavelet in time — matching MATLAB's
    RHS_TE1 which uses a frequency-dependent source amplitude.

    Parameters
    ----------
    freqs : (nf,)  Frequency array [Hz] (uniform or non-uniform).

    Returns
    -------
    window : (nf,)  Window values in [0, 1], symmetric with peak at centre.
    """
    nf = len(freqs)
    if nf == 1:
        return np.ones(1)
    # Normalise frequency positions to [0, 1]
    f_norm = (freqs - freqs[0]) / (freqs[-1] - freqs[0])
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    return (a0
            - a1 * np.cos(2 * np.pi * f_norm)
            + a2 * np.cos(4 * np.pi * f_norm)
            - a3 * np.cos(6 * np.pi * f_norm))


# ---------------------------------------------------------------------------
# GPRFM processing utilities
# ---------------------------------------------------------------------------

def agc2(data: np.ndarray, window: int, overlap: int = 0) -> np.ndarray:
    """
    Sliding-window RMS Automatic Gain Control (matches MATLAB agc2).

    For each time sample i in each trace, the output is
        out[i, j] = data[i, j] / rms(data[i0:i1, j])
    where [i0, i1] is a centred window of length `window`.

    Called in GPRFM.m as:  outData = -agc2(outData, 60, 10)
    The `-` negates (polarity flip) and is applied in the caller.

    Parameters
    ----------
    data    : (nt, nx)  Real 2-D array, rows = time, cols = traces.
    window  : int       Window length [samples].
    overlap : int       Kept for API compatibility with MATLAB agc2; unused.

    Returns
    -------
    out : (nt, nx)  AGC-normalised data.
    """
    nt, nx    = data.shape
    out       = np.zeros_like(data, dtype=float)
    half_w    = window // 2

    for i in range(nt):
        i0  = max(0, i - half_w)
        i1  = min(nt, i + half_w + 1)
        rms = np.sqrt(np.mean(data[i0:i1, :] ** 2, axis=0))
        rms = np.where(rms > 0.0, rms, 1.0)
        out[i, :] = data[i, :] / rms

    return out


def spline_upsample(data: np.ndarray, factor: float = 10.0) -> np.ndarray:
    """
    Spline (3rd-order) 2-D upsample, matching MATLAB:
        [X2,Y2] = meshgrid(1:0.1:size(data,2), 1:0.1:size(data,1))
        outData = interp2(X, Y, data, X2, Y2, 'spline')

    Both the time axis (rows) and the offset axis (columns) are upsampled
    by the same factor.

    Parameters
    ----------
    data   : (nt, nx)  Real 2-D array.
    factor : float     Upsampling factor (default 10, matching GPRFM).

    Returns
    -------
    Upsampled array of shape ~ (nt*factor, nx*factor).
    """
    return zoom(data.astype(float), (factor, factor), order=3, mode="mirror")


# ---------------------------------------------------------------------------
# Frequency-domain → time-domain (GPRFM Hermitian IFFT)
# ---------------------------------------------------------------------------

def freq_to_timedomain(
    freq_data: np.ndarray,
    freqs: np.ndarray,
    pad: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert frequency-domain CMP gather to time domain.

    Matches GPRFM.m exactly:

        precN_pf  = cat(2, conj(precN(:,end:-1:2,:)), precN)
        precN_pad = cat(2, zeros(ntr,pad,nshots), precN_pf, zeros(ntr,pad,nshots))
        FDFD_pf1  = ifftshift(precN_pad, 2)
        FDFD_pt1  = real(ifft(FDFD_pf1, [], 2))

    The Hermitian extension guarantees real output.
    ifftshift places the lowest positive frequency at bin 0 so that
    MATLAB's IFFT convention gives correct travel-time delays.

    Time axis:  dt = 1 / (ns * df_min)
    where df_min = min(diff(freqs)) is the minimum frequency step in the
    array, and ns = (nf + pad)*2 - 1 is the IFFT length.

    Parameters
    ----------
    freq_data : (nf, n_offsets)  Complex receiver responses.
                                 freq_data[k, col] = E at freqs[k].
    freqs     : (nf,)            Actual frequency values [Hz]
                                 (may be non-uniform, as in GPRFM).
    pad       : int              Zero-samples added on each side of the
                                 Hermitian spectrum (extra time range).

    Returns
    -------
    cmp_td : (ns, n_offsets)  Real time-domain CMP matrix.
    t_ns   : (ns,)            Time axis [ns].
    """
    nf        = freq_data.shape[0]
    n_offsets = freq_data.shape[1]

    # Effective frequency spacing (minimum diff for non-uniform arrays)
    if nf > 1:
        df_min = float(np.min(np.diff(freqs)))
    else:
        df_min = float(freqs[0])

    # --- Step 1: Hermitian extension ---
    # MATLAB: cat(2, conj(precN(:,end:-1:2,:)), precN)
    # 0-indexed Python: prepend conj(E[nf-1:0:-1])
    E_neg    = np.conj(freq_data[nf - 1:0:-1, :])          # (nf-1, n_offsets)
    precN_pf = np.concatenate([E_neg, freq_data], axis=0)   # (2*nf-1, n_offsets)

    # --- Step 2: Zero-padding on both sides ---
    # MATLAB: cat(2, zeros(ntr,pad,...), precN_pf, zeros(ntr,pad,...))
    if pad > 0:
        z            = np.zeros((pad, n_offsets), dtype=complex)
        precN_padded = np.concatenate([z, precN_pf, z], axis=0)
    else:
        precN_padded = precN_pf

    ns = precN_padded.shape[0]   # = (nf + pad) * 2 - 1

    # --- Step 3: ifftshift + IFFT + real ---
    # MATLAB: real(ifft(ifftshift(precN_pad, 2), [], 2))
    cmp_td = np.zeros((ns, n_offsets), dtype=float)
    for col in range(n_offsets):
        spec           = np.fft.ifftshift(precN_padded[:, col])
        cmp_td[:, col] = np.real(np.fft.ifft(spec))

    # Time axis: dt = 1 / (ns * df_min)
    dt_s = 1.0 / (ns * df_min)
    t_ns = np.arange(ns) * dt_s * 1e9   # [ns]

    return cmp_td, t_ns


# ---------------------------------------------------------------------------
# Colour CMP plot  (GPRFM style)
# ---------------------------------------------------------------------------

def plot_cmp_color(
    cmp: np.ndarray,
    half_offsets: np.ndarray,
    t_ns: np.ndarray,
    grid_style: str,
    freqs_hz: np.ndarray,
    save_path: Path,
    agc_window: int  = 60,
    upsample: int    = 10,
    clip_pct: float  = 99.0,
    dpi: int         = CMP_DPI,
) -> None:
    """
    Colour (image) CMP gather — full GPRFM post-processing pipeline:

        1. 10× spline upsample  (GPRFM: interp2 'spline')
        2. AGC2 + polarity flip (GPRFM: outData = -agc2(outData, 60, 10))
        3. imagesc with seismic colormap, percentile colour limits

    Parameters
    ----------
    cmp         : (nt, n_offsets)  Real time-domain CMP from freq_to_timedomain.
    half_offsets: (n_offsets,)     Half-offset values [m].
    t_ns        : (nt,)            Time axis [ns].
    grid_style  : str              Stencil label for title.
    freqs_hz    : (nf,)            Actual frequencies used [Hz].
    save_path   : Path             Output file path.
    agc_window  : int              AGC window [samples after upsample]; default 60.
    upsample    : int              Spline upsample factor; default 10.
    clip_pct    : float            Percentile for symmetric colour limits.
    """
    # --- Spline upsample (GPRFM: interp2 'spline') ---
    if upsample > 1:
        data_up = spline_upsample(cmp, factor=upsample)
    else:
        data_up = cmp.copy().astype(float)

    nt_up, n_off_up = data_up.shape

    # Build upsampled axes for imshow extent
    t_up  = np.linspace(t_ns[0],          t_ns[-1],           nt_up)
    x_up  = np.linspace(half_offsets[0],  half_offsets[-1],   n_off_up)

    # --- AGC2 + polarity flip (GPRFM: outData = -agc2(outData, 60, 10)) ---
    data_agc = -agc2(data_up, window=agc_window)

    # Percentile colour limits
    vmax = float(np.percentile(np.abs(data_agc), clip_pct))
    vmax = vmax if vmax > 0 else 1.0

    extent = [x_up[0], x_up[-1], t_up[-1], t_up[0]]

    fc_lo = freqs_hz[0]  / 1e6
    fc_hi = freqs_hz[-1] / 1e6

    fig, ax = plt.subplots(figsize=(8, 12), dpi=dpi)

    im = ax.imshow(
        data_agc,
        extent=extent,
        aspect="auto",
        cmap="seismic",
        vmin=-vmax, vmax=vmax,
        interpolation="bilinear",   # already upsampled; bilinear avoids double-blur
    )

    ax.set_xlabel("Half-offset [m]",         fontsize=14)
    ax.set_ylabel("Two-way travel time [ns]", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_title(
        f"CMP gather  {grid_style}  ({fc_lo:.0f}–{fc_hi:.0f} MHz)",
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
# Wiggle CMP plot  (GPRFM style)
# ---------------------------------------------------------------------------

def plot_cmp_wiggle(
    cmp: np.ndarray,
    half_offsets: np.ndarray,
    t_ns: np.ndarray,
    grid_style: str,
    freqs_hz: np.ndarray,
    save_path: Path,
    agc_window: int  = 60,
    upsample: int    = 10,
    gain: float      = 1.5,
    dpi: int         = CMP_DPI,
) -> None:
    """
    Wiggle-trace CMP — same GPRFM post-processing as colour plot
    (spline upsample → AGC2 + polarity flip), then classic GPR/seismic
    wiggle display with positive-amplitude fill.

    Parameters
    ----------
    cmp         : (nt, n_offsets)  Real time-domain CMP.
    half_offsets: (n_offsets,)     Half-offset values [m].
    t_ns        : (nt,)            Time axis [ns].
    gain        : float            Wiggle scale relative to offset spacing.
    """
    # Spline upsample
    if upsample > 1:
        data_up = spline_upsample(cmp, factor=upsample)
    else:
        data_up = cmp.copy().astype(float)

    nt_up, n_off_up = data_up.shape
    t_up = np.linspace(t_ns[0], t_ns[-1], nt_up)

    # AGC2 + polarity flip (matches GPRFM)
    data_agc = -agc2(data_up, window=agc_window)

    # Upsample the offset axis for wiggle positions
    x_up = np.linspace(half_offsets[0], half_offsets[-1], n_off_up)
    dx   = float(x_up[1] - x_up[0]) if n_off_up > 1 else float(half_offsets[1] - half_offsets[0])
    wiggle_scale = gain * dx

    fig, ax = plt.subplots(figsize=(10, 14), dpi=dpi)

    for col in range(n_off_up):
        x0     = x_up[col]
        trace  = data_agc[:, col]
        wiggle = x0 + wiggle_scale * trace

        ax.plot(wiggle, t_up, color="black", linewidth=0.3, alpha=0.8)
        ax.fill_betweenx(t_up, x0, wiggle,
                         where=(trace >= 0),
                         color="black", alpha=0.7, linewidth=0)

    ax.set_xlim(x_up[0] - dx, x_up[-1] + dx)
    ax.set_ylim(t_up[-1], t_up[0])   # time increases downward
    ax.set_xlabel("Half-offset [m]",         fontsize=14)
    ax.set_ylabel("Two-way travel time [ns]", fontsize=14)
    ax.tick_params(labelsize=12)
    fc_lo = freqs_hz[0]  / 1e6
    fc_hi = freqs_hz[-1] / 1e6
    ax.set_title(
        f"CMP wiggle  {grid_style}  ({fc_lo:.0f}–{fc_hi:.0f} MHz)",
        fontsize=14, pad=6,
    )
    ax.grid(True, linewidth=0.3, color="gray", alpha=0.4)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
