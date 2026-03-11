# Shot Gather — GERMAINE FD2TD Style

```
Project Type:          Scientific / Numerical Algorithm
Domain:                GPR (Ground Penetrating Radar) Forward Modelling
Primary Language:      Python
Scale Estimate:        Small (80 FDFD solves, ~120×80 grid, <5 min)
Deployment Target:     Local CPU (Windows, conda rfdfwimkl)
Reproducibility Level: High (deterministic FDFD, fixed model from YAML)
```

## Problem Description

The current shot gather script produces **V-shaped periodic stripes** instead of
physically meaningful events (direct wave, reflection hyperbolas, dipping events
from a clay lens). The root cause: the `freq_to_timedomain()` function from
`plot_cmp.py` uses a Hermitian IFFT with `conj()` + `ifftshift` that produces
wrap-around artifacts for this frequency layout (linspace 1–200 MHz, no DC).

The reference image (KakaoTalk_20260310_170239986.png) shows the correct result:
nearly-flat air wave at 5–10 ns, gentle reflection hyperbolas at 10–20 ns,
dipping clay lens events at 25–40 ns, and natural amplitude decay.

## Objective

Rewrite `examples/run_forward_shotgather.py` from scratch following the
**GERMAINE FD2TD.ipynb** algorithm (Daniel Koehn, 2017). The script must:
1. Solve FDFD at nf linspace frequencies (real omega, standard source)
2. Apply Tukey(1)/Hann taper to frequency-domain data
3. Build 4×nf Hermitian spectrum: `[E, zeros, zeros, flipud(E)]`
4. `np.fft.ifft` + `np.real()` — **NO ifftshift, NO conj()**
5. Extract time window, normalize by global max, display with fixed clip
6. Produce a shot gather matching the reference image

## Proposed Solution

Follow the GERMAINE FD2TD notebook verbatim. Key differences from previous code:

| Aspect | OLD (freq_to_timedomain) | NEW (GERMAINE) |
|--------|--------------------------|----------------|
| Negative freqs | `conj(flipud(E))` | `flipud(E)` (no conj) |
| Zero-padding | pad on both sides | 2×nf zeros in middle |
| Total IFFT length | `(nf+pad)*2-1` | `4*nf` |
| Phase centering | `ifftshift` before IFFT | **None** |
| Spectral taper | Blackman-Harris or Ricker | **Tukey(1) = Hann** |
| Normalization | clip_frac × max | **global max norm + fixed clip** |

## Mathematical / Algorithmic Core

### GERMAINE FD2TD Algorithm

Given complex frequency-domain receiver data `E(f_k, x_r)` for k=0..nf-1:

1. **Spectral taper:** `E *= tukey(nf, alpha=1.0)` along frequency axis
2. **Hermitian assembly (4×nf total):**
   ```
   S = [ E[0:nf],  zeros(nf),  zeros(nf),  E[nf-1:0:-1] ]
   ```
3. **IFFT:** `s(t) = Re{ IFFT(S) }`
4. **Time axis:**
   ```
   df = (FC_high - FC_low) / (nf - 1)
   TmaxFD = 0.25 / df
   dt = TmaxFD / nf  =  1 / (4 * nf * df)
   ```
5. **Extract:** `s[1 : nmaxFD]` where `nmaxFD = int(TmaxTD / dt)`
6. **Normalize:** `s /= max(|s|)`
7. **Display:** `imshow(s, vmin=-clip, vmax=clip)`

### Variables

- `nf` = 80 (number of frequencies, from config)
- `FC_low` = 1 MHz, `FC_high` = 200 MHz
- `df` ≈ 2.52 MHz
- `dt` ≈ 1.24 ns
- `TmaxFD` ≈ 99.2 ns (max recoverable time from FD)
- `TmaxTD` = 80 ns (display window)
- `nmaxFD` ≈ 64 samples
- `clip` = 2.5e-3 (from config `freq_sweep.clip`)

### Source Amplitude (matches GERMAINE RHS_source_TE.c)

```
amp = -(i * omega * mu0) / dh^2
```

This is frequency-dependent and already encodes the TE-mode source spectrum.
No additional wavelet multiplication is needed — the Tukey taper is sufficient.

### References

- Koehn, D., De Nil, D. and Rabbel, W. (2017) DOI: 10.13140/RG.2.2.29354.03523
- GERMAINE GitHub: github.com/daniel-koehn/GERMAINE/par/visu/FD2TD.ipynb
- Layek & Sengupta (2021) DOI: 10.1007/s00024-021-02685-3

## Files Modified

Only ONE file is rewritten. No other files are touched:

**`examples/run_forward_shotgather.py`** — Complete rewrite.
- Reads config from `input/input_shotgather.yaml`
- Builds Layek 2021 layered model (or any model from config)
- Solves FDFD at nf frequencies (parallelized with ThreadPoolExecutor)
- Implements GERMAINE FD2TD conversion inline (no external function)
- Plots color shot gather using `plot_shotgather_color()`
- Plots wiggle shot gather using `plot_shotgather_wiggle()`

Existing modules used unchanged:
- `scripts/forward_fdfd.py` — `build_helmholtz_2d`, `solve_forward`, `MU0`
- `scripts/plot_shotgather.py` — `plot_shotgather_color`, `plot_shotgather_wiggle`
- `scripts/config_loader.py` — config loading helpers
- `scripts/_cli.py` — CLI argument helpers
- `create_models/build_layered_layek2021.py` — model builder

## High-level Architecture

```
input/input_shotgather.yaml
        ↓
config_loader → domain, PML, freq_sweep, source, receivers, model
        ↓
build_layered_layek2021_model() → epsr(nz,nx), sigma(nz,nx)
        ↓
for fi in 1..nf:
    omega = 2π × freqs[fi]
    A = build_helmholtz_2d(epsr, sigma, dh, omega, npml, ...)
    src_amp = -(omega × MU0 × j) / dh²
    Ez = solve_forward(A, src_ix, src_iz, nx, nz, src_amp)
    freq_data[fi, :] = Ez[rec_iz, rec_indices]
        ↓
GERMAINE FD2TD:
    freq_data *= tukey(nf, 1.0)
    S = [freq_data, zeros, zeros, flipud(freq_data)]
    sg = real(ifft(S, axis=0))
    sg = sg[1:nmaxFD, :] / max(sg)
        ↓
plot_shotgather_color(sg, ..., clip_frac=clip)
plot_shotgather_wiggle(sg, ...)
        ↓
results/shotgather_layek2021/*.png, *.npz
```

## Implementation Phases

### Phase 1 — FDFD Solve Loop (reuse existing)
Same as before: loop over nf linspace frequencies, build A, solve, extract
receiver traces. This part works correctly.

### Phase 2 — GERMAINE FD2TD (the critical change)
Replace `freq_to_timedomain()` with inline GERMAINE algorithm:
- Tukey(1) window
- 4×nf concatenation without conj or ifftshift
- `np.fft.ifft` + `np.real()`
- Time axis from `0.25/df`

### Phase 3 — Display
- Global max normalization
- Fixed clip from config (2.5e-3)
- Pass to existing `plot_shotgather_color/wiggle`
- No AGC by default (matches reference appearance)

## TODO Execution Plan

```
1. Delete current run_forward_shotgather.py content
2. Write new script with GERMAINE FD2TD algorithm:
   a. CLI: --stag2, --ncpus, --config, --clip, --display-tmax-ns
   b. Config loading (same as before)
   c. Model building (same as before)
   d. FDFD solve loop with ThreadPoolExecutor (same as before)
   e. Tukey(1) taper on freq_data along axis=0
   f. GERMAINE spectrum assembly: [E, 0, 0, flipud(E)]
   g. np.fft.ifft(axis=0) + np.real()
   h. Time axis: dt = 0.25/(df*nf), extract 1:nmaxFD
   i. Global max normalization
   j. Plot with clip from config
3. Run: python examples/run_forward_shotgather.py --stag2 --ncpus 10
4. Compare output with reference image (KakaoTalk_20260310_170239986.png)
5. Adjust clip value if needed to match reference appearance
```
