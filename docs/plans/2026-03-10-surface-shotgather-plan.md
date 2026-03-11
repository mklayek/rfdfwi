# Surface Acquisition Shot Gather — Implementation Plan

**Date**: 2026-03-10
**Reference**: Layek & Sengupta (2021), Figures 5–7

## Objective

Add support for a **surface acquisition shot gather** matching Layek 2021 style:
one source at the centre of the surface, multiple receivers along the surface with
0.10 m spacing, 80 frequencies (1–200 MHz), time-domain conversion via Hermitian
IFFT, and AGC-processed colour + wiggle plots with an absolute **Distance (m)**
x-axis (not signed offset).

## Paper Parameters (Layek 2021, Figs 5–7)

| Parameter           | Value                               |
|---------------------|-------------------------------------|
| Grid                | 120 × 80, dh = 0.05 m              |
| Sources             | 60, surface, 0.10 m spacing         |
| Receivers           | 60, surface, 0.10 m spacing         |
| Shot gather source  | x = 3.0 m (centre)                  |
| Frequencies         | 80, linspace 1–200 MHz              |
| AGC window          | 20 ns, threshold 10 ns              |
| X-axis              | Distance (m) — absolute position    |
| Y-axis              | Time (ns) — increasing downward     |

## Adaptation for 200×200 Grid

- Source: ix=99, iz=20 (centre of interior domain)
- Receivers: iz=20, ix=20..179, step=2 → 80 receivers at 0.10 m spacing
- Frequencies: 80, linspace 1–200 MHz (df ≈ 2.52 MHz)

## Changes

### 1. `input/input_shotgather.yaml` [NEW — already created]

New YAML config with surface acquisition parameters.

### 2. `scripts/config_loader.py` [MODIFY]

Add `step` parameter to `get_receivers()` line mode (line 101):
```python
step = int(r.get("ix_step", r.get("step", 1)))
return [(ix, iz) for ix in range(i_start, i_end + 1, step)]
```

### 3. `examples/run_forward_shotgather.py` [MODIFY]

- Add `--x-axis {offset,distance}` CLI arg
- Add `--agc-window-ns` CLI arg (converts ns → samples)
- Prefer `input/input_shotgather.yaml` when no `--config` given
- Read default `nf` from config freq_sweep instead of hardcoding 50
- Pass `x_label`, `agc_window` to plot functions

### 4. `scripts/plot_shotgather.py` [MODIFY]

- Add `x_label` parameter to `plot_shotgather_color()` and `plot_shotgather_wiggle()`
- Default: `"Offset [m]"` (backward compatible)
- When `x_label` contains "Distance": mark source at `src_pos_m` instead of at 0

## Usage

```bash
# Layek 2021 style (distance x-axis, 80 freqs, 0.10m spacing)
python examples/run_forward_shotgather.py --stag2 --ncpus 15 \
    --config input/input_shotgather.yaml \
    --x-axis distance --rec-step 2 \
    --fc-low 1e6 --fc-high 200e6 --nf 80 \
    --agc-window-ns 20 --tmax-ns 150

# Default (offset x-axis, 50 freqs, every-cell receivers)
python examples/run_forward_shotgather.py --stag2 --ncpus 15
```
