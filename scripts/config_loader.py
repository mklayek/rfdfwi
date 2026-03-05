"""
Load and validate YAML configuration for forward and inversion.
"""
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_forward_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract forward-related config (standalone forward or nested under 'forward')."""
    if "forward" in config and isinstance(config["forward"], dict):
        return config["forward"].copy()
    return config.copy()


def get_inversion_config(config: dict[str, Any]) -> dict[str, Any]:
    """Extract inversion-related config."""
    return config.get("inversion", {})


def get_domain(config: dict[str, Any]) -> tuple[int, int, float, float]:
    d = config.get("domain", config)
    nx = int(d.get("nx", 101))
    nz = int(d.get("nz", 81))
    dx = float(d.get("dx", 0.01))
    dz = float(d.get("dz", 0.01))
    return nx, nz, dx, dz


def get_pml(config: dict[str, Any]) -> tuple[int, int]:
    pml = config.get("pml", {})
    if isinstance(pml, dict):
        return int(pml.get("npx", 10)), int(pml.get("npz", 10))
    return 10, 10


def get_freq_hz(config: dict[str, Any]) -> float:
    return float(config.get("freq_hz", 900e6))


def get_source(config: dict[str, Any]) -> tuple[int, int]:
    s = config.get("source", config.get("sources", [{}]))
    if isinstance(s, list) and s:
        s = s[0]
    return int(s.get("ix", 50)), int(s.get("iz", 2))


def get_receivers(config: dict[str, Any], nx: int) -> list[tuple[int, int]]:
    r = config.get("receivers", {})
    if isinstance(r, dict) and r.get("mode") == "line":
        iz = int(r.get("iz", 2))
        i_start = int(r.get("ix_start", 0))
        i_end = int(r.get("ix_end", nx - 1))
        return [(ix, iz) for ix in range(i_start, i_end + 1)]
    if isinstance(r, list):
        return [(int(p["ix"]), int(p["iz"])) for p in r]
    return [(50, 2)]


def get_freq_sweep(config: dict[str, Any]) -> dict[str, Any]:
    """
    Return frequency-sweep parameters, matching MATLAB inp_GPRmodel1.m.

    Keys returned:
      fc_low   [Hz]  Lower frequency
      fc_high  [Hz]  Upper frequency
      nf       [-]   Number of frequency samples
      df       [Hz]  Frequency step  (fc_high-fc_low)/(nf-1)
      clip          Blackman-Harris amplitude clip (relative)
      clip1         Secondary clip
      tmax_td  [s]   Maximum time for FD->TD transform
    """
    fs = config.get("freq_sweep", {})
    fc_low  = float(fs.get("fc_low",  50e6))
    fc_high = float(fs.get("fc_high", 200e6))
    nf      = int(fs.get("nf",  50))
    df      = (fc_high - fc_low) / max(nf - 1, 1)
    clip    = float(fs.get("clip",  2.5e-3))
    clip1   = float(fs.get("clip1", 1.0e-2))
    tmax_td = float(fs.get("tmax_td", 150e-9))
    return {
        "fc_low": fc_low, "fc_high": fc_high,
        "nf": nf, "df": df,
        "clip": clip, "clip1": clip1,
        "tmax_td": tmax_td,
    }


def get_acquisition_sources(config: dict[str, Any]) -> list[dict[str, int]]:
    """
    Return list of source dicts {ix, iz}.

    Supports:
      acquisition.mode == '4sided'  — generate MATLAB-style 4-sided layout
      acquisition.sources list      — explicit list
    Falls back to a single surface-centre source.
    """
    acq = config.get("acquisition", {})
    if acq.get("mode") == "4sided":
        from create_models.build_models import build_4sided_acquisition
        npml          = int(acq.get("npml", 10))
        nsrc_per_side = int(acq.get("nsrc_per_side", 20))
        nrec_per_side = int(acq.get("nrec_per_side", 40))
        srcs, _ = build_4sided_acquisition(npml, nrec_per_side, nsrc_per_side)
        return srcs
    src_list = acq.get("sources", [])
    if src_list:
        return [{"ix": int(s["ix"]), "iz": int(s["iz"])} for s in src_list]
    return [{"ix": 99, "iz": 20}]


def get_acquisition_receivers(config: dict[str, Any]) -> list[dict[str, int]]:
    """
    Return list of receiver dicts {ix, iz}.

    Supports:
      acquisition.mode == '4sided'  — generate MATLAB-style 4-sided layout
      acquisition.receivers dict    — line mode or list
    """
    acq = config.get("acquisition", {})
    if acq.get("mode") == "4sided":
        from create_models.build_models import build_4sided_acquisition
        npml          = int(acq.get("npml", 10))
        nsrc_per_side = int(acq.get("nsrc_per_side", 20))
        nrec_per_side = int(acq.get("nrec_per_side", 40))
        _, recs = build_4sided_acquisition(npml, nrec_per_side, nsrc_per_side)
        return recs
    rec = acq.get("receivers", {})
    if isinstance(rec, dict) and rec.get("mode") == "line":
        iz      = int(rec.get("iz", 20))
        ix_start = int(rec.get("ix_start", 20))
        ix_end   = int(rec.get("ix_end", 179))
        return [{"ix": ix, "iz": iz} for ix in range(ix_start, ix_end + 1)]
    if isinstance(rec, list):
        return [{"ix": int(r["ix"]), "iz": int(r["iz"])} for r in rec]
    return []
