"""Shared, standard-library-only inputs for manual DGS workflow benchmarks.

Neither engine imports the other. Persist this dictionary as the common JSON
configuration before running a pilot or full ordinary sequential workflow.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

SYMMETRY = {
    "1": "x,y,z",
    "6": "x,y,z;y,z,x;z,x,y;-x,-y,-z;-y,-z,-x;-z,-x,-y",
    "12": "x,y,z;y,z,x;z,x,y;y,x,z;x,z,y;z,y,x;-x,-y,-z;-y,-z,-x;-z,-x,-y;-y,-x,-z;-x,-z,-y;-z,-y,-x",
}

RAW_PIPELINE_TRIAL_MODES = (
    "baseline", "coalesced", "lookup", "lookup-coalesced", "compiled", "compiled-coalesced",
    "runs1", "runs2", "runs4", "processes2", "processes4", "pulse-linear", "prefetch", "threads2", "threads4",
    "threads2-coalesced", "threads4-coalesced",
)


def center_edges(lower, upper, step):
    """The existing nfit center-limit rule, expressed without a NumPy import."""
    count = max(math.floor((upper - lower) / step + 1e-10) + 1, 1)
    return [lower + (index - .5) * step for index in range(count + 1)]


def benchmark_settings(preset="sequoia"):
    if preset == "sequoia":
        root = Path("/SNS/SEQ/IPTS-37189")
        runs = list(range(392985, 393470)) + list(range(393500, 393632))
        prefix = "SEQ"
        instrument = "SEQUOIA"
        normalization = str(root / "shared/tempMDE/van382236_D")
        mask = normalization
        ub = str(root / "shared/tempMDE/All_Ei60_T5K_with_UB.nxs")
        grid = dict(vectors=[[1, 1, 1], [1, -1, 0], [1, 1, -2]],
                    bin_edges=[center_edges(lo, hi, step) for lo, hi, step in
                               [(-1., 1., .03), (-2., 2., .03), (-1., 1., .02), (0., 50., .5)]])
        ei = None
    elif preset in {"hyspec-50k34", "hyspec-50k70"}:
        root = Path("/SNS/HYS/IPTS-36860")
        bank = 34 if preset.endswith("34") else 70
        runs = list(range(505555, 505916)) if bank == 34 else list(range(506277, 506638))
        prefix = "HYS"
        instrument = "HYSPEC"
        normalization = None
        mask = str(root / f"shared/nfit/.nfit-diagnostics/6ar/hyspec_tip_mask_{runs[0]}.nxs")
        ub = str(root / f"shared/Ei15meV_50K_240Hz_s2_{bank}.nxs")
        grid = dict(vectors=[[1, 1, 0], [0, 0, 1], [1, -1, 0]],
                    bin_edges=[center_edges(lo, hi, step) for lo, hi, step in
                               [(-8., 8., .2), (-8., 8., .2), (-.12, .12, .03), (-14., 14., .5)]])
        ei = 15.
    else:
        raise ValueError(f"Unknown benchmark preset {preset!r}")
    return dict(schema_version=1, preset=preset, instrument=instrument,
        run_numbers=runs, raw_files=[str(root / f"nexus/{prefix}_{run}.nxs.h5") for run in runs],
        normalization_file=normalization, mask_file=mask, ub_source=ub,
        grid=grid, symmetry_operations=copy.deepcopy(SYMMETRY), primary_copies=6,
        rebin_copies=[1, 6, 12], threads=64, ram_limit_mib=200000,
        output_root=str(root / "shared/nfit/benchmarks/6A-P-node19"),
        reduction=dict(ei_override=ei, t0_override=None, emin_fraction=-.95,
                       emax_fraction=.95, bad_pulses_threshold=0.,
                       time_independent_background="", goniometer="Universal", q_frame="Q_sample"),
        cache_state="OS/filesystem caches uncontrolled; do not describe this as cold disk")


def load_settings(config_path=None, preset="sequoia"):
    settings = benchmark_settings(preset) if config_path is None else json.loads(Path(config_path).read_text())
    if settings.get("schema_version") != 1:
        raise ValueError("Unsupported manual benchmark schema")
    if len(settings["run_numbers"]) != len(settings["raw_files"]) or not settings["run_numbers"]:
        raise ValueError("Run numbers and raw sources must have matching nonempty membership")
    if len(set(settings["run_numbers"])) != len(settings["run_numbers"]):
        raise ValueError("Benchmark sources must not contain duplicate runs")
    if settings["reduction"]["q_frame"] != "Q_sample":
        raise ValueError("This benchmark compares sample-frame DGS datasets")
    return settings


def select_runs(settings, run_numbers=None, first_runs=None):
    """Return a copied configuration for a bounded pilot or the full membership."""
    result = copy.deepcopy(settings)
    if run_numbers is not None and first_runs is not None:
        raise ValueError("Choose explicit runs or first-runs, not both")
    wanted = list(settings["run_numbers"] if run_numbers is None else run_numbers)
    if first_runs is not None:
        if first_runs <= 0:
            raise ValueError("first-runs must be positive")
        wanted = wanted[:first_runs]
    lookup = dict(zip(settings["run_numbers"], settings["raw_files"], strict=True))
    if not wanted or len(set(wanted)) != len(wanted) or any(run not in lookup for run in wanted):
        raise ValueError("Pilot membership must select unique runs from the configured dataset")
    result["run_numbers"] = wanted
    result["raw_files"] = [lookup[run] for run in wanted]
    return result


def output_directory(settings, engine, tag):
    """Require a new scientific output directory under the configured IPTS."""
    if engine not in {"mantid", "nfit"} or not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("Use a known engine and a simple new job tag")
    root = Path(settings["output_root"]).expanduser().resolve()
    if not any(re.fullmatch(r"IPTS-\d+", part) for part in root.parts):
        raise ValueError("All benchmark outputs must be inside an IPTS directory")
    if "shared" not in root.parts or "nfit" not in root.parts:
        raise ValueError("Benchmark outputs belong in the IPTS shared/nfit directory")
    return root / engine / tag
