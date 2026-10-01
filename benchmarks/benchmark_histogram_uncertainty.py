"""Reproducible count-histogram uncertainty diagnostics without private data.

Run with the nfit environment from the repository root::

    PYTHONPATH=src python benchmarks/benchmark_histogram_uncertainty.py

The reference assumes independent unit-weight Poisson events and exactly known
exposure. Its observed numerator variance is the sum of counts, including zero
for a covered empty cell. That observed variance is not a confidence interval
for the unknown rate. The cell-floor diagnostic models the historical DGS
finalization rule before calling nfit's actual histogram pooling function.
It is a diagnostic of that policy, not an assertion that the policy is correct.
Shared background/calibration covariance and fractional events are outside
this reference's independent-event assumptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nfit.histogram_reduction import pool_normalized_histogram
from nfit.mdevent import FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER


def pool_counts(counts, exposure, *, cell_floor=False, mask=None):
    """Pool independent unit-weight counts through nfit's numerical service.

    ``cell_floor`` reproduces DGS's covered-zero error substitution (unit RMS
    event weight), making its downstream effect visible without raw-file I/O.
    """
    counts = np.asarray(counts, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
    measured = exposure > 0.0
    signal = np.full(counts.shape, np.nan)
    variance = np.full(counts.shape, np.nan)
    np.divide(counts, exposure, out=signal, where=measured)
    np.divide(counts, exposure**2, out=variance, where=measured)
    if cell_floor:
        zeros = measured & (counts == 0.0)
        variance[zeros] = (FELDMAN_COUSINS_ZERO_COUNT_68_PERCENT_UPPER / exposure[zeros])**2
    value, variance, total = pool_normalized_histogram(
        signal, variance, exposure, axes=(0,), mask=mask,
    )
    return {
        "intensity": float(value),
        "variance": float(variance),
        "standard_error": float(np.sqrt(variance)),
        "exposure": float(total),
    }


def sampling_reference(*, trials=20_000, seed=37189):
    """Check repeated-sampling variance and known-sigma one-sigma coverage.

    Coverage uses the known generating variance at moderate total counts. It
    does not validate small-count Gaussian intervals or per-cell error floors.
    """
    if trials < 2:
        raise ValueError("trials must be at least two")
    rate = 4.0
    exposure = np.array([1.0, 2.0, 10.0, 20.0])
    samples = np.random.default_rng(seed).poisson(rate * exposure, size=(trials, exposure.size))
    estimates, _, totals = pool_normalized_histogram(
        samples / exposure, samples / exposure**2,
        np.broadcast_to(exposure, samples.shape), axes=(1,),
    )
    expected_variance = rate / float(exposure.sum())
    return {
        "seed": seed,
        "trials": trials,
        "true_rate": rate,
        "total_exposure": float(totals[0]),
        "expected_variance": expected_variance,
        "empirical_mean": float(np.mean(estimates)),
        "empirical_variance": float(np.var(estimates, ddof=1)),
        "known_sigma_coverage": float(np.mean(np.abs(estimates - rate) <= np.sqrt(expected_variance))),
    }


def diagnostic_report(*, trials=20_000, seed=37189):
    """Return measured pooling results and independently derived references."""
    counts = np.zeros(100)
    counts[0] = 25.0
    sparse_reference = pool_counts(counts, np.ones(100))
    cell_floor = pool_counts(counts, np.ones(100), cell_floor=True)
    coarse_reference = pool_counts([25.0], [100.0])

    uneven_counts = np.array([1.0, 30.0, 0.0])
    uneven_exposure = np.array([1.0, 9.0, 10.0])
    sample = pool_counts([4.0, 6.0], [4.0, 6.0])
    background = pool_counts([20.0, 40.0], [10.0, 20.0])
    scale = 2.0
    net_variance = sample["variance"] + scale**2 * background["variance"]
    return {
        "assumptions": {
            "events": "independent unit-weight Poisson counts",
            "exposure": "known exactly; arbitrary consistent exposure units",
            "cell_floor": "historical pre-0.106 DGS covered-zero Feldman-Cousins upper limit used as a standard error",
            "confidence_limit_is_variance": False,
            "mantid_measured": False,
        },
        "sparse_cells": {
            "cells": 100,
            "covered_empty_cells": 99,
            "total_counts": 25,
            "independent_poisson_reference": sparse_reference,
            "modeled_dgs_cell_floor_then_nfit_pooling": cell_floor,
            "standard_error_ratio": cell_floor["standard_error"] / sparse_reference["standard_error"],
            "analytic_reference_variance": 25.0 / 100.0**2,
            "same_observation_in_one_cell": coarse_reference,
        },
        "uneven_exposure": {
            "counts": uneven_counts.tolist(),
            "exposure": uneven_exposure.tolist(),
            "nfit_pooled": pool_counts(uneven_counts, uneven_exposure),
            "analytic_intensity": 31.0 / 20.0,
            "analytic_variance": 31.0 / 20.0**2,
            "unweighted_cell_intensity_mean": float(np.mean(uneven_counts / uneven_exposure)),
        },
        "missing_exposure": {
            "nfit_pooled": pool_counts([8.0, 999.0, 0.0], [2.0, 0.0, 3.0]),
            "analytic_intensity": 8.0 / 5.0,
            "analytic_variance": 8.0 / 5.0**2,
        },
        "independent_background": {
            "sample": sample,
            "background": background,
            "background_scale": scale,
            "net_intensity": sample["intensity"] - scale * background["intensity"],
            "net_variance": net_variance,
            "net_standard_error": float(np.sqrt(net_variance)),
            "analytic_intensity": 10.0 / 10.0 - scale * 60.0 / 30.0,
            "analytic_variance": 10.0 / 10.0**2 + scale**2 * 60.0 / 30.0**2,
        },
        "repeated_sampling": sampling_reference(trials=trials, seed=seed),
    }


def _load_slab(path, required):
    with np.load(path, allow_pickle=False) as source:
        missing = set(required) - set(source.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        return {name: np.asarray(source[name], dtype=float) for name in required}


def _checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_l2(actual, reference):
    scale = float(np.linalg.norm(reference.ravel()))
    return float(np.linalg.norm((actual - reference).ravel()) / scale) if scale > 0.0 else None


def _ratio_summary(values):
    values = values[np.isfinite(values)]
    if not values.size:
        return {"bins": 0, "median": None, "p95": None, "max": None}
    return {
        "bins": int(values.size),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def compare_supplied_slabs(nfit_path, mantid_path, mantid_data_path, *, axes=(0, 2)):
    """Compare supplied matched 4D slabs; this does not run either reduction.

    Recover nfit's numerator variance before comparison so differing exposure
    cannot masquerade as differing accumulated event variance. Aggregate both
    on common finite, positive-exposure support. Report standard-error ratios
    with common exposure and with each engine's own exposure separately.
    """
    fields = ("signal", "norm", "counts", "errors", "numerator")
    edge_fields = tuple(f"edges{i}" for i in range(4))
    nfit = _load_slab(nfit_path, fields + edge_fields)
    mantid = _load_slab(mantid_path, fields + edge_fields)
    mantid_data = _load_slab(mantid_data_path, ("numerator", "variance"))
    shape = nfit["signal"].shape
    if len(shape) != 4:
        raise ValueError("matched slabs must have four dimensions")
    if len(set(axes)) != len(axes) or any(axis not in range(4) for axis in axes):
        raise ValueError("aggregation axes must be distinct integers from 0 to 3")
    for label, arrays in (("nfit", nfit), ("mantid", mantid), ("mantid_data", mantid_data)):
        for name, values in arrays.items():
            if name not in edge_fields and values.shape != shape:
                raise ValueError(f"{label} {name}: shape {values.shape} does not match {shape}")
    edge_differences = {}
    for i, name in enumerate(edge_fields):
        edges = nfit[name]
        if edges.shape != (shape[i] + 1,) or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError(f"invalid {name}: require finite increasing bin edges matching shape")
        other = mantid[name]
        if other.shape != edges.shape or not np.all(np.isfinite(other)) or np.any(np.diff(other) <= 0):
            raise ValueError(f"invalid Mantid {name}: require finite increasing bin edges matching shape")
        # Mantid records coordinates as float32. Allow that representational
        # difference, report it, and reject scientifically different grids.
        tolerance = 2 * np.finfo(np.float32).eps
        if not np.allclose(edges, other, rtol=tolerance, atol=tolerance * float(np.min(np.diff(edges)))):
            raise ValueError(f"nfit and Mantid {name} do not match")
        edge_differences[name] = float(np.max(np.abs(edges - other)))
    if np.any(mantid_data["variance"][np.isfinite(mantid_data["variance"])] < 0):
        raise ValueError("Mantid numerator variance cannot be negative")
    for label, arrays in (("nfit", nfit), ("mantid", mantid)):
        for name in ("errors", "counts"):
            if np.any(arrays[name][np.isfinite(arrays[name])] < 0):
                raise ValueError(f"{label} {name} cannot be negative")
    with np.errstate(invalid="ignore", over="ignore"):
        nfit_variance = (nfit["errors"] * nfit["norm"])**2
    reference_variance = mantid_data["variance"]
    common = (nfit["norm"] > 0) & (mantid["norm"] > 0)
    for arrays in (nfit, mantid, mantid_data):
        for name, values in arrays.items():
            if name not in edge_fields:
                common &= np.isfinite(values)
    common &= np.isfinite(nfit_variance)

    def region(mask):
        valid = common & mask
        positive_variance = valid & (reference_variance > 0)
        empty = valid & (nfit["counts"] == 0) & (mantid["counts"] == 0)
        vn = np.sum(np.where(valid, nfit_variance, 0.0), axis=axes)
        vm = np.sum(np.where(valid, reference_variance, 0.0), axis=axes)
        nn = np.sum(np.where(valid, nfit["norm"], 0.0), axis=axes)
        nm = np.sum(np.where(valid, mantid["norm"], 0.0), axis=axes)
        positive = (vm > 0) & (nn > 0) & (nm > 0)
        ratios = np.sqrt(vn[positive] / vm[positive])
        return {
            "common_covered_cells": int(np.count_nonzero(valid)),
            "count_max_abs_difference": float(np.max(np.abs(nfit["counts"][valid] - mantid["counts"][valid]))) if np.any(valid) else None,
            "numerator_relative_l2": _relative_l2(nfit["numerator"][valid], mantid_data["numerator"][valid]),
            "normalization_relative_l2": _relative_l2(nfit["norm"][valid], mantid["norm"][valid]),
            "nonzero_numerator_variance_relative_l2": _relative_l2(nfit_variance[positive_variance], reference_variance[positive_variance]),
            "covered_empty_cells": int(np.count_nonzero(empty)),
            "covered_empty_nfit_numerator_variance_sum": float(np.sum(nfit_variance[empty])),
            "covered_empty_mantid_numerator_variance_sum": float(np.sum(reference_variance[empty])),
            "aggregate_standard_error_ratio_common_exposure": _ratio_summary(ratios),
            "aggregate_standard_error_ratio_own_exposure": _ratio_summary(ratios * nm[positive] / nn[positive]),
        }

    result = {
        "source": "supplied diagnostic slabs; no fresh Mantid reduction or binning was run",
        "shape": list(shape),
        "edges": {name: nfit[name].tolist() for name in edge_fields},
        "edge_max_abs_differences": edge_differences,
        "edge_validation": "matched within two float32 epsilons relative plus two epsilons times minimum bin width absolute",
        "aggregation_axes": list(axes),
        "provenance": {
            label: {"path": str(Path(path)), "sha256": _checksum(path)}
            for label, path in (("nfit", nfit_path), ("mantid", mantid_path), ("mantid_data", mantid_data_path))
        },
        "all_common_support": region(np.ones(shape, dtype=bool)),
    }
    k = (nfit["edges1"][:-1] + nfit["edges1"][1:]) / 2
    energy = (nfit["edges3"][:-1] + nfit["edges3"][1:]) / 2
    roi = (np.abs(k)[None, :, None, None] <= 0.1) & ((energy >= 6) & (energy <= 14))[None, None, None, :]
    if np.any(roi & common):
        result["central_roi"] = {
            "selection": "axis 1 centers |K| <= 0.1; axis 3 centers 6 <= E <= 14; requires HKLE slab axis convention",
            **region(roi),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=37189)
    parser.add_argument("--output", type=Path, help="also write the JSON report to this path")
    parser.add_argument("--nfit-slab", type=Path)
    parser.add_argument("--mantid-slab", type=Path)
    parser.add_argument("--mantid-data", type=Path)
    parser.add_argument("--aggregation-axes", default="0,2", help="comma-separated slab axes to collapse (default: 0,2)")
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("--trials must be at least two")
    paths = (args.nfit_slab, args.mantid_slab, args.mantid_data)
    if any(path is not None for path in paths) and not all(path is not None for path in paths):
        parser.error("--nfit-slab, --mantid-slab, and --mantid-data must be supplied together")
    report = diagnostic_report(trials=args.trials, seed=args.seed)
    if all(path is not None for path in paths):
        try:
            axes = tuple(int(axis.strip()) for axis in args.aggregation_axes.split(",") if axis.strip())
            report["supplied_slabs"] = compare_supplied_slabs(*paths, axes=axes)
        except (ValueError, OSError) as error:
            parser.error(str(error))
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
