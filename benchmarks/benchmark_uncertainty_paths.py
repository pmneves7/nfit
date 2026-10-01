"""Trace native viewer uncertainty paths using supplied matched HKLE slabs.

No Mantid runtime is imported or executed. The supplied raw numerator variance
is a reference datum, not a new reduction. The empty-cell-floor removal is an
explicit diagnostic for unsubtracted event histograms; it is not a generic
correction for reconstructed or background-subtracted data. All propagation
here assumes independent diagonal statistical variances and known exposure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nfit.box_cuts import box_membership, rotated_box_profiles
from nfit.histogram_statistics import (
    EVENT_STATISTICS_KEY,
    EVENT_STATISTICS_METADATA,
    event_statistics_channels,
)
from nfit.mdhisto import MDHistoAxis, MDHistoChannel, MDHistoData
from nfit.plotting_core import (
    MDHistoSliceViewer,
    coarsen_mdhisto_view,
    integrated_box_sum,
    inverse_variance_weighted_profile,
)


def _load(path, fields):
    with np.load(path, allow_pickle=False) as source:
        missing = set(fields) - set(source.files)
        if missing:
            raise ValueError(f"{path}: missing arrays {sorted(missing)}")
        result = {name: np.asarray(source[name], dtype=float) for name in fields}
        if "mask" in source:
            result["mask"] = np.asarray(source["mask"], dtype=bool)
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(nfit, mantid, reference):
    shape = nfit["signal"].shape
    if len(shape) != 4 or any(size < 1 for size in shape):
        raise ValueError("require nonempty four-dimensional HKLE slabs")
    differences = []
    for label, arrays in (("nfit", nfit), ("mantid", mantid), ("reference", reference)):
        for name, values in arrays.items():
            if not name.startswith("edges") and values.shape != shape:
                raise ValueError(f"{label} {name}: shape does not match {shape}")
    for dim in range(4):
        a, b = nfit[f"edges{dim}"], mantid[f"edges{dim}"]
        for edges in (a, b):
            if edges.shape != (shape[dim] + 1,) or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
                raise ValueError(f"invalid axis {dim} edges")
        eps = 2 * np.finfo(np.float32).eps
        if not np.allclose(a, b, rtol=eps, atol=eps * np.min(np.diff(a))):
            raise ValueError(f"axis {dim} edges do not match")
        differences.append(float(np.max(np.abs(a - b))))
    for label, arrays, names in (
        ("nfit", nfit, ("errors", "counts")),
        ("mantid", mantid, ("errors", "counts")),
        ("reference", reference, ("variance",)),
    ):
        for name in names:
            finite = arrays[name][np.isfinite(arrays[name])]
            if np.any(finite < 0):
                raise ValueError(f"{label} {name} cannot be negative")
    return shape, differences


def fringe_mask(covered, radius=2):
    """Find covered pixels near uncovered neighbors, excluding crop exterior."""
    covered = np.asarray(covered, dtype=bool)
    if covered.ndim != 2 or radius < 1:
        raise ValueError("fringe requires a two-dimensional coverage map and positive radius")
    height, width = covered.shape
    padded = np.pad(~covered, radius, constant_values=False)
    near_missing = np.zeros_like(covered)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            near_missing |= padded[radius + dy:radius + dy + height, radius + dx:radius + dx + width]
    return covered & near_missing


def low_exposure_mask(exposure, covered, energies):
    """Select the bottom exposure decile in each E>=3 row; ties are included."""
    selected = np.zeros_like(covered)
    for row in range(len(energies)):
        good = covered[row]
        if energies[row] >= 3 and np.any(good):
            threshold = np.quantile(exposure[row, good], 0.1)
            selected[row] = good & (exposure[row] <= threshold)
    return selected


def _map_sum(values, valid):
    return np.sum(np.where(valid, values, 0.0), axis=(1, 2)).T


def _ratio_summary(actual, reference, selected):
    positive = selected & np.isfinite(actual) & np.isfinite(reference) & (reference > 0)
    ratios = actual[positive] / reference[positive]
    return {
        "positive_reference_bins": int(ratios.size),
        "min": float(np.min(ratios)) if ratios.size else None,
        "p5": float(np.percentile(ratios, 5)) if ratios.size else None,
        "median": float(np.median(ratios)) if ratios.size else None,
        "p95": float(np.percentile(ratios, 95)) if ratios.size else None,
        "max": float(np.max(ratios)) if ratios.size else None,
        "zero_reference_nonzero_current_bins": int(np.count_nonzero(selected & (reference == 0) & (actual > 0))),
    }


def _relative_l2(actual, reference):
    norm = float(np.linalg.norm(reference.ravel()))
    return float(np.linalg.norm((actual - reference).ravel()) / norm) if norm > 0 else None


def _coverage_stats(nfit, mantid):
    """Keep source-cell mismatches visible before common-support filtering."""
    geometric = []
    statistics = {}
    for name, slab in (("nfit", nfit), ("mantid", mantid)):
        exposed = np.isfinite(slab["norm"]) & (slab["norm"] > 0)
        masked = slab.get("mask", np.zeros(exposed.shape, dtype=bool))
        covered = exposed & ~masked
        geometric.append(covered)
        positive_events = np.isfinite(slab["counts"]) & (slab["counts"] > 0)
        statistics[name] = {
            "covered_cells": int(np.count_nonzero(covered)),
            "uncovered_cells": int(np.count_nonzero(~covered)),
            "geometrically_unexposed_cells": int(np.count_nonzero(~exposed)),
            "masked_cells": int(np.count_nonzero(masked)),
        }
        for label, selected in (("uncovered", ~covered), ("geometrically_unexposed", ~exposed), ("masked", masked)):
            has_events = selected & positive_events
            statistics[name][f"{label}_cells_with_positive_events"] = int(np.count_nonzero(has_events))
            statistics[name][f"events_in_{label}_cells"] = float(np.sum(slab["counts"][has_events]))
    for label, selected in (("nfit_covered_mantid_uncovered", geometric[0] & ~geometric[1]), ("mantid_covered_nfit_uncovered", geometric[1] & ~geometric[0])):
        statistics[label] = {"cells": int(np.count_nonzero(selected))}
        for name, slab in (("nfit", nfit), ("mantid", mantid)):
            finite = selected & np.isfinite(slab["counts"])
            statistics[label][f"{name}_event_sum"] = float(np.sum(slab["counts"][finite]))
            finite = selected & np.isfinite(slab["numerator"])
            statistics[label][f"{name}_numerator_sum"] = float(np.sum(slab["numerator"][finite]))
    return statistics


def _viewer(nfit, variance, common, storage, event_statistics=False):
    names = ("[H,H,H]", "[K,-K,0]", "[L,L,-2L]", "DeltaE")
    axes = tuple(MDHistoAxis(name, nfit[f"edges{i}"], "meV" if i == 3 else "r.l.u.", "energy" if i == 3 else "momentum") for i, name in enumerate(names))
    with np.errstate(divide="ignore", invalid="ignore"):
        errors = np.sqrt(variance) / nfit["norm"]
    metadata = {"normalization_denominator": nfit["norm"], "zero_event_bins_are_measured": True}
    channels = {}
    if storage == "auxiliary":
        channels["normalization_denominator"] = MDHistoChannel(nfit["norm"], label="Detector-trajectory normalization")
    if event_statistics:
        metadata[EVENT_STATISTICS_KEY] = dict(EVENT_STATISTICS_METADATA)
        channels.update(event_statistics_channels(nfit["numerator"], variance, nfit["norm"]))
    data = MDHistoData(axes, nfit["signal"], errors, ~common, nfit["counts"], metadata=metadata, auxiliary_channels=channels)
    model = MDHistoSliceViewer(data, x_dim=0, y_dim=3, integrate=True)
    for dim in (1, 2):
        centers = axes[dim].centers
        model.selections[dim] = (float(centers[0]), float(centers[-1]))
    return model.slice_arrays()


def _profile(view, roi):
    x0, x1, y0, y1 = roi
    x = (view["x_centers"] >= x0) & (view["x_centers"] <= x1)
    y = (view["y_centers"] >= y0) & (view["y_centers"] <= y1)
    values = view["signal"][np.ix_(y, x)]
    errors = view["errors"][np.ix_(y, x)]
    mean, error = inverse_variance_weighted_profile(values, errors, axis=1)
    summed, summed_error, bins = integrated_box_sum(values, errors)
    rotated = rotated_box_profiles(view, view["signal"], view["errors"], roi, angle=0)
    return {
        "regular_inverse_variance": {"energy": view["y_centers"][y], "signal": mean, "error": error},
        "rotated_zero_degree": {"energy": rotated.y[0], "signal": rotated.y[1], "error": rotated.y[2]},
        "roi_sum": {"signal": summed, "error": summed_error, "pixels": bins},
    }


def diagnose_paths(nfit_path, mantid_path, mantid_data_path, *, roi=(0.13, 0.23, 3.5, 25.0), normalization_storage="metadata", event_statistics=False):
    """Trace actual native viewer services and compare independent reference sums."""
    if normalization_storage not in {"metadata", "auxiliary"}:
        raise ValueError("normalization storage must be metadata or auxiliary")
    if len(roi) != 4 or not np.all(np.isfinite(roi)) or roi[0] >= roi[1] or roi[2] >= roi[3]:
        raise ValueError("ROI must contain increasing finite H and E bounds")
    fields = ("signal", "norm", "counts", "errors", "numerator", "edges0", "edges1", "edges2", "edges3")
    nfit, mantid = _load(nfit_path, fields), _load(mantid_path, fields)
    reference = _load(mantid_data_path, ("numerator", "variance"))
    shape, edge_differences = _validate(nfit, mantid, reference)
    coverage_stats = _coverage_stats(nfit, mantid)
    common = (nfit["norm"] > 0) & (mantid["norm"] > 0)
    for arrays in (nfit, mantid, reference):
        for name, values in arrays.items():
            if name == "mask":
                common &= ~values
            elif not name.startswith("edges"):
                common &= np.isfinite(values)
    current_v = (nfit["errors"] * nfit["norm"])**2
    common &= np.isfinite(current_v)
    removed_v = np.where(nfit["counts"] == 0, 0.0, current_v)
    variants = {"current": current_v, "reference_same_exposure": reference["variance"], "floor_removed_diagnostic": removed_v}
    views = {name: _viewer(nfit, variance, common, normalization_storage, event_statistics) for name, variance in variants.items()}
    base = views["current"]
    h, energy = base["x_centers"], base["y_centers"]
    selection = box_membership(h, energy, roi)
    if not np.any(selection):
        raise ValueError("ROI selects no histogram centers")
    profiles = {name: _profile(view, roi) for name, view in views.items()}
    maps = {"h_centers": h, "energy_centers": energy}
    maps.update({name: _map_sum(values, common) for name, values in {
        "nfit_exposure": nfit["norm"], "mantid_exposure": mantid["norm"],
        "counts": nfit["counts"], "mantid_counts": mantid["counts"],
        "numerator": nfit["numerator"], "mantid_numerator": reference["numerator"],
        "current_numerator_variance": current_v, "reference_numerator_variance": reference["variance"],
        "floor_removed_numerator_variance": removed_v,
    }.items()})
    for name, view in views.items():
        maps[f"{name}_signal"] = view["signal"]
        maps[f"{name}_error"] = view["errors"]
    geometry = []
    for slab in (nfit, mantid):
        valid = np.isfinite(slab["norm"]) & (slab["norm"] > 0)
        valid &= ~slab.get("mask", np.zeros(shape, dtype=bool))
        geometry.append(_map_sum(slab["norm"], valid))
    covered = (geometry[0] > 0) & (geometry[1] > 0)
    fringe = fringe_mask(covered)
    low = low_exposure_mask(geometry[0], covered, energy)
    high_energy = energy[:, None] >= 3
    region_masks = {
        "all_common_covered": covered & high_energy,
        "fringe_two_pixels": fringe & high_energy,
        "bottom_exposure_decile_per_energy_row": low,
        "well_covered_interior": covered & ~fringe & ~low & high_energy,
        "selected_roi": covered & selection,
    }
    regions = {}
    for name, selected in region_masks.items():
        cells = common & selected.T[:, None, None, :]
        zero = cells & (nfit["counts"] == 0)
        total_variance = float(np.sum(current_v[cells]))
        nn, nm = maps["nfit_exposure"][selected], maps["mantid_exposure"][selected]
        exposure_ratios = np.divide(nn, nm, out=np.full(nn.shape, np.nan), where=nm > 0)
        nonempty = cells & ((nfit["counts"] > 0) | (mantid["counts"] > 0))
        regions[name] = {
            "map_pixels": int(np.count_nonzero(selected)),
            "common_finite_cells": int(np.count_nonzero(cells)),
            "covered_empty_cells": int(np.count_nonzero(zero)),
            "covered_empty_fraction": float(np.count_nonzero(zero) / np.count_nonzero(cells)) if np.any(cells) else None,
            "empty_cell_variance_fraction": float(np.sum(current_v[zero]) / total_variance) if total_variance > 0 else None,
            "empty_cell_variance_current": float(np.sum(current_v[zero])),
            "empty_cell_variance_reference": float(np.sum(reference["variance"][zero])),
            "map_error_ratio_same_exposure": _ratio_summary(base["errors"], views["reference_same_exposure"]["errors"], selected),
            "exposure_ratio_nfit_over_mantid": _ratio_summary(exposure_ratios, np.ones(nn.shape), np.ones(nn.shape, dtype=bool)),
            "count_max_abs_difference": float(np.max(np.abs(maps["counts"][selected] - maps["mantid_counts"][selected]))) if np.any(selected) else None,
            "count_comparison_scope": "count_max_abs_difference compares pooled H/E map pixels; fine-cell differences below retain hidden-axis redistribution",
            "fine_cell_count_max_abs_difference": float(np.max(np.abs(nfit["counts"][cells] - mantid["counts"][cells]))) if np.any(cells) else None,
            "fine_cell_count_mismatch_cells": int(np.count_nonzero(nfit["counts"][cells] != mantid["counts"][cells])),
            "fine_cell_numerator_relative_l2": _relative_l2(nfit["numerator"][cells], reference["numerator"][cells]),
            "fine_cell_numerator_max_abs_difference": float(np.max(np.abs(nfit["numerator"][cells] - reference["numerator"][cells]))) if np.any(cells) else None,
            "fine_nonempty_numerator_variance_relative_l2": _relative_l2(current_v[nonempty], reference["variance"][nonempty]),
            "fine_nonempty_numerator_variance_max_abs_difference": float(np.max(np.abs(current_v[nonempty] - reference["variance"][nonempty]))) if np.any(nonempty) else None,
        }
    # Independent reference: sum sufficient statistics over H in the ROI as
    # well as both hidden axes; this is not an inverse-variance mean of ratios.
    x_selected = (h >= roi[0]) & (h <= roi[1])
    y_selected = (energy >= roi[2]) & (energy <= roi[3])
    cut = {"energy": energy[y_selected], "h_centers": h[x_selected]}
    for name in ("counts", "numerator", "nfit_exposure", "mantid_exposure", "mantid_numerator", "current_numerator_variance", "reference_numerator_variance", "floor_removed_numerator_variance"):
        cut[name] = np.sum(maps[name][np.ix_(y_selected, x_selected)], axis=1)
    exposure = cut["nfit_exposure"]
    with np.errstate(divide="ignore", invalid="ignore"):
        cut["analytic_pooled_signal"] = cut["numerator"] / exposure
        for label, name in (("current", "current_numerator_variance"), ("reference", "reference_numerator_variance"), ("floor_removed", "floor_removed_numerator_variance")):
            cut[f"analytic_{label}_error"] = np.sqrt(cut[name]) / exposure
        cut["mantid_own_exposure_signal"] = cut["mantid_numerator"] / cut["mantid_exposure"]
        cut["mantid_own_exposure_error"] = np.sqrt(cut["reference_numerator_variance"]) / cut["mantid_exposure"]
    coarse = {}
    for name, view in views.items():
        x_step = 2 * float(np.median(np.diff(view["x_edges"])))
        y_step = 2 * float(np.median(np.diff(view["y_edges"])))
        displayed = coarsen_mdhisto_view(view, x_step=x_step, y_step=y_step)
        coarse[name] = {"h_centers": displayed["x_centers"], "energy_centers": displayed["y_centers"], "signal": displayed["signal"], "error": displayed["errors"], "profiles": _profile(displayed, roi)}
    stage_ratios = {
        "analytic_pooled_energy_cut": _ratio_summary(cut["analytic_current_error"], cut["analytic_reference_error"], np.ones(cut["energy"].shape, dtype=bool)),
        "display_coarsened_map": _ratio_summary(coarse["current"]["error"], coarse["reference_same_exposure"]["error"], np.ones(coarse["current"]["error"].shape, dtype=bool)),
    }
    for name in ("regular_inverse_variance", "rotated_zero_degree"):
        actual = profiles["current"][name]["error"]
        expected = profiles["reference_same_exposure"][name]["error"]
        stage_ratios[name] = _ratio_summary(actual, expected, np.ones(actual.shape, dtype=bool))
    actual = profiles["current"]["roi_sum"]["error"]
    expected = profiles["reference_same_exposure"]["roi_sum"]["error"]
    stage_ratios["roi_sum"] = {"current_error": actual, "reference_error": expected, "ratio": actual / expected if expected > 0 else None}
    return {
        "source": "supplied HKLE slabs; no fresh Mantid run; native numerical viewer services executed",
        "assumptions": "independent diagonal variances; exactly known exposure; empty-cell removal only diagnoses unsubtracted event histograms",
        "shape": list(shape), "roi": list(roi), "normalization_storage": normalization_storage,
        "event_statistics_contract": bool(event_statistics),
        "viewer_has_normalization_channel": "normalization_denominator" in base,
        "edge_max_abs_differences": edge_differences,
        "hidden_axis_ranges": {
            str(dim): {"edges": nfit[f"edges{dim}"], "selected_center_range": [(nfit[f"edges{dim}"][0] + nfit[f"edges{dim}"][1]) / 2, (nfit[f"edges{dim}"][-2] + nfit[f"edges{dim}"][-1]) / 2]}
            for dim in (1, 2)
        },
        "display_axis_edges": {"h": nfit["edges0"], "energy": nfit["edges3"]},
        "provenance": {label: {"path": str(path), "sha256": _sha256(path)} for label, path in (("nfit", nfit_path), ("mantid", mantid_path), ("mantid_data", mantid_data_path))},
        "coverage_disagreement": {
            "nfit_covered_mantid_uncovered_pixels": int(np.count_nonzero((geometry[0] > 0) & (geometry[1] == 0))),
            "mantid_covered_nfit_uncovered_pixels": int(np.count_nonzero((geometry[1] > 0) & (geometry[0] == 0))),
        },
        "fine_cell_coverage_stats": coverage_stats,
        "region_definitions": "fringe: two-pixel Chebyshev neighborhood of uncovered map cells, no crop exterior; low exposure: per-row bottom decile with ties; interior excludes both",
        "path_semantics": {
            "hidden_axes": "actual native exposure pooling over both supplied hidden-axis extents",
            "regular_and_rotated_cuts": "actual pure inverse-variance means; nonpositive errors are excluded, including measured zero-error cells",
            "display_coarsening": (
                "actual C/V/N pooling over 2x native H and E steps"
                if event_statistics else "actual inverse-variance means over 2x native H and E steps"
            ),
            "roi_sum": "actual sum of normalized map intensities and diagonal errors; not a count numerator/exposure pooled estimate or bin-width integral",
            "floor_removed_diagnostic": "zero inferred variance only where unsubtracted source event count is zero; not a generic production correction",
        },
        "region_masks": region_masks, "regions": regions, "maps": maps,
        "energy_profiles": {"analytic": cut, "native": profiles}, "display_coarsened": coarse,
        "stage_error_ratios_current_over_reference_same_exposure": stage_ratios,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nfit-slab", required=True, type=Path)
    parser.add_argument("--mantid-slab", required=True, type=Path)
    parser.add_argument("--mantid-data", required=True, type=Path)
    parser.add_argument("--normalization-storage", choices=("metadata", "auxiliary"), default="metadata")
    parser.add_argument("--event-statistics", action="store_true", help="Replay the explicit C/V/N contract of nfit 0.106+ event histograms; omit for historical slabs")
    parser.add_argument("--roi", default="0.13,0.23,3.5,25", help="Hmin,Hmax,Emin,Emax")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        roi = tuple(float(item) for item in args.roi.split(","))
        report = diagnose_paths(args.nfit_slab, args.mantid_slab, args.mantid_data, roi=roi, normalization_storage=args.normalization_storage, event_statistics=args.event_statistics)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    rendered = json.dumps(_json_safe(report), indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
