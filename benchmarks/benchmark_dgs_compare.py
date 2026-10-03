"""Compare untimed DGS diagnostic arrays without loading either engine.

Run on the cluster alongside the benchmark outputs. Reports original cells and
the NiO low-coverage HHH/energy cut. Numerical summaries are not timed workflow
operations. The strict native gate compares a candidate with its baseline;
Mantid comparisons report discrepancies rather than hiding boundary differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import socket
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_dilation

CHANNELS = ("numerator", "variance", "normalization", "counts")


def _summary(actual, reference):
    finite = np.isfinite(actual) & np.isfinite(reference)
    a, b = actual[finite], reference[finite]
    delta = a - b
    nonzero = b != 0
    peak = float(np.max(np.abs(b), initial=0.))
    denominator = float(np.sum(b*b))
    return dict(cells=int(a.size), different_cells=int(np.count_nonzero(delta)),
                max_absolute=float(np.max(np.abs(delta), initial=0.)),
                max_relative_nonzero=float(np.max(np.abs(delta[nonzero]/b[nonzero]), initial=0.)),
                max_relative_to_peak=float(np.max(np.abs(delta), initial=0.)/peak) if peak else None,
                relative_l2=float(np.sqrt(np.sum(delta*delta)/denominator)) if denominator else None,
                actual_sum=float(np.sum(a)), reference_sum=float(np.sum(b)),
                finite_support_equal=bool(np.array_equal(np.isfinite(actual), np.isfinite(reference))))


def _whole_cells(actual, reference, *, require_native_parity, block_cells=1_000_000):
    shape = actual[CHANNELS[0]].shape
    if shape != reference[CHANNELS[0]].shape:
        raise ValueError("Compared histograms have different shapes")
    if any(actual[key].shape != shape or reference[key].shape != shape for key in CHANNELS):
        raise ValueError("Diagnostic channel shapes differ")
    per_plane = int(np.prod(shape[1:]))
    width = max(1, block_cells//per_plane)
    totals = {key: dict(cells=0, different_cells=0, max_absolute=0., max_relative_nonzero=0.,
                       actual_sum=0., reference_sum=0., finite_support_equal=True,
                       squared_difference=0., squared_reference=0., reference_peak=0.) for key in CHANNELS}
    support = dict(actual_exposed=0, reference_exposed=0, differing_exposure_support=0,
                   actual_events_without_exposure=0, reference_events_without_exposure=0)
    differences = []
    for start in range(0, shape[0], width):
        region = slice(start, min(start+width, shape[0]))
        a = {key: np.asarray(actual[key][region]) for key in CHANNELS}
        b = {key: np.asarray(reference[key][region]) for key in CHANNELS}
        an, bn = a["normalization"] > 0, b["normalization"] > 0
        support["actual_exposed"] += int(an.sum())
        support["reference_exposed"] += int(bn.sum())
        support["differing_exposure_support"] += int(np.count_nonzero(an != bn))
        support["actual_events_without_exposure"] += int(np.count_nonzero(~an & (a["counts"] > 0)))
        support["reference_events_without_exposure"] += int(np.count_nonzero(~bn & (b["counts"] > 0)))
        for key in CHANNELS:
            summary = _summary(a[key], b[key])
            total = totals[key]
            for name in ("cells", "different_cells", "actual_sum", "reference_sum"):
                total[name] += summary[name]
            for name in ("max_absolute", "max_relative_nonzero"):
                total[name] = max(total[name], summary[name])
            total["finite_support_equal"] &= summary["finite_support_equal"]
            finite = np.isfinite(a[key]) & np.isfinite(b[key])
            aa, bb = a[key][finite], b[key][finite]
            total["squared_difference"] += float(np.sum((aa-bb)**2))
            total["squared_reference"] += float(np.sum(bb**2))
            total["reference_peak"] = max(total["reference_peak"], float(np.max(abs(bb), initial=0.)))
            if require_native_parity:
                if key == "normalization":
                    np.testing.assert_allclose(a[key], b[key], rtol=1e-12, atol=0., equal_nan=True)
                else:
                    np.testing.assert_array_equal(a[key], b[key])
        mismatch = np.argwhere(a["counts"] != b["counts"])
        for index in mismatch[:max(0, 12-len(differences))]:
            point = tuple(int(i) for i in index)
            full = (point[0]+start, *point[1:])
            differences.append(dict(index=full, actual=float(a["counts"][point]),
                                    reference=float(b["counts"][point])))
        if require_native_parity:
            np.testing.assert_array_equal(an, bn)
    for total in totals.values():
        d = total.pop("squared_difference")
        r = total.pop("squared_reference")
        peak = total.pop("reference_peak")
        total["relative_l2"] = float(np.sqrt(d/r)) if r else None
        total["max_relative_to_peak"] = total["max_absolute"]/peak if peak else None
    return dict(shape=shape, channels=totals, exposure_support=support,
                first_count_differences=differences)


def _nio_cut(actual, reference):
    """Pool actual original cells in K=0±.03, L=.33±.02, without interpolation."""
    if len(actual["counts"].shape) != 4:
        return None
    ae = [np.asarray(actual[f"edges/{dim}"]) for dim in range(4)]
    be = [np.asarray(reference[f"edges/{dim}"]) for dim in range(4)]
    indices = []
    for dim, bounds in ((1, (-.03, .03)), (2, (.31, .35))):
        centers = .5*(ae[dim][:-1]+ae[dim][1:])
        selected = np.flatnonzero((centers >= bounds[0]) & (centers <= bounds[1]))
        if not len(selected):
            return None
        indices.append(slice(selected[0], selected[-1]+1))
    selection = (slice(None), *indices, slice(None))
    aa = {key: np.sum(actual[key][selection], axis=(1, 2)).T for key in CHANNELS}
    bb = {key: np.sum(reference[key][selection], axis=(1, 2)).T for key in CHANNELS}
    for pooled in (aa, bb):
        exposure = pooled["normalization"]
        for name, numerator in (("signal", pooled["numerator"]),
                                ("uncertainty", np.sqrt(pooled["variance"]))):
            pooled[name] = np.divide(numerator, exposure,
                                     out=np.full(exposure.shape, np.nan), where=exposure > 0)
    n = bb["normalization"]
    covered = n > 0
    fringe = covered & binary_dilation(~covered, iterations=2)
    low = np.zeros(n.shape, bool)
    for row in range(len(n)):
        valid = covered[row]
        if np.any(valid):
            low[row] = valid & (n[row] <= np.quantile(n[row, valid], .1))
    centers_h = .5*(be[0][:-1]+be[0][1:])
    centers_e = .5*(be[3][:-1]+be[3][1:])
    roi = covered & (centers_h[None, :] >= .13) & (centers_h[None, :] <= .23)
    roi &= (centers_e[:, None] >= 3.5) & (centers_e[:, None] <= 25.)
    return dict(hidden_cell_indices=[[int(s.start), int(s.stop)] for s in indices],
                target="pooled numerator/variance/exposure from original cells",
                exposure_support=dict(actual_exposed=int(np.count_nonzero(aa["normalization"] > 0)),
                                      reference_exposed=int(covered.sum()),
                                      differing_cells=int(np.count_nonzero((aa["normalization"] > 0) != covered))),
                regions={name: {key: _summary(aa[key][mask], bb[key][mask])
                                for key in (*CHANNELS, "signal", "uncertainty")}
                         for name, mask in (("all", np.ones(n.shape, bool)),
                                           ("coverage_fringe", fringe), ("row_lowest_exposure_decile", low),
                                           ("central_low_coverage_roi", roi))})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nio-cut", action="store_true")
    parser.add_argument("--require-native-parity", action="store_true")
    args = parser.parse_args()
    before = {str(path): [path.stat().st_size, path.stat().st_mtime_ns]
              for path in (args.actual, args.reference)}
    with h5py.File(args.actual, "r") as actual, h5py.File(args.reference, "r") as reference:
        edges = []
        for dim in range(len(actual["counts"].shape)):
            a = np.asarray(actual[f"edges/{dim}"])
            b = np.asarray(reference[f"edges/{dim}"])
            if a.shape != b.shape:
                raise ValueError("Compared edge arrays have different shapes")
            edges.append(_summary(a, b))
            if args.require_native_parity:
                np.testing.assert_array_equal(a, b)
        result = dict(host=socket.getfqdn(), actual=str(args.actual), reference=str(args.reference),
                      inputs=before, edges=edges,
                      native_candidate_parity_required=args.require_native_parity,
                      whole_cells=_whole_cells(actual, reference,
                                               require_native_parity=args.require_native_parity),
                      nio_cut=_nio_cut(actual, reference) if args.nio_cut else None,
                      timing_scope="untimed scientific diagnostics; excluded from workflow wall time",
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    after = {str(path): [path.stat().st_size, path.stat().st_mtime_ns]
             for path in (args.actual, args.reference)}
    if after != before:
        raise RuntimeError("A diagnostic input changed during comparison")
    result["original_inputs_unchanged"] = True
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
