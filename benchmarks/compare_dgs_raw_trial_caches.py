"""Stream saved reduced-event caches for a native trial/baseline comparison.

No reduction engine is loaded. Match runs by source identity, not dataset UUID.
Require identical reduction signatures, headers, detector exposure payloads,
event order and raw totals. Allow only 1e-12 coordinate roundoff in Å⁻¹/meV;
corrected weights and variance remain literal. Histogram membership is checked
separately by benchmark_dgs_compare.py.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np


def _entries(manifest):
    entries = {}

    def visit(value):
        if isinstance(value, dict):
            metadata = value.get("metadata", {})
            cache = metadata.get("raw_dgs_reduction_cache")
            if cache is not None and "source_file" in metadata:
                source = metadata["source_file"]
                if source in entries:
                    raise ValueError("Repeated source in the trial project")
                entries[source] = cache
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(manifest)
    if not entries:
        raise ValueError("No reduced-event references in the project")
    return entries


def compare(actual_path, reference_path):
    raw_total = accepted_total = blocks = 0
    max_coordinate_delta = 0.
    with zipfile.ZipFile(actual_path) as actual, zipfile.ZipFile(reference_path) as reference:
        aa = _entries(json.loads(actual.read("project.json")))
        bb = _entries(json.loads(reference.read("project.json")))
        if aa.keys() != bb.keys():
            raise ValueError("Projects have different source membership")
        for source in aa:
            if aa[source]["signature"] != bb[source]["signature"]:
                raise ValueError(f"Reduction signature differs: {source}")
            with actual.open(aa[source]["member"]) as astream, reference.open(bb[source]["member"]) as bstream:
                with np.load(astream, allow_pickle=False) as a, np.load(bstream, allow_pickle=False) as b:
                    ah = json.loads(str(a["header_json"].item()))
                    bh = json.loads(str(b["header_json"].item()))
                    if ah != bh:
                        raise ValueError(f"Reduced-event header differs: {source}")
                    for key in ("detector_ids", "direction", "solid", "charge"):
                        np.testing.assert_array_equal(a[key], b[key])
                    ar = br = 0
                    for index in range(ah["chunk_count"]):
                        ae, be = a[f"events_{index}"], b[f"events_{index}"]
                        if ae.shape != be.shape:
                            raise ValueError(f"Accepted-event membership differs: {source}")
                        np.testing.assert_allclose(ae[:, :4], be[:, :4], rtol=1e-12, atol=1e-12)
                        np.testing.assert_array_equal(ae[:, 4:], be[:, 4:])
                        max_coordinate_delta = max(max_coordinate_delta,
                            float(np.max(np.abs(ae[:, :4] - be[:, :4]), initial=0.)))
                        accepted_total += len(ae)
                        blocks += 1
                        ar += int(a[f"raw_count_{index}"])
                        br += int(b[f"raw_count_{index}"])
                    if ar != br:
                        raise ValueError(f"Raw-event totals differ: {source}")
                    raw_total += ar
    return dict(status="passed", runs=len(aa), blocks=blocks, raw_events=raw_total,
                accepted_events=accepted_total, max_coordinate_absolute_difference=max_coordinate_delta,
                coordinate_gate="rtol=1e-12; atol=1e-12 in Q_lab (Å⁻¹) and DeltaE (meV)",
                corrected_weights_and_variance="literal equality",
                headers_signatures_and_detector_exposure="literal equality")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(args.actual, args.reference)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
