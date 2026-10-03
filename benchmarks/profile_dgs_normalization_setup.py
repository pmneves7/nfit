"""Read-only metadata setup timing with cached and uncached saved-mask parsing.

Reads real MDE experiment metadata only, never event arrays. The exact-text
parse cache is disabled for alternating reference calls; every detector and run
payload is compared literally. Saves only scalar timings and provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from nfit import mdevent_dataset_group
from nfit import mdevent_detector_masks as masks
from nfit.mdevent import _trajectory_payloads


def assert_equal(first, second):
    """Compare the complete nested detector/run payload, preserving ordering."""
    if isinstance(first, (tuple, list)):
        assert type(first) is type(second)
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            assert_equal(left, right)
    elif isinstance(first, np.ndarray):
        assert first.dtype == second.dtype
        assert np.array_equal(first, second)
    else:
        assert first == second


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    original = (args.source.stat().st_size, args.source.stat().st_mtime_ns)
    cached = masks._cached_mask_ids
    cached.cache_clear()
    started = time.perf_counter()
    group = mdevent_dataset_group(args.source)
    inspection = time.perf_counter()-started
    cache_after_inspection = cached.cache_info()._asdict()
    results = []
    try:
        for repeat in range(args.repeat):
            order = ("cached", "uncached") if repeat % 2 == 0 else ("uncached", "cached")
            payloads = {}
            seconds = {}
            for mode in order:
                masks._cached_mask_ids = cached if mode == "cached" else masks._parse_mask_ids
                started = time.perf_counter()
                payloads[mode] = _trajectory_payloads(group, group.datasets, np.eye(4))
                seconds[mode] = time.perf_counter()-started
            assert_equal(payloads["cached"], payloads["uncached"])
            results.append({"repeat": repeat, "order": order, **seconds,
                            "all_payloads_literal": True})
            print(json.dumps(results[-1]), flush=True)
    finally:
        masks._cached_mask_ids = cached
    assert original == (args.source.stat().st_size, args.source.stat().st_mtime_ns)
    cached_median = statistics.median(item["cached"] for item in results)
    uncached_median = statistics.median(item["uncached"] for item in results)
    receipt = {
        "source": str(args.source), "source_size_bytes": original[0],
        "source_mtime_ns": original[1], "original_input_unchanged": True,
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "nfit_module": __import__("nfit").__file__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mask_module_sha256": hashlib.sha256(Path(masks.__file__).read_bytes()).hexdigest(),
        "run_count": len(group.datasets), "unique_geometry_payloads": len(payloads["cached"][0]),
        "inspection_seconds": inspection, "cache_after_inspection": cache_after_inspection,
        "cache_after_setup": cached.cache_info()._asdict(), "results": results,
        "cached_median_seconds": cached_median, "uncached_median_seconds": uncached_median,
        "setup_speedup": uncached_median/cached_median,
        "scope": "Metadata setup only; no event reads, trajectory integration or histogram allocation",
    }
    args.output.write_text(json.dumps(receipt, indent=2)+"\n")


if __name__ == "__main__":
    main()
