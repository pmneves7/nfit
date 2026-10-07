"""Compare lossless histogram writing and saving an existing session NPZ.

Supply a diagnostic HDF5 histogram containing signal, errors, mask,
num_events, N (normalization denominator), and edge0 through edge3. All
scientific outputs stay in the explicitly selected output directory; choose
an IPTS directory on ORNL. The input histogram and user projects are read-only.

The baseline writer is an older src/nfit/array_archive.py snapshot; relative
imports resolve against the current package. This manual benchmark uses the
current project publisher for both candidates, includes fsync, and checks the
numerical bytes outside the timed intervals. It does not perform reduction.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np

from nfit._parallel import thread_budget
from nfit.analysis import artifacts
from nfit.mdhisto import MDHistoAxis, MDHistoData
from nfit.project_archive import write_project_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histogram", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-writer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = h5py.File(args.histogram, "r")
    arrays = {name: source[name][...] for name in ("signal", "errors", "mask", "num_events", "N")}
    for array in arrays.values():
        array.setflags(write=False)
    axes = tuple(
        MDHistoAxis(
            f"axis{i}",
            source[f"edge{i}"][...],
            "meV" if i == 3 else "rlu",
            "energy" if i == 3 else "momentum",
        )
        for i in range(4)
    )
    source.close()
    data = MDHistoData(
        axes,
        arrays["signal"],
        arrays["errors"],
        arrays["mask"],
        arrays["num_events"],
        metadata={"normalization_denominator": arrays["N"]},
    )
    rows = []
    original = artifacts.write_array_archive
    spec = importlib.util.spec_from_file_location("nfit._trial_array_archive", args.baseline_writer)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    with thread_budget(args.workers):
        for label, writer in (("baseline", baseline.write_array_archive), ("candidate", original)):
            artifacts.write_array_archive = writer
            for repeat in range(2):
                path = out / f"{label}.npz"
                t = time.perf_counter()
                cpu = time.process_time()
                artifacts.write_dataset_artifact(data, path)
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                row = dict(
                    stage="first_histogram_write",
                    label=label,
                    repeat=repeat,
                    seconds=time.perf_counter() - t,
                    cpu_seconds=time.process_time() - cpu,
                    bytes=path.stat().st_size,
                )
                rows.append(row)
                print(json.dumps(row), flush=True)
            # Validate every byte of each numerical payload outside measured write.
            with np.load(path) as decoded:
                for key, value in artifacts._payload(data).items():
                    actual = decoded[key]
                    assert actual.dtype == np.asanyarray(value).dtype
                    assert actual.shape == np.asanyarray(value).shape
                    assert actual.tobytes() == np.asanyarray(value).tobytes()
        artifacts.write_array_archive = original
        # Old first-save path decodes and recompresses an existing spill file.
        manifest = {"format": "nfit-project", "version": 4, "settings": {}, "data_groups": []}
        member = "assets/binnings/diagnostic/data.npz"
        for label in ("baseline", "candidate"):
            t = time.perf_counter()
            cpu = time.process_time()
            if label == "baseline":
                decoded = artifacts.read_dataset_artifact(out / "baseline.npz")
                cache = out / "recompressed.npz"
                artifacts.write_array_archive = baseline.write_array_archive
                artifacts.write_dataset_artifact(decoded, cache)
                artifacts.write_array_archive = original
                del decoded
            else:
                cache = out / "baseline.npz"
            destination = out / f"{label}.nfit"
            write_project_manifest(destination, manifest, binning_artifacts={member: cache})
            row = dict(
                stage="save_existing_spill",
                label=label,
                seconds=time.perf_counter() - t,
                cpu_seconds=time.process_time() - cpu,
                bytes=destination.stat().st_size,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        with (
            zipfile.ZipFile(out / "candidate.nfit") as archive,
            (out / "baseline.npz").open("rb") as expected,
            archive.open(member) as actual,
        ):

            def digest(stream):
                h = hashlib.sha256()
                while block := stream.read(4 * 1024**2):
                    h.update(block)
                return h.hexdigest()

            assert digest(expected) == digest(actual)
    (out / "workflow-receipt.json").write_text(
        json.dumps(
            dict(
                host=platform.node(),
                workers=args.workers,
                shape=list(data.shape),
                expanded_array_bytes=sum(x.nbytes for x in arrays.values()),
                lossless=True,
                rows=rows,
            ),
            indent=2,
        )
    )
    for name in (
        "baseline.npz",
        "candidate.npz",
        "recompressed.npz",
        "baseline.nfit",
        "candidate.nfit",
    ):
        (out / name).unlink(missing_ok=True)
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
