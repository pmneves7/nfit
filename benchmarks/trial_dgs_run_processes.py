"""Manual whole-run process trial using the existing desktop executable.

Separate runtime processes avoid h5py's interpreter-wide lock. Manifests and
completed per-run caches live beside the benchmark project; only disk references
are returned. This is deliberately not an application worker implementation.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path


def _plain(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported manifest value: {type(value).__name__}")


def stop_owned_jobs(jobs):
    """Join owned POSIX process groups, including the time wrapper's child."""
    for job in jobs:
        if job.poll() != 0:
            try:
                os.killpg(job.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            job.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(job.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            job.wait()


def process_producer_binner(reference, *, workers):
    from nfit.raw_dgs_cache import _ReducedEventCache, cached_reduction, reduction_signature
    from nfit.reduction_recipes import effective_reduction_config

    def bin_group(group, **kwargs):
        candidates = kwargs.get("datasets")
        selected = [dataset for dataset in (group.datasets if candidates is None else candidates)
                    if dataset.enabled and dataset.fit_weight > 0.]
        missing = []
        for dataset in selected:
            config = effective_reduction_config(group, dataset)
            if not config.get("cache_reduced_events", True):
                return reference(group, **kwargs)
            signature = reduction_signature(dataset, config)
            if cached_reduction(dataset, signature) is None:
                missing.append((dataset, signature))
        if not missing:
            return reference(group, **kwargs)
        owner = missing[0][0].metadata.get("_project_path")
        if owner is None:
            raise ValueError("Process trials require an explicit benchmark project owner")
        root = Path(owner).parent / "parallel-run-producers"
        root.mkdir(exist_ok=False)
        jobs = []
        records = []
        try:
            for index in range(min(workers, len(missing))):
                assigned = missing[index::workers]
                directory = root / f"worker-{index}"
                directory.mkdir()
                manifest = directory / "manifest.json"
                manifest.write_text(json.dumps(dict(
                    group_metadata=group.metadata,
                    datasets=[dict(name=d.name, kind=d.kind, data_type=d.data_type,
                                   metadata=d.metadata, id=d.id) for d, _ in assigned],
                ), default=_plain) + "\n")
                environment = dict(os.environ, NFIT_DGS_RUN_MANIFEST=str(manifest),
                                   TMPDIR=str(directory), MPLCONFIGDIR=str(directory / "matplotlib"),
                                   XDG_CACHE_HOME=str(directory / "cache"))
                with (directory / "worker.log").open("x") as log:
                    job = subprocess.Popen(
                        ["/usr/bin/time", "-v", "-o", str(directory / "resources.txt"),
                         "/SNS/users/paulneves/bin/nfit", "--run-script", str(Path(__file__).resolve())],
                        env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                jobs.append(job)
                records.append(directory)
            for job, directory in zip(jobs, records, strict=True):
                code = job.wait()
                if code:
                    raise RuntimeError(f"Run producer failed ({code}); inspect {directory / 'worker.log'}")
            completed = {}
            for directory in records:
                for record in json.loads((directory / "complete.json").read_text()):
                    completed[record["dataset_id"]] = record
            # Publish only after every worker succeeded and every signature was
            # checked. Original benchmark datasets are changed on this thread.
            for dataset, signature in missing:
                if completed[dataset.id]["signature"] != signature:
                    raise ValueError("Producer returned a different reduction signature")
                if reduction_signature(dataset, effective_reduction_config(group, dataset)) != signature:
                    raise ValueError("Scientific inputs changed during run cache production")
            for dataset, signature in missing:
                record = completed[dataset.id]
                path = Path(record["path"])
                dataset._raw_dgs_reduction_cache = _ReducedEventCache(
                    signature, path, disk_bytes=path.stat().st_size)
                dataset.metadata["raw_dgs_reduction_cache"] = record["metadata"]
        finally:
            stop_owned_jobs(jobs)
        return reference(group, **kwargs)
    return bin_group


def worker_main():
    from trial_dgs_raw_pipeline import run_producer_binner

    from nfit.pipeline import DatasetEntry, DatasetGroup

    path = Path(os.environ["NFIT_DGS_RUN_MANIFEST"])
    manifest = json.loads(path.read_text())
    datasets = [DatasetEntry(data=None, **item) for item in manifest["datasets"]]
    group = DatasetGroup("Private run producer", datasets=datasets, metadata=manifest["group_metadata"])
    # Use the same producer arithmetic, geometry checks and cache writer as the
    # thread experiment. No histogram is built by this child.
    run_producer_binner(lambda *args, **kwargs: None, workers=1)(group)
    records = []
    for dataset in datasets:
        cache = dataset._raw_dgs_reduction_cache
        target = path.parent / f"{dataset.id}.npz"
        os.replace(cache.content, target)
        records.append(dict(dataset_id=dataset.id, path=str(target), signature=cache.signature,
                            metadata=dataset.metadata["raw_dgs_reduction_cache"]))
    (path.parent / "complete.json").write_text(json.dumps(records) + "\n")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    worker_main()
