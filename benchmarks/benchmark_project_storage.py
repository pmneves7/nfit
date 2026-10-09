"""Measure opaque project persistence; never reduce, rebin, or change the source.

Run with nfit's Python runtime. All temporary scientific files live under the
explicit output directory and are removed after the measurements. Timings
include manifest encoding, artifact copying and durability synchronization;
they exclude initial histogram encoding and scientific reconstruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import tempfile
import time
import zlib
from dataclasses import asdict
from pathlib import Path

from nfit import project_archive as api
from nfit.project_store import (
    StoreBusyError,
    inspect_project_store,
    open_project_zip,
    reuse_project_members,
    write_incremental,
)

BUFFER = 8 * 1024**2
MANIFEST_COMPRESSION_LEVEL = 6  # Match ZipFile's default DEFLATE setting.


def timed(callback):
    start, cpu = time.monotonic(), time.process_time()
    result = callback()
    return result, {"seconds": time.monotonic() - start, "cpu_seconds": time.process_time() - cpu}


def digest(stream):
    result = hashlib.sha256()
    while block := stream.read(BUFFER):
        result.update(block)
    return result.hexdigest()


def sqlite_create(path, source, *, page_bytes=65536):
    # Comparator only: opaque immutable NPZ blobs, rollback journal, no WAL
    # sidecars needed for a closed portable copy. No SQLite runtime dependency.
    with sqlite3.connect(path) as connection, open_project_zip(source) as archive:
        connection.execute(f"PRAGMA page_size={int(page_bytes)}")
        connection.execute("PRAGMA cache_size=-65536")
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE members(name TEXT UNIQUE, data BLOB)")
        for info in archive.infolist():
            if info.filename == api.PROJECT_MANIFEST:
                data = zlib.compress(archive.read(info), MANIFEST_COMPRESSION_LEVEL)
                connection.execute("INSERT INTO members VALUES (?,?)", (info.filename, data))
            else:
                cursor = connection.execute("INSERT INTO members VALUES (?,zeroblob(?))", (info.filename, info.file_size))
                with connection.blobopen("members", "data", cursor.lastrowid) as blob, archive.open(info) as stream:
                    while block := stream.read(BUFFER):
                        blob.write(block)
        connection.commit()


def filesystem_safety(directory):
    """Exercise process locks, commit interruption and pinned readers on this FS."""
    from nfit import project_store as store

    path = directory / "safety.nfit"
    api.write_project_manifest(path, {"value": 1}, storage_mode="incremental")
    results = {}
    if not hasattr(os, "fork"):
        return {"skipped": "Native POSIX process tests unavailable"}
    for stage, expected in (("after_body_fsync", 1), ("after_slot_fsync", 2)):
        child = os.fork()
        if child == 0:
            store._checkpoint = lambda point, boundary=stage: os._exit(71) if point == boundary else None
            api.write_project_manifest(path, {"value": 2})
            os._exit(72)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 71
        assert api.read_project_manifest(path)["value"] == expected
        results[stage] = "passed"
    read_ready, write_ready = os.pipe()
    read_release, write_release = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_ready)
        os.close(write_release)

        def hold(destination, current):
            reuse_project_members(destination, current)
            os.write(write_ready, b"ready")
            os.read(read_release, 1)

        write_incremental(path, hold)
        os._exit(0)
    os.close(write_ready)
    os.close(read_release)
    try:
        assert os.read(read_ready, 5) == b"ready"
        try:
            write_incremental(path, lambda *_: None)
        except StoreBusyError:
            results["competing_writers"] = "passed"
        else:
            raise AssertionError("Concurrent writer was not excluded")
    finally:
        os.write(write_release, b"x")
        os.close(read_ready)
        os.close(write_release)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
    with open_project_zip(path) as old:
        api.write_project_manifest(path, {"value": 3})
        assert json.loads(old.read(api.PROJECT_MANIFEST))["value"] == 2
    results["pinned_reader"] = "passed"
    return results


def benchmark(source, output, *, rounds, compare_sqlite):
    output.mkdir(parents=True, exist_ok=True)
    report = {"host": platform.node(), "logical_cpus": os.cpu_count(), "source": source.name,
              "source_bytes": source.stat().st_size, "scope": __doc__.strip(), "rounds": rounds,
              "manifest_compression_level": MANIFEST_COMPRESSION_LEVEL}
    with tempfile.TemporaryDirectory(prefix="project-store-", dir=output) as temporary:
        directory = Path(temporary)
        report["filesystem_safety"] = filesystem_safety(directory)
        manifest = api.read_project_manifest(source)
        original_copy = api._copy_archive_stream
        copied = 0

        def counted(src, dst):
            nonlocal copied
            while block := src.read(BUFFER):
                copied += len(block)
                dst.write(block)

        api._copy_archive_stream = counted
        try:
            targets = {mode: directory / f"{mode}.nfit" for mode in ("legacy", "incremental")}
            report["create"] = {}
            for mode, target in targets.items():
                copied = 0
                _, result = timed(lambda target=target, mode=mode: api.write_project_manifest(target, manifest, asset_source=source, storage_mode=mode))
                result.update(file_bytes=target.stat().st_size, copied_payload_bytes=copied)
                report["create"][mode] = result
                print("CREATE", mode, json.dumps(result), flush=True)
            report["saves"] = {mode: [] for mode in targets}
            for iteration in range(rounds):
                settings = {**manifest, "storage_benchmark_iteration": iteration}
                for mode, target in targets.items():
                    copied = 0
                    before = target.stat().st_size
                    _, result = timed(lambda target=target, settings=settings: api.write_project_manifest(target, settings))
                    result.update(copied_payload_bytes=copied, file_growth_bytes=target.stat().st_size - before)
                    report["saves"][mode].append(result)
                    print("SAVE", mode, json.dumps(result), flush=True)
            representative = next(name for name in open_names(source) if name.startswith("assets/binnings/"))
            report["artifact_sha256"] = {}
            for mode, target in {"source": source, **targets}.items():
                with api.open_project_artifact(target, representative) as stream:
                    report["artifact_sha256"][mode] = digest(stream)
            assert len(set(report["artifact_sha256"].values())) == 1
            for mode, target in targets.items():
                _, report.setdefault("manifest_reopen", {})[mode] = timed(lambda target=target: api.read_project_manifest(target))
            report["incremental_revision"] = asdict(inspect_project_store(targets["incremental"]).revision)
            if compare_sqlite:
                database = directory / "sqlite.nfit"
                _, report["sqlite_create"] = timed(lambda: sqlite_create(database, source))
                report["sqlite_create"]["file_bytes"] = database.stat().st_size
                report["sqlite_create"]["page_bytes"] = 65536
                report["sqlite_saves"] = []
                for iteration in range(rounds):
                    settings = {**manifest, "storage_benchmark_iteration": iteration}

                    def update(settings=settings):
                        encoded = zlib.compress((json.dumps(settings, indent=2, sort_keys=True) + "\n").encode(), MANIFEST_COMPRESSION_LEVEL)
                        with sqlite3.connect(database) as connection:
                            connection.execute("PRAGMA synchronous=FULL")
                            connection.execute("UPDATE members SET data=? WHERE name=?", (encoded, api.PROJECT_MANIFEST))
                            connection.commit()

                    _, result = timed(update)
                    report["sqlite_saves"].append(result)
                    print("SAVE sqlite", json.dumps(result), flush=True)
                with sqlite3.connect(database) as connection:
                    row = connection.execute("SELECT rowid FROM members WHERE name=?", (representative,)).fetchone()
                    with connection.blobopen("members", "data", row[0], readonly=True) as blob:
                        assert digest(blob) == report["artifact_sha256"]["source"]
        finally:
            api._copy_archive_stream = original_copy
    report["temporary_projects_removed"] = True
    (output / "storage-benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT", json.dumps(report), flush=True)


def open_names(path):
    with open_project_zip(path) as archive:
        return archive.namelist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--sqlite", action="store_true")
    arguments = parser.parse_args()
    benchmark(arguments.source, arguments.output, rounds=arguments.rounds, compare_sqlite=arguments.sqlite)
