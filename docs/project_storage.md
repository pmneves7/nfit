# Project storage

A project keeps its settings, saved plots, analysis outputs, reduced events and
optional cached histograms in one `.nfit` file. Numerical artifacts remain lazy:
opening project metadata does not decode its event or histogram arrays.
Storage changes do not alter reduction, binning, precision or uncertainty rules.

## Incremental storage prototype

The default for new projects remains the existing ZIP format. An experimental
single-file format is available through Python for testing incremental saves:

```python
from nfit import load_project, save_project, project_storage_info

project = load_project("original.nfit")
save_project(project, "incremental-test.nfit", storage_mode="incremental")
print(project_storage_info("incremental-test.nfit"))

# Subsequent GUI or script saves preserve the existing destination format.
save_project(project, "incremental-test.nfit")
```

Use a **new destination** when converting a legacy project. The original remains
unchanged. Save As copies the required artifact closure and rebases declared
source paths through the same public project API as ordinary saving. External
raw files and calibrations remain references unless already embedded.

An incremental save appends new or changed artifacts, a new `project.json`, and
a complete live-member index. Unchanged artifacts keep their existing bytes and
file offsets; they are neither decoded, recompressed nor copied. The numerical
NPZ representation and compression settings are identical to legacy projects.
This helps repeated saves after metadata edits, adding runs, or changing some
binnings. The **first save still encodes every unsaved histogram** and copies any
imported reduced-event caches. Its cost must be measured separately.

## Publication, readers and recovery

A checksummed header identifies the file. Two fixed commit records select
complete generations and validate their ZIP index trailers. Saving flushes
the new body before publishing the next commit record. nfit readers ignore an
incomplete appended tail, without scanning it. Already-open artifact streams
retain their original descriptor and committed extent, including during
compaction. Lazy references remain usable when their artifact is unchanged,
including after unrelated analysis saves or cancellation. Replaced artifacts
invalidate their references; deferred copies cannot silently substitute newer
scientific data.

A nonblocking writer lock excludes cooperating writers. A revision check rejects
changes made while a save is being prepared. This protects overlapping saves;
it does not merge edits from two project windows. The GUI retains its existing
external-change review. Low-level callers may pass an `expected_revision` to
`nfit.project_archive.write_project_manifest` for an explicit conflict guard.
That function returns the committed revision for incremental storage. Project
saving checks this receipt before adopting cache backings: a newer commit
requires reopening and cannot silently replace the just-saved scientific data.

Cancellation is honored before publication. If synchronization fails after
publication starts, the result is reported as uncertain: **reopen the file before
retrying**. The writer does not truncate potentially published work.

If a commit record or its index is corrupt but an older valid generation exists,
`project_storage_info(path)["recovered"]` is true. Export that recovered project
to a new destination; in-place saving and compaction are refused. Header
corruption or loss of both valid commits fails closed. These checks detect
storage corruption; they do not authenticate a project supplied by another party.

Ordinary copies of a clean, closed file are portable and need no permanent
sidecar. Use nfit's reader for incremental files and recovery; generic ZIP tools
and older nfit releases do not implement the commit protocol. Do not copy a
file while it is being written. POSIX interruption and competing-process checks
are automated. Native Windows and cross-node shared-filesystem locking still
need validation before this becomes the default.

## Reclaiming replaced data

Replaced and deleted objects remain physically present until explicit
compaction. They are absent from the committed live-member index and are not
loaded. Ordinary saves never compact automatically.

```python
from nfit import compact_project

# Close project windows, compact, then reopen the project.
compact_project("incremental-test.nfit")
```

Compaction copies the live closure into a new single file and publishes it
atomically under the writer lock. It needs free space for a second copy, retains
sharing permissions, and gives the replacement a new file identity. Reopen
project windows afterwards so their lazy cache references use that identity.

Converting back to ordinary ZIP also uses a new destination:

```python
project = load_project("incremental-test.nfit")
save_project(project, "portable-legacy.nfit", storage_mode="legacy")
```

GUI conversion, compaction and recovery presentation remain planned. Existing
projects are not migrated automatically.

## Measured scope

On an ORNL analysis node, a metadata-only save of a saved 60 GB HYSPEC project
took 79.5–80.5 seconds with a ZIP rewrite and 1.30–1.41 seconds with incremental
storage. Initial copies took about 79–80 seconds with either format. These
measurements include manifest encoding and durable publication. They exclude
first-time histogram encoding, reduction and binning.

The benchmark script is `benchmarks/benchmark_project_storage.py`; the detailed
comparison and remaining SQLite evaluation are recorded in
`benchmarks/results/project-storage-node21.md` in the source repository.
