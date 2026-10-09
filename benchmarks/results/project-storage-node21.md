# Project persistence on ORNL node21

## Scope and source

Measured on 2026-10-09 on `analysis-node21.sns.gov` (64 logical CPUs), using
the saved `HYSPEC_all_raw.nfit` project from HYS IPTS-36860. Its file size was
59,971,212,456 bytes. The source project was read-only throughout these trials.
Scientific staging and reports stayed in the experiment's
`shared/nfit/benchmarks/storage-prototype-20261009` directory. Temporary project
copies were removed after each trial.

The benchmark includes JSON encoding, opaque artifact copying, index generation
and durability synchronization. It does **not** reconstruct events, rebin data
or encode newly generated histogram arrays. The live node23 first-save slowdown
is a separate investigation; these results do not establish its cause or fix it.

## ZIP versus incremental ZIP generations

| Operation | Legacy ZIP rewrite | Incremental prototype |
| --- | ---: | ---: |
| Initial copy of the saved project | 79.98 s | 79.03 s |
| Metadata save, trial 1 | 79.51 s | 1.41 s |
| Metadata save, trial 2 | 80.52 s | 1.30 s |
| Manifest reopening | 0.48 s | 0.58 s |
| Scientific payload copied per metadata save | 59,965,352,859 bytes | 0 bytes |

Repeated saves were approximately 59 times faster on average. Each incremental
save appended 5,227,036 bytes of manifest and member-index data. Existing event
and histogram bytes were retained at their original offsets. A representative
histogram's streamed digest agreed between source and both copies; copying also
validated source ZIP CRCs.

On the same shared filesystem, process tests passed for interruption after body
sync, interruption after commit sync, competing writer exclusion and an already
open reader retaining its old generation. Cross-node locking and native Windows
behavior have not been certified. A closed-file copy and compaction are covered
by automated local tests.

## SQLite comparator

SQLite was tested with opaque NPZ blobs, rollback journaling (`DELETE`),
`synchronous=FULL`, and no permanent WAL sidecar. It is a benchmark comparator,
not an implemented project backend.

| Configuration | Initial copy | Metadata save 1 | Metadata save 2 |
| --- | ---: | ---: | ---: |
| Default 4 KiB pages | 588.96 s | 0.64 s | 0.58 s |
| 64 KiB pages, 64 MiB page cache | 59.16 s | 0.63 s | 0.57 s |
| 64 KiB pages, matching level 6 compression | 59.49 s | 0.91 s | 0.86 s |

The first two SQLite metadata trials used zlib level 1; ZIP used level 6. They
show that SQLite tuning matters and that updating metadata need not rewrite
scientific blobs. The final row repeats the comparison with matching level 6
compression and durable commit settings. SQLite remains competitive after that
correction. These trials still compare persistence, not the performance of a
production chunk reader. The benchmark script now uses level 6 for both.

## Storage decision and remaining work

The append implementation remains opt-in. New projects continue to use ordinary
ZIP, and existing projects are not automatically migrated. The prototype keeps
the current numerical formats and direct bounded NPZ reads while eliminating
the repeated whole-project copy.

SQLite remains a strong candidate for the eventual default: it delegates
transaction recovery to an established database engine. The tuned initial-copy
result is competitive. A production implementation would need immutable chunks
for large artifacts and a seekable reader, rather than the comparator's single
blob per member. Short read transactions and retained object lifetimes must
allow simultaneous saves and viewers without copying whole cubes into RAM.

Before choosing the default:

1. Retain matching compression and durability settings in further comparisons.
2. Measure reduced-event iteration and histogram loading with multiple viewers.
3. Validate cross-node writer exclusion, interrupted writes and reader lifetimes.
4. Test closed-file portability and coordinated compaction.
5. Trace the old live GUI's first histogram encoding separately.

If SQLite matches the read workloads without material overhead or reader
complexity, its established transaction machinery makes it preferable. The
append prototype remains useful evidence and a working incremental path. This
checkpoint does not change reduction, binning, precision or uncertainty rules.
