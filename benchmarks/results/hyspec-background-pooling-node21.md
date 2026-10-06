# Pooling native HYSPEC background exposure

Measured October 6, 2026 on analysis-node21, sequentially with 16 workers,
100,000 MiB RAM allowance and 192 MiB batch allowance. Both variants use 90
evenly spaced sample runs from the native low-temperature bank-70 collection
(26,974,544 raw events), all 11 associated dummy files, eight symmetry
operations and the saved in-plane grid: 161 × 221 × 12 × 49 cells.
The original project was unchanged; separate diagnostic projects, arrays,
temporary storage and logs were kept inside the HYSPEC IPTS.

| Timed operation | Current version | Pooled exposure |
|---|---:|---:|
| Sample reading, reduction, cache and binning | 21.65 s | 21.11 s |
| Background preparation, reduction and replay | 140.29 s | 99.34 s |
| Project save, including reduced caches | 9.60 s | 9.05 s |
| Sum of these stages | 171.54 s | 129.50 s |
| Peak process memory | 4.96 GiB | 4.61 GiB |

The background stage is **29.2% faster**; the sum of timed stages is 24.5%
shorter. These are one-pass trials on a shared node, with uncontrolled
filesystem caches. Initial imports/project metadata loading and diagnostic HDF
writes are outside the stage totals. This is a matched optimization trial,
not a timing of the complete HYSPEC project or a Mantid comparison.

## Change and guards

The baseline already retained reduced events on original background entries.
It nevertheless integrated each file's detector trajectories separately.
Exposure is linear in proton charge, so compatible source files can share
trajectory integration. Original event replay and covariance recipes remain
separate: the event count times sample-angle count is unchanged.

A focused service collects bounded exposure recipes and passes them to the
existing trajectory reducer. Detector IDs, directions, normalization arrays,
shapes and dtypes must match literally; hashes only select candidates. The
existing task reducer then checks affine transforms, Ei and the effective
energy window before summing positive exposures. Output masks partition the
work. No instrument-name or nominal-bank-angle equivalence is assumed.
Recipe storage is limited to one eighth of the batch allowance, capped at
32 MiB, plus one source's preparation payload before the next flush. Existing
event scratch, output-grid and worker limits remain in force.

All 11 source entries retained reduced caches in both trials and saves. These
caches can be reused by other binnings or sample datasets referencing that same
background group. Separately imported copies still own separate caches.

## Numerical gate

Every one of the 20,921,628 cells was compared. Masks, counts, edges and exposure
support match literally, including coverage fringes. Maximum relative differences
are 2.34e-15 for exposure, 2.33e-15 for signal and 2.33e-15 for uncertainty,
passing the 1e-12 relative gate with zero absolute tolerance.

Unit and integration tests also cover mixed detector geometry, masks,
per-run incident energies, calibration/fit weights, both event precision
policies, compiled and NumPy paths, cancellation and bounded flushing.
Original-source covariance recipes are identical with pooled and separately
flushed exposure calculations. Forced hash collisions verify complete identity
checks. The benchmark preceded that final collision guard; the guard does not
alter its numerical inputs or arithmetic.

The [receipt](hyspec-background-pooling-node21.json) records source hashes,
configuration, memory, timings and numerical differences. The reproduction
entry point is `benchmarks/trial_dgs_background_workflow.py`.
