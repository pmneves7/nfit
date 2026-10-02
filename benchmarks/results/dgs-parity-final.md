# SEQUOIA DGS parity and performance validation

Manual validation on 2026-10-02, `analysis-node21.sns.gov`, of the scientific
working tree prepared for nfit 0.107.0. [Aggregate receipts](dgs-parity-final.json)
contain source hashes, per-cell comparisons, exposure-region definitions and
resolved symmetry matrices. Scientific projects were opened read-only.

## Same MDE, same histogram

Both engines used the Mantid-produced 617-run `All_Ei60_T5K_all_617_runs.nxs`
(177778751 events), `van382236_D` for vanadium and mask, identical UB, momentum
axes (1,1,1), (1,−1,0), (1,1,−2), and 0.5 meV energy steps. Runs are
392985–393469 and 393500–393631. No sapphire subtraction was performed.

The default comparison uses first-run trajectory Ei (60.77019987153442 meV),
Mantid numerical conventions and independent symmetry-copy variance.
C denotes the weighted event numerator, V its additive count-space variance,
and N the trajectory exposure. The displayed intensity is C/N and standard
uncertainty is sqrt(V)/N, conditional on N. Intensity and N retain the
experiment's arbitrary normalization units; event-correction weights are
dimensionless. Pooling selected cells sums C,V,N before dividing.

| Grid | Shape | First edges | Steps |
|---|---|---|---|
| Thin | 21×3×3×101 | −0.025, −0.035, 0.29, −0.25 | 0.03,0.03,0.02 r.l.u.; 0.5 meV |
| Full cube | 67×134×101×101 | −1.015, −2.015, −1.01, −0.25 | 0.03,0.03,0.02 r.l.u.; 0.5 meV |
| Fine selected slab | 61×7×5×61 | −0.005, −0.035, 0.305, −0.25 | 0.01,0.01,0.01 r.l.u.; 0.5 meV |

C,V and event counts are **literally equal in every cell** for thin3bar,
thin3barm, the full3bar cube and the fine3barm slab. The full cube comparison
covers 91584578 cells, including the previously discrepant geometry boundaries.
The cube contains 352458245 symmetry-copy event contributions in both engines.
Coverage is identical: 59182598 positive-N cells, with 13 event contributions
in N≤0 cells in both engines. Covered empty cells have V=0; no artificial
positive error floor remains.

Exposure is numerically close, not bitwise identical. Full-cube relative L2
error is 1.95e−10. The worst relative N and sigma discrepancy is 12.03 ppm at
(H,K,L,E)≈(−.76,1.21,.10,0 meV), on the lowest energy output boundary:
N=260.60066879 native versus 260.60380363 Mantid, count 1 and identical C,V.
The absolute sigma discrepancy is 5.28e−8. Neighbor differences do not cancel;
histograms alone did not identify the individual trajectory arithmetic causing
this small residual. A subsequent [targeted boundary trace](dgs-trajectory-boundary.md)
accounts for the entire difference through two below-edge segments admitted by
Mantid's rounded midpoint classification.

For the lowest exposure decile of individual 4D cells at E≥3.5 meV
(N≤161.99621754,5377673 cells), C,V and count remain exact. Maximum sigma
relative discrepancy is 1.61e−11. For the fine slab, N relative L2 error is
2.95e−11; maximum fine-cell sigma relative error is 1.00e−8. In the pooled
H=.13… .23/E=3.5…25 region, sigma maximum relative error is 1.96e−14; in the
pooled low-exposure decile it is 2.37e−14. Thus the low-exposure band is real
and shared by the engines; this histogram comparison has no native-only
uncertainty inflation there.

These ratios use common N>0 support, with no additional tiny-exposure cutoff.
Maximum relative errors omit zero reference values. Geometry fringes are
positive-N cells within two axis-neighbor steps of interior N=0 cells;
artificial zeros outside the output grid are excluded.

The flagged pooled pixel near H=.24,E=0 contains 286 events in both engines,
with identical pooled C=324.81804060935974 and V=368.9055302143097.
C,V and count also match independently in each of the 35 underlying fine cells. Shared-MDE N is 170335.9876275103 native versus 170335.987719243
Mantid:−5.3854e−10 relative. Its intensity difference is 1.03e−12 and sigma
difference 6.07e−14. The largest underlying N difference is−4.5867e−5 at
K/L cell(0,2), with N≈4578.63. The ppm residual plot intentionally magnifies
these small differences; it does not show a practical intensity outlier.
The [subsequent trajectory trace](dgs-trajectory-boundary.md) identifies the
operation-level cause of the worst cube residual; this pooled fine pixel is a
separate, much smaller numerical difference.

## Raw event reduction

All 617 monitor calibrations agree with saved Mantid Ei and T0 to floating-point
roundoff: maximum Ei error 3.98e−13 meV and T0 error 1.91e−11 microseconds,
with no fallback calibrations. Current records provide per-run Ei, not an
independently measured Ei for each pulse. Pause/deadtime and bad-pulse filtering
are checked separately from energy and detector corrections.

Fresh representative Mantid reductions of 392985,393089,393387 contain
25569,7878,421781 accepted events respectively. The final production native
reducer matches all 455228 weights literally, accepted counts and detector
sequences exactly, and energy transfer to 5.54e−13 meV. Independent event V
agrees to 3.55e−15; this residual comes from extracting Mantid sigma and
squaring it for the diagnostic rather than reading float 32 errorSquared.
Signal and variance sums agree exactly.

The last weight discrepancy was traced to Mantid's cylinder ray intersection,
which obtains an effective radius 0.012699997477625402 m rather than nominal
0.0127 m for the coarse SEQUOIA pixel height 0.009375 m. Reproducing the
source-ordered intersection and He3 correction arithmetic removes the weight
ULPs. The correction checks shape/height and instrument geometry, rather than
assuming every file shares SEQUOIA geometry. High-precision mode retains nominal
geometry; unsupported Mantid-compatible cylinder geometry is explicit.

The final fresh 617-run pass built only transient reduced-event caches and
produced the fine3barm slab. Reusing those events then produced the full3bar
cube. **C,V and counts equal the Mantid reference in every cell of both outputs**,
not merely in their sums. Cube N relative L2 discrepancy is 1.95e−10; its
maximum sigma relative discrepancy is 12.03 ppm at the same 0 meV output boundary.
The lowest-exposure 4D decile sigma maximum relative discrepancy is 5.35e−11;
the fine pooled H=.13… .23/E=3.5…25 region is 3.92e−13.

The fresh pass recorded 617 cache misses and no hits; the second binning recorded
617 hits and no misses. Temporary cache files occupied 11,492,603,178 bytes
(10.70 GiB), with no event payload held in RAM as a whole. Explicit cleanup
verified that all temporary cache files were removed. The original main project
and the saved zoom project retained their sizes and modification times.

## Timings and ordinary sequential Mantid estimates

| Operation | nfit | Mantid/Shiver | Comparison scope |
|---|---:|---:|---|
| Same MDE → thin3bar |107.71 s|154.29 s|1.43× engine wall-time ratio |
| Same MDE → thin3barm |165.99 s|305.13 s|1.84× |
| Same MDE → full3bar cube |244.20 s|751.00 s|**3.08×** |
| Same MDE → fine3barm slab |170.29 s|298.17 s|1.75× |
| Fresh 617 reduction +fine3barm +temporary cache |371.46 s (6.19 min)|—|Not isolated reduction time |
| Same reduced events → cached full3bar cube |238.93 s (3.98 min)|—|No second raw reduction |
| Three representative raw runs |3.57 s|40.47 s|Different materialization outputs; diagnostic work excluded |
| Standard 617-run Mantid reduction estimate |—|**~139 min**|Sequential external job, shared setup counted once; final merge/save additional |

The representative native time covers metadata/monitor loading, shared
geometry, mask/charge setup and reduction to Qlab,E,W,V event rows. Its sample
rotation and histogram projection occur during binning. Mantid additionally
materializes an MDE workspace through ConvertToMD; SaveMD and diagnostic
extraction/export are excluded. These phases have different output contracts,
so the three-run numbers are not a standalone end-to-end speedup claim.
The native caches retain the full imported energy window (−0.95 Ei to+0.95 Ei),
not just the selected histogram grids. Cache bytes include headers and
per-detector normalization payloads as well as events, and cannot be converted
to an event count by dividing by the six-column row size. Exact fresh-raw
parity above concerns the selected fine/cube histogram cells, not every event
outside their momentum and energy bounds.

Mantid MDE loading was 118.43 s for the thin/cube session and 114.26 s for
the fine session; vanadium loading was 2.52 and 2.25 s. Native event reading
is included in every MDE binning time; metadata preparation was~4 s. Mantid
binning starts with its MDE already resident. Thus read costs are explicitly
separated, and engine-wall ratios are not end-to-end reduction speedups.

The standard 617-run estimate is 40.474 s/3×617, not a measured full reduction.
The three selected runs differ in size and are not a random sample. Combining
that estimate with measured full-cube binning gives~151.25 min, with final
merge/save additional; reopening a saved MDE adds its measured loading time.
Historical node03 estimates and timings are retained in the
[prior report](dgs-reference-current.md) and must not be treated as same-host
comparisons. The earlier [eight-job Shiver experiment](sequoia-performance.md)
measured 1627 s (27.1 min), including recovery, merge and MDE save; it is a
historical external-parallel-job reference, not a new current-policy measurement.

Both engines ran sequentially, one external process at a time, with up to 64
internal threads, BLAS 1 and OpenMP 64. The host has two AMD EPYC 9334 CPUs,
64 physical cores and 1.5 TiB RAM. Native limits were 64 CPUs and 200000 MiB.
OS caches were not flushed and first native dispatch can include JIT.
Native Python 3.14.7/NumPy 2.5.3 and Mantid 6.16.0.1/Python 3.12.13/NumPy 2.1.3
were supplied by existing installations. No runtime was installed in the data
folder. Peak RSS is a process-lifetime high-water mark: full-MDE native 38.25 GiB,
Mantid 42.89 GiB; small native fine slab~1.25 GiB. Fresh-raw fine peak RSS was 3.49 GiB and cached-cube peak RSS 41.83 GiB.

## Saved recipe and uncertainty scope

The saved `Al2O3 3barm zoom` recipe belongs to the 617-run NiO sample group,
not the 738-run sapphire background source, despite its name. The sample group
has an enabled background link; the present comparisons use unsubtracted NiO.
Its cached axes confirm 0.01 r.l.u. momentum steps and energy centers 0…50 meV;
−2.915 meV is only a viewer limit. Inclusive center selections K=−.03…+.03
and L=.31… .35 select 7 and 5 fine cells.

That sample-owned recipe uses
`rotate(order=3, axis=[1,1,1]); mirror(plane=(1,1,-2)); rotate(order=2, axis=[1,1,-2])`.
It resolves to 12 matrices, but they differ from the requested explicit 3 barm
permutations. The JSON preserves all matrices. A native-only same-MDE comparison
changed the selected slab from 103180 to 93356 event contributions and weighted
C 119808.64→108587.59, while N 558.98 M→564.28 M. This is a recipe difference,
not evidence of a variance implementation error. No recipe was modified and
this audit does not decide which symmetry should be physically imposed.

The Mantid-compatible diagonal convention treats copied events independently.
Optional within-bin covariance correction is distinct from full covariance:
copies in different bins remain correlated when integrated later. Optional
high-precision arithmetic and per-run trajectory Ei intentionally depart from
Mantid conventions. Stable monitor variance arithmetic differs for 26 NiO runs
(maximum Ei 0.01104 meV/T0 0.827 microseconds); it is explicit, not the default.
These alternatives are not claimed to provide universally more accurate
uncertainties.

Observed-zero count variance is the plug-in convention, not proof that an empty
covered cell has zero physical uncertainty. Normalization-calibration uncertainty,
full covariance, generic point-measurement estimators and every viewer cut path
are outside this histogram parity result. Those remain separate refactor gates.

## Reproduction and source evidence

Manual scripts and full histogram fields stay under the user's existing
`~/.cache/nfit-converter-1d/` diagnostics. Small pooled slabs and plots were
copied locally with explicit authorization. The aggregate JSON contains no
full event arrays. The source snapshot's stale `nfit.egg-info` reports 0.89.13
in that diagnostic process; the unchanged installed desktop reports 0.106.1.
Scientific source hashes and paths identify the 0.107.0 working-tree code,
independently of this metadata shadowing.

Reference implementation details were checked against official Mantid 6.16.0
sources: [MDNorm](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/MDAlgorithms/src/MDNorm.cpp),
[BinMD](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/MDAlgorithms/src/BinMD.cpp),
[SlicingAlgorithm](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/MDAlgorithms/src/SlicingAlgorithm.cpp),
[He3TubeEfficiency](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/Algorithms/src/He3TubeEfficiency.cpp),
[ShapeFactory](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/Geometry/src/Objects/ShapeFactory.cpp)
and [Line intersections](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/Geometry/src/Math/Line.cpp).
The cluster's `define_data_example-2.py`, `define_slices_example-2.py` and
`slice_maker_example-2.py` supplied recipe context; measured results, rather
than recipe names, establish the comparisons.
