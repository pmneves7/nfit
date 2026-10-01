# NiO uncertainty paths and coverage fringes

Measured 2026-10-01 with nfit 0.105.5 and Mantid 6.16.0.1 on ORNL.
The [aggregate report](nio-uncertainty-fringes.json) contains checksums and
numerical comparisons. Private source events and diagnostic slabs are not
included in the repository. Production numerical behavior is unchanged.

## Matched input and grid

Both engines histogrammed the same 177,778,751 reduced events in
`All_Ei60_T5K_all_617_runs.nxs`, with the same stored UB, vanadium/mask
`van382236_D`, 617 NiO runs, and six 3bar symmetry operations:
`x,y,z;y,z,x;z,x,y;-x,-y,-z;-y,-z,-x;-z,-x,-y`.
This isolates histogramming and normalization from raw-event reconstruction.
Mantid was run manually outside pytest. nfit's diagnostic and unit tests neither
import nor execute Mantid or Shiver.

Axes are (1,1,1), (1,-1,0), (1,1,-2), and energy transfer. Edges and steps are:

| Axis | Lower edge | Upper edge | Step | Unit |
| --- | ---: | ---: | ---: | --- |
| H | −0.025 | 0.605 | 0.03 | r.l.u. |
| K | −0.035 | 0.055 | 0.03 | r.l.u. |
| L | 0.29 | 0.35 | 0.02 | r.l.u. |
| Energy | −0.25 | 50.25 | 0.5 | meV |

These reproduce the current saved cube's center selections, including GUI
snapping, rather than inferring the screenshot's finer grid. The slab shape is
21 × 3 × 3 × 101. K and L are pooled into an H/energy map. The energy cut selects
H centers 0.14, 0.17, 0.20, 0.23 r.l.u. and energy centers 3.5–25 meV.
Mantid MDNorm also made a direct cut with H edges 0.125–0.245 r.l.u., all selected
K/L extent, and the same energy bins.

## Fringe results

Coverage means positive trajectory exposure, independently of observed counts.
The geometric fringe is within two map pixels of an uncovered pixel, excluding
the artificial crop exterior. The low-exposure subset is the bottom decile of
exposure in each energy row, with ties included. Both use energy ≥3 meV.

| Subset | Map pixels | Empty covered fine cells | Current / event-variance map error: median | p95 |
| --- | ---: | ---: | ---: | ---: |
| Geometric fringe | 92 | 41.0% | 1.084 | 1.855 |
| Bottom exposure decile | 155 | 90.0% | 2.434 | 3.476 |

Error ratios use pixels with positive reference variance and **the same nfit
exposure** in both calculations. The low-exposure subset contains a further
84 map pixels with zero event variance and positive current error; the
geometric fringe contains 23. Their ratios are undefined and are not silently
included as finite ratios. Empty-cell prescriptions account for 87.8% of the
current accumulated numerator variance in the low-exposure subset.

In both subsets, fine-cell counts match exactly, and corrected numerators and
nonempty numerator variances match to floating-point precision. Thus the added
error is not unexplained event-weight variance in these fringe measurements.
Across the complete slab, four fine cells differ by one event: paired hidden-axis
redistributions at 1 and 29 meV. These cancel in the H/energy map. They remain
to be checked against individual event coordinates and engine boundary arithmetic;
pooled equality does not prove every fine cell is identical.

Coverage differs in eight nfit-only and two Mantid-only fine cells, all with
zero events. Neither engine places positive events in its uncovered cells in
this slab. At map level, one nfit-only covered pixel remains.

## Three distinct causes

1. **Confidence limits are propagated as variances.** nfit substitutes a scaled
   zero-count Feldman–Cousins upper endpoint into the stored symmetric error.
   Hidden-axis pooling then squares and adds it. This reproduces the
   partition-dependent behavior established by the synthetic baseline. The RMS
   correction scale depends on events landing inside the requested histogram,
   not an independently estimated local response of the empty fringe cell.
2. **Trajectory energy changes boundary exposure.** nfit uses per-run incident
   energies; this Mantid reference uses a common first-run trajectory energy.
   Fringe exposure ratios have median 1.021, p95 1.228, and maximum 3.524.
   Holding nfit trajectories at Mantid's common 60.7702 meV changes these to
   median 1.000004, p95 1.000033, maximum 1.000740. This identifies the energy
   convention as the dominant normalization discrepancy in this comparison.
   It does not justify discarding measured per-run energies. Stored MDE event
   coordinates remain unchanged by this diagnostic override.
3. **Cuts change the estimator.** Hidden-axis integration pools count numerators
   and exposure, but regular/rotated box profiles and display coarsening use
   inverse variances of normalized intensities. The ROI annotation instead sums
   normalized intensities and errors in quadrature, without bin widths. These
   target different quantities and do not reproduce a direct MDNorm cut.

For the chosen cut, direct Mantid binning and fine-then-pooled Mantid statistics
have identical counts, numerator, and numerator variance; exposure differs by
2.42 × 10⁻⁷ in relative L2 norm. In the native cut, current inverse-variance
errors divided by pooled event-variance errors range from 0.843 to 6.564
(median 1.263). The native cut intensity differs from the exposure-pooled
intensity by 14.3% in relative L2 norm. This demonstrates estimator differences,
not a universal bias measurement or proof that all large errors are excessive.

At 12 meV the pooled cut contains three events. At fixed exposure, the current
pooled error is 4.97 × 10⁻⁵ a.u.; the event-variance error is 1.36 × 10⁻⁵ a.u.
The current inverse-variance cut reports 4.65 × 10⁻⁵ a.u.

## What large errors do and do not establish

The old saved cube has a single covered empty fine cell at H=0.17 r.l.u.,
energy 24.5 meV. Its exposure is 196.7116 recipe units and its stored error is
0.0082293 a.u., exactly the current 1.29 × RMS / exposure prescription. That
pixel's large error does not require accumulating many empty-cell limits.
Its weak exposure can legitimately imply a large upper limit for the unknown
intensity. Mantid's zero accumulated event variance is not zero uncertainty
about that unknown intensity.

A correct count/exposure likelihood includes measured zeros, then constructs
the final measurement's interval. A uniform spatial average or integral may
have substantial uncertainty from weakly measured portions even if adjacent
cells are well measured. It needs geometric weighting and an explicit target,
not an automatic preference for smaller error bars. Weighted-event interval
coverage and covariance from fractional sharing or symmetry copies remain
separate validation tasks.

The archived native cube has reduction-cache version 1; current native
reduction uses version 2. Its count/weight differences from shared Shiver MDE
are therefore not evidence about a fresh current native reduction. No project
or production cache was modified. Raw Ei/T0 and pulse-filter parity remain
for the later DGS checkpoint.

## Reproduce the native path trace

Export normalized nfit/Mantid slabs and Mantid numerator/variance on the grid
above, then run `benchmark_uncertainty_paths.py` as documented in
[the benchmark guide](../README.md#histogram-uncertainty-diagnostics).
The helper runs actual native slice, regular/rotated cut, ROI sum, and display
coarsening services, and reports full arrays locally. Its floor-removal variant
is explicitly an unsubtracted-event diagnostic, not a proposed generic fix.
The 2× display-coarsening comparison is illustrative; the screenshot's setting
is unknown. Rotated profiles also choose their own projected center grid, so
they must not be compared index-for-index with regular profiles.

The next checkpoint should preserve additive event statistics and separately
validate final-bin confidence intervals. Subsequent estimator work must make
count pooling, measurement means, and spatial integration explicit throughout
the GUI and scripting APIs.
