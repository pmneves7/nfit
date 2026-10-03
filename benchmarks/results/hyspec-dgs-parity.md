# HYSPEC DGS acceptance: checkpoint 6A-R

The independent fresh comparison passes for unsubtracted HYSPEC data. The saved background-subtracted reference differs; the completed [6A-R2 analysis](hyspec-background-analysis.md) isolates its principal geometry/normalization discrepancy and records the remaining final-cut covariance work. The aggregate receipt is `hyspec-dgs-parity.json`; no numerical arrays are included.

## Fresh raw and MDE comparison

Three real IPTS-36860 runs were reduced with the installed Shiver MDE recipe: Ei = 15 meV; automatic T0 = 100.11159018271724 µs; pause filtering; no bad-pulse rejection, time-independent background or detector grouping; Shiver's tip-pixel mask; recorded raw TOF crop; helium-3 and incident/final wave-vector corrections; unity detector-trajectory normalization. Runs 506277/506278 use Tank angle approximately −70° and different sample rotations; run 505555 uses approximately −34°.

The 80 × 80 × 1 × 57 histogram uses H,H,0; 0,0,L; H,−H,0; and energy. Explicit edges reproduce Mantid's actual grid. Bounds supplied without explicit edges to nfit are center bounds and would produce a different grid.

| Run | Original Shiver MDE events | Physical-bound MDE / nfit events | Maximum relative exposure difference |
| --- | ---: | ---: | ---: |
| 506277 | 380,121 | 380,122 | 1.48 × 10⁻¹¹ |
| 506278 | 475,873 | 475,874 | 8.90 × 10⁻¹² |
| 505555 | 203,183 | 203,185 | 2.15 × 10⁻¹¹ |

For the original MDE import, the event numerator C, accumulated event variance V, and event counts agree **exactly** with Mantid. For raw reduction compared with Mantid using the requested physical energy bounds, the same three histograms agree exactly, and every event's detector identity, energy, Q in sample coordinates, weight and variance agrees bit for bit. Coverage support is identical. Exposure N agrees to the precision shown above, including the lowest-exposure decile and coverage fringes. Final integrated cuts reproduce C/N and √V/N with relative differences below 2.28 × 10⁻¹³. Exposed zero-count cells remain included. Acceptance assertions enforce these claims.

### Observed-extrema conversion loss

The actual installed Shiver converter uses `ConvertToMDMinMaxGlobal` followed by `ConvertToMD` with its returned bounds. The energy bounds are observed event extrema, rather than the requested physical ±14.25 meV domain. Converting an extremum to float32 can move it beyond its double bound, rejecting an otherwise valid event. For example, run 506277's maximum is 14.100887297485091 meV; the stored event coordinate becomes 14.100887298583984 meV. The other losses follow the same minimum/maximum criterion.

Changing only the Mantid conversion's energy bounds to ±14.25 meV restores exactly the nfit event multiset. nfit currently retains these valid events; the original recipe's one/two-event difference is recorded separately. No compatibility option was adopted in this diagnostic.

### Geometry and preprocessing

Independent Mantid probes establish time-weighted default log extraction, six-significant-digit log serialization before equations, dynamic Euler rotation replacing static orientation, and Rx @ Ry @ Rz composition. Relative additional Tank rotation composes old @ Y(offset), tested with noncommuting rotations. Raw TOF crop endpoints are both inclusive. Loaded MDE detector masks are retained from the saved instrument parameter map and applied to normalization; Mantid's numerator bins the already reduced MDE events.

## Saved histogram: unresolved background comparison

The read-only identity-symmetry 50 K reference is `3D_HHL_3meV_50K_metallix_1.nxs`. Its history records Mantid 6.15.0.1, two 361-run sample collections, two angle-integrated backgrounds, refined UB, unity solid-angle/flux normalization, and no smoothing. Actual grid size is 160 × 220 × 8 × 48; the requested 0.5 upper limit on H,−H,0 extends to 0.62 to complete 0.14-wide bins.

Replaying both banks with the historical refined UB and actual uniform bounds does **not** reproduce the saved subtraction. Over 3,145,911 mutually finite cells, signal RMS difference is 0.004956 and uncertainty RMS difference is 0.001493 in the saved arbitrary units. In the lowest-exposure decile, signal RMS difference is 0.015201; coverage-fringe signal RMS difference is 0.005624. These are substantive differences, and finite support also differs.

The current measured-background service uses sample-charge fractions across angles and combines copies of a source event before computing variance. The historical Mantid recipe bins a background once per sample angle and accumulates independent-copy variance. The service retains normalized signal/errors and exposure, rather than a certified additive C/V payload. This diagnostic reconstructs C = I N and V = (σ N)² only for comparison, excluding zero-count display confidence intervals; it installs no reconstructed payload. These model differences require investigation in 6A-R2. They do not establish that either treatment is the accurate estimator, nor completely explain every residual.

### Background acceptance target

Paul reports that `HYSPEC_all.nfit` removed the background well and that the
historical Mantid subtraction looked worse. Read-only saved metadata confirms
that all four sample collections under `Workspace1/Group1` use enabled
`measured_events` background links. Their names contain “powder”, but the stored
mode replays full laboratory-frame event directions at sample angles. The
separate `powder averages` branch uses center projection.

The [completed 6A-R2 analysis](hyspec-background-analysis.md) validates that
directional treatment and final-grid source correlations, including coverage
fringes. Cached final-cut dependencies remain an implementation follow-up. Agreement
with the historical subtraction is a diagnostic comparison rather than the
scientific acceptance criterion. The available `histograms/export_metallix.py`
passes MDE backgrounds directly to `MDNorm`; this alone does not demonstrate a
radial upstream background model. The source construction and remaining
residuals still require attribution. Neither the successful visual subtraction
nor the current same-voxel variance handling establishes complete uncertainty
correctness.

All original local projects, MDE files and reference histogram passed before/after size and modification-time checks. The fresh raw/reference inputs also remained unchanged. ORNL scientific products stayed in the IPTS diagnostics folder. Only small scalar receipts were copied locally.

## Timing scope and reproducibility

Fresh native reduction plus binning took approximately 0.36–0.97 s per run; loaded-MDE binning took 0.046–0.078 s after an approximately 4 s first-call compilation. The manual Mantid chain includes mask and MDE persistence and has different overhead, so these are not a matched speed benchmark. The full saved-histogram diagnostic took approximately 62 s across both sample histograms and background replays on the local nfit interpreter.

The scripts are `validate_hyspec_mantid_reference.py`, `validate_hyspec_dgs_parity.py`, and `validate_hyspec_saved_histogram.py` under `benchmarks/`. The independent reference builder uses installed Mantid only as a manual diagnostic. The native scripts import no Mantid. Receipts record source hashes, reference versions, geometry probes, channel differences and uncertainty comparisons. No production or pytest dependency on Mantid was introduced.
