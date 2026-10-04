# HYSPEC saved-MDE geometry roundtrip attribution

## Result

The small exposure discrepancy in the cached 12-operation HYSPEC comparisons originates in Mantid's saved-MDE geometry roundtrip. It is reproducible within Mantid using the **same six-operation histogram before and after SaveMD/LoadMD**. Replaying exposure in nfit with the actual reopened Mantid detector directions reproduces its changed exposure; the original resolved directions reproduce the resident Mantid result.

This is a diagnostic of Mantid 6.16.0.1 and Shiver 1.10.0 on the two 12-run, 50 K HYSPEC pilots. It does not justify changing nfit geometry or emulating this additional serialization loss. The ordinary raw-to-resident comparison remains the reduction parity target.

## Controlled evidence

`N` below is the trajectory-normalization exposure channel, in the reference
workspace's charge × meV × detector-weight units. Detector weights are one for
these no-vanadium HYSPEC controls. Relative L2 means `norm(N_loaded − N_resident) / norm(N_resident)` over the full grid. Both histogram edge arrays and recorded MDNorm settings are identical.

| Bank | Six-operation resident → reopened relative L2 | Largest absolute exposure change | Changed support cells |
| --- | ---: | ---: | ---: |
| 34° | 1.100145168261256e−6 | 0.3606527497422576 | 0 |
| 70° | 1.4945692762455069e−5 | 2.750994804931679 | 4 |

At the respective largest six-operation exposure residuals:

| Bank / grid index | Resident Mantid `N` | Reopened Mantid `N` | nfit replay with reopened geometry | Replay − reopened Mantid |
| --- | ---: | ---: | ---: | ---: |
| 34° / `(33,32,0,4)` | 3119.435992719995 | 3119.796645469737 | 3119.796645469818 | 8.09e−11 |
| 70° / `(54,29,5,44)` | 1528.744382263905 | 1531.4953770688367 | 1531.4953770702205 | 1.38e−9 |

These replay differences satisfy the existing `rtol=1e−12` nonzero-exposure gate at the selected cells. Replaying the original saved physical detector angles instead matches resident Mantid to 9.41e−11 and 5.64e−10, respectively. The replay uses the complete original grid, accepted charges, original energy limits, UB matrices, goniometers, and identical symmetry operations. Reducing the diagnostic to a one-cell grid would change the float32 grid-index arithmetic and is not a valid substitute.

A metadata-only LoadMD probe found **exactly zero changes** in UB, goniometer matrices, and accepted proton charge for all 12 experiment records in each bank. The first experiment's actual reopened instrument directions were measured for all 20,480 detector pixels; saved geometry and masks are common across the 12 original records per bank. Sampled event numerator, event variance, and event counts at the largest residual cells are unchanged. No exhaustive full-array claim about those event channels is made here.

For the 16 largest cached 12-operation residual cells per bank, native exposure agrees with the resident six-operation exposure plus its third-axis reflection within 1.84e−10 (34°) and 1.79e−8 (70°). Reopened Mantid 12-operation exposure agrees with the corresponding **reopened** six-operation sum within 9.10e−13. This independently attributes those reported 12-operation outliers to the roundtrip rather than the additional symmetry operations.

## Geometry mechanism

There are two separate precision steps. The original instrument-definition equation substitutes a run-log scalar formatted by a default C++ stream, hence six significant digits. The Tank rotation actually used during initial geometry resolution is therefore −33.9836° or −69.9899°, rather than the full-precision `s2` log. nfit already follows this original-resolution convention.

SaveMD then records the Tank rotation as a quaternion parameter through the parameter-map string. Quaternion parameters use the generic default stream precision; unlike double and vector parameters, they have no higher-precision serialization specialization. LoadMD rebuilds the instrument from its saved XML and reapplies this saved parameter map. Its component rotation update normalizes the relative quaternion transform before moving descendant detector positions.

| Bank | Full-precision `s2` log, degrees | Original resolved Tank angle, degrees | Serialized Tank quaternion `[w,x,y,z]` | Reopened Tank angle, degrees | Actual direction rotation, degrees |
| --- | ---: | ---: | --- | ---: | ---: |
| 34° | −33.98356246948242 | −33.9836 | `[0.956347,0,−0.292235,0]` | −33.98360397901862 | −3.979018615041241e−6 |
| 70° | −69.9898681640625 | −69.9899 | `[0.819203,0,−0.573504,0]` | −69.98985138617554 | +4.861382446108564e−5 |

The direction change predicted from **reopened quaternion angle minus the original six-significant-digit angle** matches all measured detector unit vectors to 3.06e−15 (34°) and 8.47e−16 (70°). An independent fitted rigid rotation gives the same changes, with residuals below 2.89e−15. Comparing the quaternion angle to the unrounded `s2` log gives the wrong change and fails the exposure replay; quaternion normalization alone against that log is not the attribution.

Small geometry shifts can change trajectory intersections with bin edges and empty-cell support, making a larger localized exposure change than their angular magnitude suggests. These measurements establish the mechanism without replacing the scientific geometry with the serialization artifact.

### Installed-version source references

- Original log equation substitution: [XMLInstrumentParameter::createParamValue](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/Geometry/src/Instrument/XMLInstrumentParameter.cpp#L186).
- Generic parameter stream precision and higher-precision double/vector specializations: [ParameterType::asString](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/Geometry/src/Instrument/Parameter.cpp#L30); quaternion formatting: [Quat::printSelf](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/Kernel/src/Quat.cpp#L609).
- Parameter-map serialization: [ParameterMap::asString](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/Geometry/src/Instrument/ParameterMap.cpp#L932).
- Saved XML reconstruction and rotation parameter application: [ExperimentInfo::loadInstrumentInfoNexus](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/API/src/ExperimentInfo.cpp#L1037), [ExperimentInfo::readParameterMap](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/API/src/ExperimentInfo.cpp#L1252).
- Normalized component transform and detector position update: [ComponentInfo::doSetRotation](https://github.com/mantidproject/mantid/blob/v6.16.0/Framework/Beamline/src/ComponentInfo.cpp#L273).

## Artifacts and limits

All cluster scientific artifacts remain under `/SNS/HYS/IPTS-36860/shared/nfit/benchmarks/6A-P-node19/`:

- `mantid/hys{34,70}-pilot12-shiver-reference-01/arrays_primary_6copies.h5`, `arrays_rebin6_6copies.h5`, `arrays_rebin12_12copies.h5`, `merged_mde.nxs`, and `receipt.json`.
- `nfit/hys{34,70}-pilot12-v011603-timeweighted/histogram-{6,12}.h5`.
- `diagnostics/hyspec-loaded-geometry-probe.{npz,json,log}`: metadata-only LoadMD direction probe, no event loading or MDNorm.
- `diagnostics/hys{34,70}-timeweighted-12-residual-locations.json`: previously selected residual pixels.
- `diagnostics/hyspec-roundtrip-normalization-replay.json` and `hyspec-roundtrip-rigid-geometry.json`: derived scalar replay receipts.
- `diagnostics/hys{34,70}-v011604-vs-resident-shiver-12.json`: fresh resident twelve-operation controls with identical support and exposure relative L2 5.18e−15 / 2.51e−14.

Small local metadata mirrors and bounded replay scripts/results are under `/private/tmp/hyspec-*-geometry-diagnostic.npz`, `/private/tmp/hyspec-loaded-geometry-probe.npz`, `/private/tmp/hyspec-roundtrip-normalization-replay.{py,json}`, and `/private/tmp/hyspec-roundtrip-rigid-geometry.json`. The source geometry originals and benchmark projects were not modified. The decisive checks cover full-grid six-operation exposure differences, all detector directions in the probed experiment, and selected worst-cell exposure replay; fresh resident 12-operation controls are a separate acceptance check.
