# DGS trajectory boundary residual

Targeted diagnostic on 2026-10-02 explains the largest exposure difference in
the [0.107.0 parity comparison](dgs-parity-final.md). Scientific code and saved
projects were not changed. The target cube cell is near
(H,K,L,E)=(−.76,1.21,.10,0 meV), with lower energy edge −0.25 meV.

## Confirmed cause

The compiled native trajectory integrator clips segments at the requested
energy window before midpoint classification. Mantid retains spatial trajectory
segments outside the window, then calculates/classifies float32 midpoint energy.
The nfit Python fallback follows the latter route.

Two segments from detector 86388, operation `y,z,x`, sample angle 159.5 degrees,
have true energy endpoints −0.2500000000000071 and −0.25001233769975073 meV.
Their true midpoint is −0.2500061688497084 meV, outside the grid, but Mantid's
rounded midpoint is −0.24999961256980896 meV, inside its first bin.

| Run | Additional exposure from the retained segment |
| --- | ---: |
| 393569 | 0.0030484033835756295 |
| 393324 | 0.00008643859672787152 |
| Total | 0.003134841980317781 |

Scalar replay with full original grid edges produces:

| Route | Exposure | Difference from measured engine |
| --- | ---: | ---: |
| Clipped | 260.6006687925138 | −5.68e−14 from nfit |
| Unclipped | 260.6038036344941 | +3.04e−11 from Mantid |

The actual production Python fallback reproduces both additional segment
contributions. Cropping the grid around the target cell does not reproduce the
original float32 indexing, so the diagnostic retains its original grid origin.

This accounts for the entire 12.03 ppm difference. It is boundary leakage after
rounding, rather than accumulation order or an event-reconstruction discrepancy.
Counts, weighted numerator, and numerator variance remain identical. The native
clipping follows the requested mathematical energy boundary more closely in this
case. No scientific default or backend treatment was changed by this diagnostic.

## Evidence and follow-up

The compact numerical receipt and standalone fallback replay are retained in the
local diagnostic directory as `/tmp/nfit-residual-confirmed-cause.json` and
`/tmp/nfit_residual_fallback.py`. Full run/detector tracing stays in the user's
existing ORNL diagnostic cache under `nfit-converter-1d/residual-trace`.

Validation/adoption of alternatives and resolution of this backend convention
difference are deferred to the late measurement-pipeline acceptance checkpoints.
