# Powder angular convergence

A powder dataset supplies $Q=|\mathbf Q|$ rather than a direction, so every
model that predicts powder data evaluates the single-crystal response on
`powder_orientations` deterministic approximately equal-area sphere directions
and averages them.

That count is a numerical approximation exactly like an integration mesh or a
lifetime broadening, and it deserves the same treatment: a stated tolerance and
a certificate rather than a default taken on trust.
`powder_convergence_scan` evaluates the observable at a sequence of orientation
counts, compares each against the densest one, and reports the largest absolute
and relative deviation over the fitted points.

## Scanning

```python
from nfit import powder_convergence_scan

result = powder_convergence_scan(
    component,
    powder_data,
    orientation_counts=(14, 26, 50, 100, 200),
    relative_tolerance=1e-3,
)
print(result.converged_count)          # smallest certified count, or None
print(result.max_relative_deviation)   # one value per count
```

The scan is model-agnostic. It varies only `powder_orientations` on a *copy* of
the component, so the caller's setting is never modified and the scan works for
every powder-averaging model — `heisenberg_rpa`, `generalized_paramagnon`, and
the electronic-response models — without knowing how any of them evaluates.

| Field | Meaning |
| --- | --- |
| `orientation_counts` | the counts evaluated, ascending |
| `values` | `(n_counts, n_points)` model prediction at each count |
| `max_absolute_deviation` | largest deviation from the reference, per count |
| `max_relative_deviation` | the same, divided by the reference magnitude |
| `reference_count` | densest count, which every other row is compared against |
| `converged_count` | smallest count meeting the tolerance, or `None` |

## Convergence is not monotone in the orientation count

The directions are a quasi-uniform spiral rather than a nested sequence, so a
coarse count can land favourably by accident while a denser one does not. On a
Heisenberg chain the deviations against a 400-direction reference run

| directions | 8 | 14 | 26 | 50 | 100 | 200 |
| --- | --- | --- | --- | --- | --- | --- |
| max relative deviation | 1.9e-3 | **6.6e-3** | 1.1e-3 | 1.8e-4 | 4.7e-5 | 1.6e-5 |

The 14-direction quadrature is worse than the 8-direction one. Reporting the
first count under the tolerance would therefore certify a count whose immediate
neighbours fail, so `converged_count` is the smallest count for which that count
**and every denser count** meet the tolerance. Read the trend across several
counts rather than a single pair.

An isotropic response is independent of orientation, so it agrees across counts
to machine precision. Any deviation the scan reports for a real model is
therefore physical angular structure, not noise in the quadrature machinery.

## Choosing a count

The default of 50 directions gives a relative deviation of order $10^{-4}$ for a
weakly anisotropic model and degrades as the response becomes more structured
in $\mathbf Q$. Cost is proportional to the count: every response point is
evaluated once per direction, so 50 directions make a powder prediction 50 times
more expensive than the equivalent single-crystal one. Scan before committing to
a production count, and rescan if the fitted parameters move far enough to
change the angular structure.

## Related certificates

The electronic-response path has its own mesh and broadening certificates; see
[Bare Lindhard response](lindhard.md) and
[Electronic sampling](electronic_sampling.md). Those cover the integration mesh
and the lifetime broadening, which are independent of this angular axis — a
powder electronic-response calculation should converge all three.
