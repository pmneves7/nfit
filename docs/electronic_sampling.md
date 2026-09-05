# Automatic Brillouin-zone sampling

Electronic observables converge at different rates. A mesh that is adequate
for a broadened density of states (DOS) may be inadequate for a sharp Lindhard
response, so nfit selects meshes against the calculated observable rather than
from one universal number of points.

Automatic selection is implemented for DOS, explicit-domain bare Lindhard
inspection, and complete dataset-facing tight-binding--Lindhard--RPA
pipelines. Fermi-surface meshes remain visualization settings because surface
topology needs a different convergence test.

## Accuracy profiles

| Profile | Normalized tolerance | Intended use |
| --- | ---: | --- |
| Preview | 5% | rapid model construction and qualitative inspection |
| Standard | 1% | routine analysis and the default starting point |
| High | 0.2% | final checks when the data warrant the extra cost |
| Custom | user supplied | a declared analysis-specific requirement |

The profile sets an accuracy target, not a particular mesh. The search must
pass the target on two successive refinements. This guards against accepting
one accidentally favorable comparison. After two passes, nfit accepts the
middle mesh: it has been compared directly with the finer reference, while the
preceding pass establishes that the ladder has entered a stable refinement
regime.

The maximum number of refinements and maximum mesh points are independent
safety budgets under **Advanced**. Exhausting a budget produces an explicit
`budget_exhausted` certificate; it does not weaken the selected tolerance or
silently claim convergence. A complete-pipeline search stops once the remaining
candidate meshes cannot possibly supply the required consecutive passes; the
certificate records that stopping reason instead of spending the final budget
on a result that cannot change the decision.

## Mesh search and certificate

nfit constructs a deterministic refinement ladder. DOS certification starts
with 16 points along the longest reciprocal direction and increases the linear
density by 1.3. Axis counts are proportional to the lengths of the
corresponding reciprocal vectors, giving approximately uniform physical
spacing in anisotropic cells. Every attempted mesh and comparison is saved in
a `SamplingCertificate`.

In the GUI, a progress window remains visible during a search. It
records the iteration, candidate mesh, maximum, root-sum-square, and integrated
metrics, pass state, and wall time for each completed candidate. Closing the
record after completion does not affect the stored certificate.

For successive results $A_c$ and $A_f$, nfit records a globally normalized
maximum change and an $L^2$ change,

$$
\epsilon_{\max} =
\frac{\max |A_f-A_c|}{\max |A_f|},\qquad
\epsilon_2 =
\frac{\lVert A_f-A_c\rVert_2}{\lVert A_f\rVert_2}.
$$

$A_c,A_f$ are arrays of the same observable at identical evaluation points,
computed using the coarser and finer integration meshes. Their entries have
the observable's units; both $\epsilon$ metrics are dimensionless.
The maximum runs over all compared entries and
$\|X\|_2=\sqrt{\sum_j|X_j|^2}$ after flattening the array (including real
and imaginary information for complex values). This is not a matrix spectral
norm. A finer-mesh reference is an estimate, not an exact solution.

Small denominators use a fixed numerical floor. DOS certification also records
the energy-integrated absolute difference divided by the integrated absolute
fine-mesh DOS. The comparison passes only when every applicable metric is at
or below the requested tolerance.

A certificate contains:

- the observable and declared calculation domain;
- the requested profile, tolerance, and stopping rule;
- all attempted meshes and their error metrics;
- the selected production mesh, or a budget-exhausted status; and
- model, method, symmetry, and thermodynamic provenance.

Changing the Hamiltonian, selected mesh, broadening, integration method, or
other certified input makes the corresponding certificate stale. A stale
certificate is retained as provenance but is not an accuracy claim.

## DOS certification

DOS certification compares the total DOS and every requested orbital
projection on the same fixed energy grid. For Gaussian integration,
`dos_broadening_meV` is held fixed. Tetrahedron integration uses its
unbroadened piecewise-linear result and a complete three-dimensional topology.
For a certified nfit-built model, `auto` diagonalizes only symmetry-unique
points, expands the eigenvalues exactly onto the ordered full grid, and then
performs the unchanged tetrahedron integration. `full` disables this
acceleration, while `reduced` requires it. Arbitrary orbital projections fall
back to full evaluation unless the projection spans the complete basis.

In the tight-binding model's **Calculate and inspect** tab, choose **Automatic
certification**, select an accuracy profile, and press **Check/refine
convergence**. nfit replaces `dos_mesh` only if the requested tolerance is
certified. **Manual mesh** uses `dos_mesh` directly and makes no automatic
accuracy claim. Automatic DOS searches begin with 16 points along the longest
reciprocal direction and increase that linear density by a factor of 1.3; the
other axes follow the same physical reciprocal-space spacing. The DOS viewer
uses the same model-level production mesh. Set the automatic or explicit energy
limits, number of comparison energies, integration method, Gaussian standard
deviation when applicable, and symmetry policy in **Density-of-states
sampling** before starting certification. These settings define the domain and
evaluation policy claimed by the stored certificate.

## Electronic-response pipeline certification

The project-facing Lindhard action certifies the complete observable compiled
for each applicable fit dataset. nfit follows the selected tight-binding and
Lindhard dependencies through every enabled consuming RPA component, then
includes orbital form factors, neutron polarization, powder averaging, bulk
conversion, and dataset normalization in the compared values. RPA enhancement
therefore cannot hide behind a certificate on the bare tensor.

The domain comes from the same masked, rebinned, temperature- and field-aware
`FitDatasetInput` objects used by fitting. Small datasets are evaluated in
full. For a larger dataset, nfit deterministically selects up to
`response_sampling_points_per_dataset` points by farthest-point coverage of
$H$, $K$, $L$, energy, temperature, and magnetic-field coordinates while
retaining coordinate extrema. Each dataset is normalized and compared
separately; every dataset must pass, so a large or high-amplitude dataset
cannot conceal another dataset's mesh dependence. Selected source indices,
coordinate ranges, original and valid point counts, and a domain digest are
stored in the certificate.

At each candidate mesh, arbitrary experimental wavevectors are evaluated
directly. The broadening $\eta$, powder orientation count, backend, component
parameters, and all dataset conditions remain fixed. This isolates the
Brillouin-zone integration mesh from interpolation, angular-quadrature, and
physical-broadening dependence. The model-owned convergence plot retains its
explicit representative $\mathbf Q$, energy, and temperature settings as a
separate inspection diagnostic; those plot settings no longer define the
project certificate.

Automatic certification stores the accepted mesh in `response_mesh`. Fits then
use that concrete mesh without adapting it inside the optimizer, preserving a
fixed numerical objective. Re-run certification at the fitted parameters when
the fit changes the Hamiltonian or broadening enough to make the saved
certificate stale.

## Scripted use

The low-level functions operate directly on an `ElectronicModel`:

```python
import numpy as np

from nfit import certify_dos_sampling, sampling_policy

certificate = certify_dos_sampling(
    electronic_model,
    np.linspace(-500.0, 500.0, 601),
    seed_mesh=[40, 40, 40],
    broadening_meV=5.0,
    policy=sampling_policy("standard"),
    progress_callback=lambda step: print(step.iteration, step.mesh, step.phase),
)
if not certificate.certified:
    raise RuntimeError("DOS mesh did not converge within the declared budget")
production_mesh = certificate.chosen_mesh
```

For project components, `certify_tight_binding_dos_sampling(component)` stores
the DOS certificate. `certify_group_lindhard_sampling(group, component)`
prepares the ordinary fit datasets and stores a complete-pipeline response
certificate. Lower-level callers with prepared data can use
`certify_lindhard_pipeline_sampling(component, components, datasets)` or
`certify_electronic_pipeline_sampling(...)`.

`certify_lindhard_component_sampling(component, components, ...)` remains the
explicit-domain bare-response API. It accepts `q_reduced`, `energy_meV`, and
`temperature_K` for an analysis-specific tensor inspection, but does not claim
convergence of a downstream RPA or experimental observable. All wrappers
accept an optional `progress_callback`; GUI progress is built on the same
public hook.

```python
from nfit import (
    certify_group_lindhard_sampling,
    load_project,
    save_project,
)

project = load_project("analysis.nfit")
group = project.data_groups[0]
lindhard = group.models["Bare response"]
certificate = certify_group_lindhard_sampling(group, lindhard)
if not certificate.certified:
    raise RuntimeError("response mesh was not certified within the budget")
save_project(project, "analysis.nfit")
```

Automatic selection does not establish Fermi-surface topology convergence,
global electronic-RPA stability, powder angular convergence, wavevector-
interpolation accuracy, or convergence with respect to lifetime broadening.
Those remain separate diagnostics. Changing $\eta$ changes the physical model
and is not a substitute for refining the integration mesh.
