# Automatic Brillouin-zone sampling

Electronic observables converge at different rates. A mesh that is adequate
for a broadened density of states (DOS) may be inadequate for a sharp Lindhard
response, so nfit selects meshes against the calculated observable rather than
from one universal number of points.

Automatic selection is currently implemented for DOS and the bare Lindhard
susceptibility. Fermi-surface meshes remain visualization settings because
surface topology needs a different convergence test.

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
silently claim convergence.

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

## Lindhard certification

Lindhard certification compares the full complex Cartesian spin tensor
$\chi^0_{\alpha\beta}(\mathbf Q,E)$, not only its imaginary or neutron-projected
part. The broadening $\eta$, temperature, chemical-potential policy, operators,
and sampled $(\mathbf Q,E)$ points remain fixed throughout the search.
Arbitrary certification points are evaluated directly, so interpolation error
is not confused with integration-mesh error.

The GUI's **Check/refine convergence** action uses the momentum, energy window,
temperature, and representative energy count shown by the Lindhard convergence
viewer. Scripts can instead supply the actual experimental points. A
certificate on representative points supports a declared local claim; it is
not a proof over the entire Brillouin zone or every fit condition.

New Lindhard components use a finite zone-boundary representative momentum,
with reduced coordinate `0.5` on each periodic electronic axis. This avoids
silently selecting the unusually demanding static Pauli limit at
$\mathbf Q=0$, $E=0$. Choosing that limit explicitly remains supported; the
sampling panel warns that its narrow $-\partial f/\partial E$ shell may need a
much denser mesh and is not smoothed by the finite-energy response broadening.
The same panel exposes the complete scientific certificate domain
($\mathbf Q$, energy interval and count, and temperature), symmetry and mesh
shift, accuracy profile, concrete production mesh, and refinement and
full-mesh point budgets. Backend, memory, and interpolation diagnostics remain
under **Advanced** because they do not redefine the requested integral.

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

For project components,
`certify_tight_binding_dos_sampling(component)` and
`certify_lindhard_component_sampling(component, components, ...)` store the
serialized certificate and update the production mesh only after success.
The Lindhard wrapper accepts explicit `q_reduced`, `energy_meV`, and
`temperature_K` arguments for analysis-specific domains. Both wrappers accept
the same optional `progress_callback`; GUI progress is built on this public
hook. Copied calculation scripts include the concrete mesh and stored
certificate.

Automatic selection does not establish Fermi-surface topology convergence,
global electronic-RPA stability, or convergence with respect to lifetime
broadening. Those require separate diagnostics.
