# Bare Lindhard spin susceptibility

`lindhard` calculates the complex particle--hole response of a referenced
`tight_binding` component. Use it when measured magnetic fluctuations can be
compared with the noninteracting band response before a Stoner or
Hubbard--Hund interaction dressing is introduced.

The tight-binding component remains a separate electronic-structure model.
Its onsite, hopping, and spin--orbit coefficients can be fixed or fitted
through the same simultaneous fit, while `lindhard` owns the response
broadening and response-specific settings.

## Complex susceptibility

The starting ansatz is a normal-state gas of independent quasiparticles in the
fitted static [tight-binding Hamiltonian](tight_binding.md#reciprocal-space-hamiltonian-and-bands).
Each excitation removes an electron from band $n$ at reduced momentum
$\mathbf k$ and adds it to band $m$ at $\mathbf k+\mathbf q$.
The two momenta are dimensionless coordinates of the electronic reciprocal
basis; $n,m=1,\ldots,N_b$ label all bands of the represented basis.
$E=\hbar\omega$ is transferred energy, $\epsilon_{n\mathbf k}$ is a band
energy, and $\eta>0$ is a constant retarded-denominator broadening, all in meV.
This is a particle--hole energy width, not automatically a one-electron
lifetime or an instrumental resolution width.

For an ordered set of dimensionless one-particle matrices $O_A$, with $A,B$
operator labels rather than band indices, the generalized bare response is

$$
\chi^0_{AB}(\mathbf q,E)=
-\sum_{\mathbf k}w_{\mathbf k}\sum_{nm}
\frac{f(\epsilon_{n\mathbf k})-f(\epsilon_{m,\mathbf k+\mathbf q})}
{E+\epsilon_{n\mathbf k}-\epsilon_{m,\mathbf k+\mathbf q}+i\eta}
M^A_{nm}M^{B*}_{nm},
$$

The weights obey $w_{\mathbf k}\ge0$ and $\sum_{\mathbf k}w_{\mathbf k}=1$;
a uniform full mesh of $N_k$ points has $w_{\mathbf k}=1/N_k$. The band sums
are not averaged. The result is therefore extensive per electronic model cell,
with units meV$^{-1}$ for dimensionless operators. An implicit-spin orbital
bubble is calculated for one spin species; physical spin factors are applied
only at projection, as specified below.

The occupation is
$f(\epsilon)=[\exp((\epsilon-\mu)/(k_BT))+1]^{-1}$, with chemical potential
$\mu$ in meV, temperature $T$ in K, and $k_B=0.08617333262$ meV/K.
At zero temperature, occupations are 1 below $\mu$, 0 above, and 1/2 at $\mu$.
$\epsilon$ and $\mu$ use the same energy zero. The full transition definition is

$$
H(\mathbf k)u_n(\mathbf k)=\epsilon_{n\mathbf k}u_n(\mathbf k),
\qquad
\sum_a u_{an}^*(\mathbf k)u_{am}(\mathbf k)=\delta_{nm},
$$

$$
\begin{aligned}
M^A_{nm}(\mathbf k,\mathbf Q)
&=u_n(\mathbf k)^\dagger O_A(\mathbf Q)u_m(\mathbf k+\mathbf q)\\
&=\sum_{a,b=1}^{N_b}u_{an}^*(\mathbf k)
[O_A(\mathbf Q)]_{ab}u_{bm}(\mathbf k+\mathbf q).
\end{aligned}
$$

Here $a,b$ index the orthonormal orbital/spin basis, $u_{an}$ is component $a$
of normalized eigenvector $n$ (the column returned by diagonalization), and
$[O_A]_{ab}$ has destination row $a$ and source column $b$. The vector is in
the Wannier gauge of $H(\mathbf k)$; this is a finite-dimensional matrix
contraction, not an overlap of Bloch states at different momenta with a
momentum-conserving identity. $O_A(\mathbf Q)$ is the vertex connecting those
two momentum sectors. Its full experimental transfer $\mathbf Q$ may enter
site phases and form factors even when band energies use only
$\mathbf q=\mathbf Q\bmod\mathbf G$.
The star in $M^{B*}_{nm}$ conjugates the complete matrix element with the same
$n,m,\mathbf k,\mathbf Q$. Thus the second channel is the adjoint probe
$O_B^\dagger$. For $O_A=|a\rangle\langle b|$,
$M^A_{nm}=u_{an}^*(\mathbf k)u_{bm}(\mathbf k+\mathbf q)$ and its conjugate
operator has label $(b,a)$. Arbitrary eigenvector phases cancel from
$M^A_{nm}M^{B*}_{nm}$.

The physical dissipative matrix is
$\boldsymbol\chi^{0\prime\prime}=(\boldsymbol\chi^0-\boldsymbol\chi^{0\dagger})/(2i)$,
which gives

$$
\chi^{0\prime\prime}_{AB}=
\sum_{\mathbf k,n,m}w_{\mathbf k}
\frac{[f(\epsilon_{n\mathbf k})-f(\epsilon_{m,\mathbf k+\mathbf q})]\eta}
{[E+\epsilon_{n\mathbf k}-\epsilon_{m,\mathbf k+\mathbf q}]^2+\eta^2}
M^A_{nm}M^{B*}_{nm}.
$$

A scalar or diagonal entry reduces to its ordinary imaginary part.
For complex off-diagonal elements, the displayed spectral matrix can itself
have complex entries and is not `values.imag`. `SusceptibilityResult.chi_prime`
and `.chi_double_prime` expose elementwise real and imaginary arrays. To form
the physical absorptive tensor in a script, use
`(values - values.conj().swapaxes(-1, -2)) / (2j)` on
`values_per_meV_cell`. The unpolarized real symmetric projector gives the same
scalar intensity when contracted with the elementwise imaginary array.

This convention gives positive diagonal $\chi^{0\prime\prime}$ for a
positive-energy absorption process. The response obeys the corresponding
causal frequency and $\mathbf q\leftrightarrow-\mathbf q$ conjugation
relation; a Hermitian scalar response at a symmetry-equivalent wavevector has
$\chi^0(-E)=\chi^0(E)^*$.

### Degenerate transitions and the static limit

At $E=0$ a transition with
$\epsilon_{n\mathbf k}=\epsilon_{m,\mathbf k+\mathbf q}$ would give an
undefined ratio $0/0$ in the unbroadened static formula. At finite $\eta$
the numerator instead vanishes; nfit explicitly replaces that broadened
value by the thermodynamic static limit. Two details
matter when interpreting a static response:

- The substitution applies only where $|E|\le10^{-14}$ meV **and**
  $|\epsilon_{n\mathbf k}-\epsilon_{m,\mathbf k+\mathbf q}|\le10^{-10}$ meV.
  It is the exact degeneracy limit, not a small-denominator regularization.
- At $T>0$ the limit is $-\partial f/\partial\epsilon=f(1-f)/k_BT$, evaluated
  without $\eta$. At $T=0$ that derivative is a delta function, so nfit uses
  the Lorentzian $\eta/\{\pi[(\epsilon-\mu)^2+\eta^2]\}$ instead; the two branches
  therefore describe the Fermi window with different widths, and a $T=0$ static
  response depends on $\eta$ where a finite-temperature one does not.

Away from exact degeneracy the finite $\eta$ suppresses near-degenerate
intraband weight whenever
$|\epsilon_{n\mathbf k}-\epsilon_{m,\mathbf k+\mathbf q}|\ll\eta$, which is the
usual small-$\mathbf q$ artefact of a broadened Lindhard function. Approach the
uniform limit by evaluating exactly at $\mathbf q=0$ rather than at a small
finite $\mathbf q$, and check the broadening convergence scan.

For the unweighted total spin of an implicit-spin model,
$\chi^{0}_{s,\alpha\alpha}(\mathbf0,0)=D_{\uparrow,T}(\mu)/2$, where
$D_{\uparrow,T}(\mu)=\sum_{\mathbf k,n}w_{\mathbf k}[-f'(\epsilon_{n\mathbf k})]$
is the thermally smeared per-spin DOS in states/(meV cell). At $T=0$, the
implemented derivative uses the Lorentzian of width $\eta$ above. The familiar
$D_\uparrow(\mu)/2$ means the zero-temperature, converged zero-width limit;
see
[Physics conventions](physics_conventions.md#electronic-response-conventions).

`orbital_pair_operator_basis(model)` constructs the complete ordered
$|a\rangle\langle b|$ basis and its conjugate map.
`bare_lindhard_susceptibility` evaluates any declared one-particle operator
basis. The model component uses `bare_spin_susceptibility`, which projects
directly onto Cartesian $S_x$, $S_y$, and $S_z$.

## Spin and neutron projection

For an extended-zone transfer $\mathbf Q$, the magnetic operator contains the
orbital-center phase

$$
S_\alpha(\mathbf Q)=
\sum_{ab}
e^{2\pi i\mathbf Q\cdot\mathbf r_a}
(S_\alpha)_{ab}|a\rangle\langle b|.
$$

Here $\mathbf r_a$ is fractional orbital center $a$ (also called
$\boldsymbol\tau_a$ on the tight-binding page), so the phase is dimensionless.
$S_\alpha$ is the dimensionless electron-spin matrix defined in
[Spin operators](physics_conventions.md#spin-operators-and-equilibrium-averages).
For implicit spin,
$\operatorname{Tr}_{\rm spin}[(\sigma_\alpha/2)(\sigma_\beta/2)]
=\delta_{\alpha\beta}/2$: the one-spin orbital bubble receives a factor 1/2
per Cartesian component, not the factor 2 used for total electronic DOS.

The Hamiltonian is evaluated at $\mathbf q=\mathbf Q\bmod\mathbf G$, but the
operator retains the full $\mathbf Q$. An explicit spinor model supplies its
spin matrices. A spin-independent model uses the equivalent isotropic
spin-degenerate response without expanding the Hamiltonian to a redundant
spinful basis.

For neutron scattering, nfit contracts the Cartesian tensor with

$$
P_{\alpha\beta}(\mathbf Q)
=\delta_{\alpha\beta}-\widehat Q_\alpha\widehat Q_\beta .
$$

Indices $\alpha,\beta=x,y,z$ refer to the crystal Cartesian frame,
$\delta_{\alpha\beta}$ is the identity tensor, and
$\widehat{\mathbf Q}=\mathbf Q_{\rm cart}/|\mathbf Q_{\rm cart}|$ is formed
from physical inverse-angstrom coordinates, not by normalizing raw HKL.

At $\mathbf Q=0$, nfit uses the angular-average convention
$P_{\alpha\beta}=\tfrac23\delta_{\alpha\beta}$. The response configuration
supplies the magnetic form factor; the dataset's declared spectral convention
then applies the fluctuation--dissipation factor and normalization. Dataset
scale remains separate, as described in
[Physics conventions](physics_conventions.md).

## Parameters

| API name | Meaning | Unit | Acceptable input example |
| --- | --- | --- | --- |
| `broadening` | positive particle--hole lifetime broadening $\eta$ | meV | `2.0` |

The broadening regularizes the response but is not a replacement for checking
mesh convergence. It may be fitted. Onsite, hopping, and spin--orbit fit
parameters remain on the referenced tight-binding component.

## Configuration

The GUI separates routine choices from execution overrides. **Response**
selects the electronic model and occupations, **Sampling** controls the
integration and experimental-$\mathbf Q$ accuracy, and **Experimental
coupling** contains bulk-normalization choices. Orbital magnetic form factors
belong to the referenced tight-binding model. Plot-specific
coordinates and energy grids live in the corresponding viewer. The remaining
controls are under **Advanced**; they are still serialized and available to
scripts.

### Response and sampling

| Setting | Meaning | Default | Acceptable input example |
| --- | --- | --- | --- |
| `electronic_component` | enabled sibling `tight_binding` component supplying $H(\mathbf k)$; empty selects it automatically when exactly one is enabled | `""` | `"Bands"` |
| `response_mesh` | concrete full-zone integration mesh used by calculations and fits | `[16, 16, 16]` | `[48, 48, 24]` |
| `response_sampling_mode` | derive and certify a production mesh, or use a manual mesh without an automatic accuracy claim | `"automatic"` | `"automatic"` or `"manual"` |
| `response_sampling_accuracy` | normalized tensor-error profile | `"standard"` | `"preview"`, `"standard"`, `"high"`, or `"custom"` |
| `response_sampling_custom_rtol` | positive tolerance used only by the custom profile | `0.01` | `0.005` |
| `response_q_evaluation` | direct, required commensurate, validated interpolated, or automatic wavevector evaluation | `"auto"` | `"interpolated"` |
| `response_q_interpolation_rtol` | accepted relative error for automatic periodic interpolation; zero disables this criterion | `0.0` | `0.01` |
| `chemical_potential_mode` | use the source chemical potential or solve from filling | `"source"` | `"source"` or `"filling"` |
| `filling_per_cell` | electron count per electronic model cell in filling mode | `1.0` | `3.0` |
| `formula_units_mode` | infer formula units in the actual electronic model cell or use an explicit override | `"auto"` | `"auto"` or `"manual"` |
| `formula_units_per_cell` | formula units in the electronic model cell when manual mode is selected | `1.0` | `2.0` |
| `magnetic_normalization_mode` | count unique represented tight-binding sites or use a manual magnetic-center count | `"auto"` | `"auto"` or `"manual"` |
| `magnetic_normalization_species` | represented basis species or element selected for a per-magnetic-ion dataset | `""` | `"V"` or `"V4+"` |
| `magnetic_centers_per_model_cell` | reference centers per model cell in manual mode | `1.0` | `4.0` |
| `bulk_g_factor` | Landé factor for bulk conversion | `2.0` | `2.1` |

### Response normalization and magnetic form factor

The normalized Brillouin-zone sum produces an intrinsic susceptibility per
electronic model cell. The dataset comparison layer then resolves one explicit
normalization:

$$
\chi_{\rm target}=\frac{\chi_{\rm model\ cell}}
{N_{\rm target/model\ cell}}.
$$

For a per-formula-unit dataset, $N_{\rm target/model\ cell}$ is the number of
formula units represented by the actual electronic cell. For a
per-magnetic-ion dataset it is the number of unique selected tight-binding
sites. Automatic magnetic-center counting succeeds only for one represented
species, or when `magnetic_normalization_species` selects one unambiguously;
manual mode supports an effective or deliberately restricted subspace. A
dataset normalized per unit cell uses the electronic model cell. An unknown
basis retains the model-cell ordinate without making an absolute claim.

Normalization and form factors are independent. Selecting a V radial profile
on a tight-binding orbital does not select or count V sites. Conversely,
normalizing per V does not choose a V form factor. The resolved model-cell,
formula-unit, and magnetic-center counts are shown in **Experimental coupling**.

Each tight-binding basis orbital may own a radial profile. The neutron probe is

$$
S^\alpha_f(\mathbf Q)=\sum_a f_a(|\mathbf Q|)
e^{2\pi i\mathbf Q\cdot\mathbf r_a}S^\alpha_a,
$$

$f_a$ is a dimensionless signed radial amplitude evaluated at the physical
$|\mathbf Q|$ in Å$^{-1}$, and $S_a^\alpha=P_aS_\alpha$ with
$P_a=|a\rangle\langle a|$ the basis-row selector. For localized physical spin
blocks this is the local spin contribution; an imported nonlocal effective
operator uses the declared row-center convention. An omitted profile means
$f_a=1$.
Thus an orbital pair contributes $f_a f_b^*$, including its interference phase.
The intrinsic total-spin operator, with every $f_a=1$, remains separate for
Stoner and matrix-RPA denominators. Thus radial attenuation never changes the
interaction instability criterion. The orbital-density interpolation contracts
both probe and total-spin correlations by bounded point batches, and the saved
certificate validates the final form-factor-weighted probe response.

Automatic formula-unit normalization expands the complete crystallographic
cell, reduces its integer composition, and accounts for any certified
primitive-cell reduction used by the electronic Hamiltonian. It deliberately
refuses partial occupancy, mixed occupancy, missing elements, or an
incompatible model cell. A per-formula-unit neutron or bulk comparison then
requires a manual override. Older projects without `formula_units_mode` retain
their stored manual normalization.

The GUI's **Exact Q evaluation** policy is
`response_q_evaluation="auto"` with both interpolation tolerances zero.
**Validated interpolation** sets a positive relative tolerance. Direct,
commensurate-only, and forced-interpolation policies remain available as
advanced script settings.

Validated interpolation is intended for large point clouds and optimizer
iterations. nfit first evaluates a deterministic subset of the requested
off-mesh $(\mathbf Q,\hbar\omega)$ points both directly and through a periodic
orbital-density interpolation. It accepts the approximation only when every
validation residual satisfies

$$
|\chi_{\rm interp}-\chi_{\rm direct}|
\leq a_{\rm tol}+r_{\rm tol}\max_i|\chi_{{\rm direct},i}|.
$$

The selected interpolation mesh, direct validation coordinates and energies,
observed errors, electronic-model digest, attempted refinements, and a digest
of the certificate are stored in
`response_q_interpolation_certificates` for the current component state and
dataset. Changing an onsite energy, hopping, self-energy parameter,
broadening, or interaction causes a new validation during that trial; a
certificate from another Hamiltonian is never reused. `auto` falls back to an
exact full calculation when validation fails. `interpolated` instead raises an
error, which is useful when a fit must not silently take a slower path.
Scalar Stoner and user-matrix RPA dressings are checked again after the
nonlinear interaction denominator is applied, so pole enhancement cannot turn
a certified bare-response error into an uncertified dressed result.

For an implicit-spin orbital model, the fast path evaluates the full
orbital-density response on the certified periodic mesh, reuses each
node--energy pair, interpolates in bounded point chunks, and applies the
extended-zone orbital phases exactly at every requested point. The measured
points remain available for arbitrary viewer binning and integration; this is
not a visible-slice approximation. Required tolerances are model and domain
dependent. If no divisor of the integration mesh passes, increase the
integration mesh, loosen a scientifically justified tolerance, or use exact
evaluation.

### Advanced execution

| Setting | Meaning | Default | Acceptable input example |
| --- | --- | --- | --- |
| `response_mesh_shift` | offsets in mesh steps; the default half shift reduces special-point and Fermi-surface shell artifacts | `[0.5, 0.5, 0.5]` on periodic axes | `[0, 0, 0]` for a Γ-centered mesh |
| `response_sampling_max_refinements` | maximum candidate meshes in one certificate | `7` | `8` |
| `response_sampling_max_mesh_points` | independent full-mesh point budget | `500000` | `1000000` |
| `response_sampling_points_per_dataset` | maximum deterministic fit-domain representatives per applicable dataset | `32` | `48` |
| `response_sampling_certificate` | derived serialized meshes, errors, domain, and provenance | `{}` | normally written by nfit |
| `response_q_interpolation_certificates` | derived per-dataset validation certificates for the current live parameter state | `{}` | normally written by nfit |
| `response_symmetry` | full mesh, required certified reduction, or automatic reduction with recorded fallback | `"auto"` | `"auto"`, `"full"`, or `"reduced"` |
| `response_q_interpolation_atol` | accepted absolute complex-response error; zero disables this criterion | `0.0` | `1e-5` |
| `response_q_interpolation_mesh` | optional initial commensurate interpolation mesh; empty chooses automatically | `[]` | `[8, 8, 8]` |
| `response_q_validation_points` | deterministic off-mesh points calculated directly during certification | `8` | `16` |
| `response_backend` | CPU execution policy, or explicit CuPy execution | `"auto"` | `"auto"`, `"numpy"`, `"threaded"`, or `"cupy"` |
| `response_workers` | total CPU allocation; zero detects affinity, cgroup, or scheduler allocation | `0` | `0` or `8` |
| `response_transition_backend` | CPU particle--hole contraction; automatic selection uses exact orbital-pair factorization or workload-gated Numba | `"auto"` | `"auto"`, `"numpy"`, or `"numba"` |
| `response_validate_backend` | compare a deterministic probe with serial NumPy before threaded or GPU use | `true` | `false` |
| `response_backend_probe_points` | mesh points in the backend-equivalence probe | `8` | `12` |
| `response_backend_rtol` | relative eigenvalue tolerance for the probe | `1e-10` | `1e-9` |
| `response_backend_atol_meV` | absolute eigenvalue tolerance for the probe | `1e-8` | `1e-7` |
| `response_max_batch_mb` | eigensystem temporary-memory target | `256.0` | `512.0` |
| `response_transition_max_batch_mb` | particle--hole transition temporary-memory target | `256.0` | `512.0` |
| `response_cache_mb` | host and device limit for retained eigensystems, completed responses, and CuPy Hamiltonian components; zero disables this response cache | `512.0` | `1024.0` |
| `response_cache_entries` | maximum retained response-cache entries | `64` | `128` |
| `powder_orientations` | deterministic directions used automatically for a powder dataset | `50` | `96` |

The half-shifted response mesh is the general default because metallic
susceptibilities are often sensitive to whether a mesh samples a special point
or a narrow Fermi-surface shell exactly. It retains exact commensurate
wavevector permutation on nfit's even automatic mesh ladder. A Γ-centered mesh
remains available when sampling Γ itself is required by a specific numerical
comparison or convention. Nonperiodic reciprocal axes are stored with zero
shift in GUI-built lower-dimensional models.

`response_backend="auto"` stays on the reference CPU implementations and
selects serial or bounded threaded execution from the workload and available
CPU allocation. GPU execution remains an explicit `cupy` choice and is
checked against serial NumPy before use.

### Viewer calculations

These settings belong to calculation viewers rather than the model-building
form. **Apply and recalculate** validates them, stores them in the component,
and updates the existing viewer. `configure_lindhard_plot` provides the same
atomic operation in a script.

| Setting | Meaning | Default | Acceptable input example |
| --- | --- | --- | --- |
| `plot_q_reduced` | representative extended-zone $\mathbf Q$ for model-owned response and convergence plots | `[0.5, 0.5, 0.5]` in 3D; new linked components use `0.5` only on periodic axes | `[0.5, 0.5, 0]` |
| `plot_energy_min_meV` | lower plotted energy transfer | `-100.0` | `-50.0` |
| `plot_energy_max_meV` | upper plotted energy transfer | `100.0` | `50.0` |
| `plot_energy_points` | plotted energy samples | `401` | `501` |
| `plot_temperature_K` | plotted response temperature | `10.0` | `20.0` |
| `convergence_mesh_scales` | factors applied to each response-mesh dimension | `[0.5, 0.75, 1.0]` | `[0.5, 1.0, 1.5]` |
| `convergence_broadening_scales` | factors applied to the current $\eta$ | `[2.0, 1.0, 0.5]` | `[2.0, 1.0, 0.5]` |
| `convergence_energy_points` | representative energies across the plot window | `9` | `17` |
| `convergence_relative_floor` | denominator floor for relative changes | `1e-12` | `1e-10` |

The mesh must match the periodic dimensionality of the electronic model. A
three-entry mesh or shift is also accepted for a reduced-dimensional model;
nfit selects its periodic axes. With `response_symmetry="auto"`, nfit reduces
only an nfit-built implicit-spin response under operations that leave every
requested $\mathbf q$ fixed modulo a reciprocal vector. The result records the
little-group size and irreducible mesh. Imported models, explicit-spin
operators, orbital-pair matrices, incompatible meshes, and generic
wavevectors fall back to the full mesh. `"reduced"` turns the same fallback
into an error.

**Check/refine convergence** constructs an anisotropic mesh ladder and compares
the complete tight-binding--Lindhard--RPA observable on deterministic points
from every applicable enabled fit dataset. The calculation uses the ordinary
masked and rebinned fit inputs, dataset temperatures and fields, exact
experimental-$\mathbf Q$ evaluation, orbital form factors, neutron projection,
powder averaging, and bulk or spectral normalization. Every dataset must pass
two successive refinements. Preview, Standard, and High correspond to 5%, 1%,
and 0.2% normalized error. A successful search stores one concrete
`response_mesh`; a budget failure is reported without weakening the tolerance.
The model-owned mesh/broadening plot remains an explicit-domain inspection and
does not replace this project certificate. See
[Automatic Brillouin-zone sampling](electronic_sampling.md) for the common
certificate and the explicit-domain scripting API.

`response_q_evaluation="auto"` does not interpolate by default. A transfer
commensurate with the integration mesh reuses the base eigensystem by an exact
periodic index permutation; other transfers use direct evaluation. Setting a
positive interpolation tolerance permits periodic linear interpolation of the
complex bare response. nfit calculates only the commensurate stencil nodes
needed by the requested points, compares deterministic points with direct
calculations, and refines through divisors of the integration mesh. Failure to
meet the tolerance falls back to direct evaluation. `"interpolated"` raises
instead of falling back, while `"commensurate"` rejects an off-mesh transfer.

Interaction dressing follows interpolation. Position-dependent magnetic
operators retain the exact extended-zone $\mathbf Q$, so orbital-center
phases, form factors, and neutron polarization are not replaced by their
values at an interpolation node. The result provenance records the resolved
policy, mesh attempts, validation errors, and exact or interpolated point
counts.

Threaded execution constructs bounded waves of Hamiltonians serially before
parallel eigensolution. This prevents complex Hamiltonian assembly from
overlapping platform LAPACK calls while preserving input order. Threaded and
CuPy backends are compared with serial NumPy for each new model digest before
response evaluation. Explicit CuPy use fails clearly if the requested backend
is unavailable or exceeds the configured tolerance.

On CPU, `response_transition_backend="auto"` recognizes the ordered
$|a\rangle\langle b|$ basis used by Hubbard--Hund RPA and contracts its
separable orbital factors without constructing dense operator matrix elements.
General sufficiently large multi-operator problems use the fused Numba
kernel. Both paths retain float64/complex128 arithmetic and the same
finite-temperature equal-energy limit as the NumPy reference. Explicit
point-dependent operator matrices use the general path because they need not
retain the ordered-pair structure. Select `"numpy"` or `"numba"` to force a
reference implementation for comparison. Explicit CuPy response evaluation
remains entirely on the GPU and does not use this setting.

When a direct calculation contains several independent transferred
wavevectors, nfit may divide `response_workers` between those wavevectors and
the work within each one. The workload threshold prevents thread dispatch for
small responses. This scheduling changes neither the mesh nor the order of
the returned points; `q_parallel_execution` in the result provenance records
whether it was used.

Filling mode solves one chemical potential at each required temperature. It
therefore requires $T>0$ in the current implementation. Source mode uses
`chemical_potential_meV` from the tight-binding component.

## Calculable data

The model calculates:

- single-crystal and powder inelastic neutron scattering from
  $\chi^{0\prime\prime}_s(\mathbf Q,E)$;
- single-crystal and powder quasistatic elastic magnetic scattering from the
  static $\chi^{0\prime}_s(\mathbf Q,0)$;
- uniform bulk susceptibility; and
- magnetic moment or magnetization in linear response to an applied field.

Powder calculations average the tensor response and neutron polarization over
deterministic sphere directions selected automatically when a powder dataset
is encountered. Bulk calculations use the isotropic
$\operatorname{Tr}\boldsymbol\chi_s(\mathbf 0,0)/3$ response, convert from the
electronic model-cell normalization to a formula-unit normalization, and then
follow the bulk units on the physics-conventions page.

The bulk comparison is optional. A bare band response may be intentionally
insufficient at $\mathbf q=0$, especially near an interaction-driven magnetic
instability. That discrepancy can motivate an interaction dressing rather
than requiring the bulk dataset to be omitted.

## Scripting, fitting, and export

The calculation API is independent of Qt:

```python
import numpy as np

from nfit import bare_spin_susceptibility, k_mesh

mesh = k_mesh(model, [32, 32, 32], symmetry="full")
energy_meV = np.linspace(-40.0, 40.0, 401)
Q_reduced = np.broadcast_to([0.5, 0.5, 0.0], (energy_meV.size, 3))
result = bare_spin_susceptibility(
    model,
    Q_reduced,
    energy_meV,
    mesh,
    temperature_K=20.0,
    chemical_potential_meV=0.0,
    broadening_meV=2.0,
)
chi_prime = result.chi_prime
chi_double_prime = result.chi_double_prime
```

`SusceptibilityResult` records reduced $\mathbf q$, extended-zone $\mathbf Q$,
complex values, operator labels, normalization, thermodynamic settings, mesh,
backend, and provenance. `to_dict()` and `from_dict()` provide a portable
round trip for modest results.

The GUI model plot shows the real and imaginary isotropic response at
`plot_q_reduced`. Its right panel owns the plotted momentum, energy grid, and
temperature. The main model editor's **Sampling** tab owns the representative
certificate momentum, energy interval and count, temperature, symmetry and
mesh shift, accuracy profile, production mesh, and search budgets. **Copy
script** exports the electronic source, response settings, calculation, and
rendering calls as editable Python. The standard fit machinery saves and
exports the linked electronic and response component states, fitted
parameters, dataset bindings, and calculated fit channels.

The second model plot evaluates numerical convergence and exposes its
representative momentum, energy window, mesh scales, and broadening scales in
the same kind of right panel. It reports each mesh
against the reference mesh at the same $\eta$, and each broadening against the
reference broadening on the same mesh. `ResponseConvergenceResult` preserves
the complex samples, absolute and relative metrics, reference indices, and a
portable dictionary round trip. This separation prevents a broader linewidth
from being mistaken for wavevector-mesh convergence.

`ElectronicResponseCache` reuses immutable eigensystems and completed bare
responses. Its keys include the electronic-model digest, sampling, operators,
thermodynamic state, broadening, backend request, and execution settings.
Changing an onsite, hopping, SOC, temperature, or operator cannot reuse a
stale response. Interaction-only fits can reuse $\boldsymbol\chi^0$ directly.
Particle--hole transitions and energy transfers are evaluated in separate
deterministic memory-bounded batches.

For distributed runs, `partition_response_points` creates deterministic
contiguous work units and `merge_response_chunks` verifies their scientific
settings before ordered assembly. `response_slurm_array_script` renders an
editable scheduler wrapper; calculation and merging remain scheduler-neutral.

## Current limits

- The full band-pair sum can still be expensive for large orbital bases or
  dense meshes, although transition memory is bounded.
- Automatic Lindhard certification selects a fixed mesh before fitting; it
  does not adapt the mesh inside the optimizer. Recheck a stale certificate at
  the fitted parameters. Fermi-surface topology still requires a separate
  manual density check.
- Automatic symmetry reduction is intentionally unavailable for imported
  models, explicit-spin responses, and orbital-pair response matrices until
  their operator transformations can be certified.
- The chunk API does not submit jobs or manage a cluster filesystem; those
  policies remain in editable launcher scripts.
- This component is the bare bubble with finite lifetime broadening. Separate
  scalar Stoner, matrix, and Hubbard--Hund RPA components can dress it;
  self-energy and superconducting extensions are not yet included.
- The basis is orthonormal. Nonorthogonal overlap matrices are not supported.

## References

- J. Lindhard, *Kgl. Danske Videnskab. Selskab, Mat.-Fys. Medd.* **28**,
  no. 8 (1954).
- S. Graser *et al.*, *New J. Phys.* **11**, 025016 (2009),
  [doi:10.1088/1367-2630/11/2/025016](https://doi.org/10.1088/1367-2630/11/2/025016).
