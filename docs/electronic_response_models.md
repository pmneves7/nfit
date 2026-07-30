# From bands to magnetic response

Electronic-response components turn a `tight_binding` Hamiltonian into a
complex spin susceptibility and then into the observable declared by a
dataset. The layers remain separate so the same electronic model can be tested
with different interactions.

| Component | Role | Main assumption |
| --- | --- | --- |
| [`lindhard`](lindhard.md) | bare particle--hole susceptibility $\boldsymbol\chi^0$ | independent quasiparticles with a constant lifetime broadening |
| [`stoner_rpa`](stoner_rpa.md) | scalar isotropic enhancement | one effective local spin-channel interaction |
| [`matrix_rpa`](matrix_rpa.md) | user-defined Cartesian spin interaction | a declared Hermitian vertex in the $(S_x,S_y,S_z)$ basis |
| [`hubbard_hund_rpa`](hubbard_hund_rpa.md) | local multiorbital RPA | an implicit-spin orbital model with local $U,U',J_H,J_{\rm pair}$ |

The linked tight-binding coefficients and the response parameters can be
fitted simultaneously. A dressing consumes the bare response as an observable
on its dataset scope, preventing the bare and dressed intensities from being
added together.

## 1. Resolve the electronic state

At each fit evaluation, nfit resolves the tight-binding parameters, electronic
mesh, temperature, chemical potential, and broadening $\eta$.

The chemical potential is either:

- `source`: use `chemical_potential_meV` from the tight-binding component; or
- `filling`: solve the Fermi function for `filling_per_cell` at the dataset
  temperature.

Filling is electrons per primitive electronic cell and includes the declared
spin degeneracy. The present filling solver requires positive temperature.

## 2. Calculate the bare susceptibility

For an ordered one-particle operator basis $A,B$, nfit evaluates

$$
\chi^0_{AB}(\mathbf q,E)=
-\sum_{\mathbf k,n,m}w_{\mathbf k}
\frac{
f(\varepsilon_{n\mathbf k})-
f(\varepsilon_{m,\mathbf k+\mathbf q})
}{
E+\varepsilon_{n\mathbf k}
-\varepsilon_{m,\mathbf k+\mathbf q}+i\eta
}
M^A_{nm}M^{B*}_{nm}.
$$

$w_{\mathbf k}$ are normalized integration weights,
$f$ is the Fermi function, $E$ is neutron energy transfer in meV, and
$M^A_{nm}$ contains the band eigenvectors and operator matrix. The convention
gives positive diagonal $\chi^{0\prime\prime}$ for positive-energy
absorption.

This is a bare bubble of the fitted static bands. The constant $\eta>0$
represents finite lifetime and numerical broadening but is not a
frequency-dependent self-energy. The result contains coherent particle--hole
excitations only.

For an implicit spin-degenerate model, spin traces are evaluated analytically.
For an explicit spinor model, the stored $S_x,S_y,S_z$ matrices determine the
spin response. The more general ordered orbital-pair basis is used internally
for the Hubbard--Hund vertex.

## 3. Keep reduced and experimental momentum distinct

Band energies are periodic in the reduced electronic wavevector
$\mathbf q$. Neutron scattering occurs at an extended-zone momentum
$\mathbf Q$. nfit reduces $\mathbf Q$ for the band energies while retaining
the original transfer in orbital-position phases and the neutron projection.

This distinction is essential in a multi-site cell: two transfers related by
an electronic reciprocal vector have the same band denominators but can have
different interference and polarization factors.

Dataset coordinates in r.l.u. or Å$^{-1}$ are converted through the dataset
and electronic reciprocal lattices. Powder datasets are evaluated over
deterministic directions at fixed $|\mathbf Q|$ and then averaged.

## 4. Apply an optional interaction dressing

The matrix RPA form is

$$
\boldsymbol\chi(\mathbf q,E)=
\left[
\mathbb1-\boldsymbol\chi^0(\mathbf q,E)
\boldsymbol\Gamma
\right]^{-1}
\boldsymbol\chi^0(\mathbf q,E).
$$

The multiplication order, operator ordering, interaction unit, and physical
channel are stored with the result. A singular-value threshold detects a trial
point at or too near an RPA pole.

The available dressings differ only in how
$\boldsymbol\Gamma$ is constructed:

- scalar Stoner uses one fitted $I$;
- matrix RPA multiplies a fitted scale by a fixed dimensionless Hermitian
  Cartesian matrix; and
- Hubbard--Hund builds a local spin-channel vertex on selected
  `correlated_shell` labels.

The Hubbard--Hund option may impose
$U'=U-2J_H$ and $J_{\rm pair}=J_H$. These parameters are interaction vertices,
not tight-binding onsite energies. RPA changes the two-particle response but
does not feed a Hartree--Fock self-energy back into $H(\mathbf k)$.

RPA is most credible when quasiparticle bands remain meaningful and vertex
corrections beyond the repeated local interaction are not dominant. Close to a
pole, or in a regime with strong incoherent spectral weight, an apparently
good RPA fit should not be interpreted as a controlled microscopic solution.

## 5. Project onto the experimental observable

The spin tensor is contracted with the unpolarized neutron projector

$$
\delta_{\alpha\beta}-\hat Q_\alpha\hat Q_\beta.
$$

The selected `ion` or custom coefficients supply $|f(\mathbf Q)|^2$ exactly
once. For inelastic data, nfit sends the resulting $\chi''$ through the
dataset's declared spectral convention, including the fluctuation--dissipation
population factor and absolute cross-section factors when requested.

For quasistatic elastic data, nfit evaluates the real static response and uses
the documented quasistatic conversion. For bulk susceptibility or
magnetization, it evaluates the real isotropic $\mathbf Q=0$, $E=0$ response,
divides by `formula_units_per_cell`, and applies `bulk_g_factor` and the
dataset's unit convention.

The electronic response is normalized per primitive electronic cell before
these conversions. `formula_units_per_cell` is therefore a physical
normalization input, not a fit scale. Experimental intensity scale remains a
dataset quantity and is fixed to one for normalized data or fitted through
dataset scale parameters otherwise.

Instrument resolution, finite-bin integration, absorption, and related
measurement corrections belong to the optional dataset systematics layer
immediately before comparison. They do not modify the intrinsic electronic
susceptibility.

Shared conventions for the Bose factor, form factor, polarization, elastic
limit, bulk units, and dataset scale are defined in
[Physics conventions](physics_conventions.md).

## Calculable data

The bare and dressed electronic-response components calculate:

- single-crystal inelastic neutron scattering;
- powder inelastic neutron scattering;
- single-crystal and powder quasistatic elastic magnetic scattering;
- uniform bulk susceptibility; and
- magnetization in linear response.

The $\mathbf Q=0$ prediction is available even when the model was developed
from finite-$\mathbf Q$ neutron data. Simultaneous bulk fitting is optional.
Failure at $\mathbf Q=0$ can diagnose an inadequate band or interaction
model, but a finite-$\mathbf Q$ RPA approximation need not be quantitatively
valid for the uniform response.

## Numerical validation

Response calculations support:

- bounded eigensystem and particle--hole transition batches;
- eigensystem caching keyed by the complete electronic-model digest;
- serial NumPy, bounded CPU threading, and explicit CuPy execution;
- deterministic backend comparison with serial NumPy;
- fail-closed little-group reduction for certified implicit-spin responses;
- separate mesh and broadening convergence scans; and
- scheduler-neutral response-point chunks with an editable Slurm launcher.

`response_symmetry="auto"` uses a reduced mesh only when nfit can prove that
the operator and every requested wavevector are invariant under the retained
little group. Imported models, spinor operators, orbital-pair matrices, or
generic wavevectors fall back to the full mesh. The detailed settings and
defaults are defined on the [bare Lindhard page](lindhard.md).

Mesh convergence and broadening dependence answer different questions.
Increasing $\eta$ can hide an underconverged mesh, so nfit reports these axes
separately.

## Choosing a model

Start with the bare Lindhard response to determine what follows from the band
structure alone. Add a scalar Stoner interaction when one isotropic
enhancement scale is sufficient. Use matrix RPA for a compact anisotropic or
phenomenological vertex. Use Hubbard--Hund RPA only when the correlated
orbital basis and shell labels are physically defensible.

If the data constrain only a broad peak and relaxation scale, a generalized
paramagnon may be more identifiable than a large electronic model. More
microscopic structure is useful only when the data or independent electronic
information constrain it.
