# Spin-fluctuation models

This page documents the physics models for itinerant, nearly magnetically
ordered systems: a fully local relaxational spin, the Millis–Monien–Pines
(MMP) nearly-antiferromagnetic form, and a Heisenberg-coupled RPA model whose
exchange constants live on symmetry-distinct bond orbits. All three compute
the imaginary part of the dynamic susceptibility $\chi''(\mathbf{Q}, E)$ and
convert it to measured intensity through the shared cross-section convention.

## From susceptibility to intensity

Every model in this family produces

$$
I(\mathbf{Q}, E) = s\, \tfrac{2}{3}\, |f(Q)|^2\,
\frac{\chi''(\mathbf{Q}, E)}{1 - e^{-E/k_B T}},
$$

where

- $s$ is a fittable overall **scale** for unnormalized data,
- $2/3$ is the polarization (orientation) factor for **isotropic (Heisenberg)
  spins**: unpolarized neutrons couple only to spin components perpendicular
  to $\mathbf{Q}$, and the isotropic average of
  $\sum_{\alpha} (1 - \hat{Q}_\alpha^2)$ is $2/3$,
- $|f(Q)|^2$ is the magnetic form factor (below),
- $[1 - e^{-E/k_B T}]^{-1}$ is the detailed-balance (Bose) factor connecting
  $\chi''$ to $S(\mathbf{Q}, E)$; the odd-in-$E$ $\chi''$ times this factor
  satisfies detailed balance automatically.

The **temperature** is read from each dataset (`PointData4D.temperature`).
Fitting a spin-fluctuation model against a dataset without a valid temperature
raises an error naming the fix: set the per-dataset temperature in the GUI
(dataset details, "T (K)") or import data carrying temperature metadata.
Backgrounds are separate additive components (`constant_background`,
`linear_background`), not part of these models.

## Magnetic form factor

The `nfit.form_factors` module tabulates the $\langle j_0 \rangle$
analytic approximation

$$
f(s) = A e^{-a s^2} + B e^{-b s^2} + C e^{-c s^2} + D, \qquad
s = \frac{|Q|}{4\pi} \ \text{in Å}^{-1},
$$

for the common $3d$, $4d$, $4f$, and $5f$ ions, keyed by labels such as
`"Mn2"`, `"Fe2"`, `"Yb3"`, `"U4"`. In the GUI, select a tabulated entry from
the form-factor dropdown, or choose `Custom...` to show the coefficient field
for explicit $(A, a, B, b, C, c, D)$ values from the ILL tables.
Computing $|Q|$ in Å$^{-1}$ requires lattice metadata on the fit points; the
GUI attaches the group's lattice parameters automatically.

Sources: P. J. Brown, *International Tables for Crystallography*, Vol. C,
Section 4.4.5; the ILL form-factor tables
(<https://www.ill.eu/sites/ccsl/ffacts/>); coefficients transcribed from the
public-domain `periodictable` package.

## Model 1: local relaxational spin (`local_relaxational`)

A totally local spin relaxing at rate $\Gamma$:

$$
\chi''(E) = \chi_{\mathrm{loc}}\, \frac{\Gamma E}{E^2 + \Gamma^2}.
$$

$\chi''$ peaks at $E = \Gamma$; there is no $\mathbf{Q}$ dependence beyond the
form factor, so the model also applies to powder data. `scale` and
`chi_loc` are exactly degenerate — fix one.

## Model 2: MMP relaxational (`mmp_relaxational`)

The phenomenological susceptibility of a nearly antiferromagnetic metal
[Millis, Monien & Pines 1990; Monthoux & Pines 1993]:

$$
\chi(\mathbf{q}, \omega) =
\frac{\chi_{\mathrm{pk}}}{1 + \xi^2 |\mathbf{q} - \mathbf{Q}_0|^2
- i \omega/\omega_{\mathrm{sf}}},
$$

$$
\chi''(\mathbf{q}, E) = \chi_{\mathrm{pk}}\,
\frac{E/\omega_{\mathrm{sf}}}
{\left(1 + \xi^2 |\mathbf{q} - \mathbf{Q}_0|^2\right)^2
+ (E/\omega_{\mathrm{sf}})^2},
$$

with correlation length $\xi$ in Å, $|\mathbf{q} - \mathbf{Q}_0|$ in
Å$^{-1}$ (computed from HKL through the lattice matrix), and
spin-fluctuation energy $\omega_{\mathrm{sf}}$ in meV. At $\mathbf{Q}_0$ the
lineshape is relaxational with $\Gamma = \omega_{\mathrm{sf}}$; away from
$\mathbf{Q}_0$ the response broadens and weakens. This form describes, e.g.,
the normal-state response of optimally doped BaFe$_{1.85}$Co$_{0.15}$As$_2$
[Inosov et al. 2010].

## Model 3: Heisenberg RPA (`heisenberg_rpa`)

The centerpiece model couples local relaxational spins through real-space
Heisenberg exchange in the random phase approximation.

Terms and symbols used below:

- **Heisenberg exchange** is an isotropic pair interaction between spins; each
  fitted exchange constant is a scalar $J$ in meV rather than a spin-space
  tensor.
- **RPA** means the random phase approximation: the local susceptibility is
  dressed by repeated exchange scattering, producing the matrix inverse below.
- $\mathbf{Q} = (H,K,L)$ is momentum transfer in reciprocal lattice units
  (r.l.u.); $E = \hbar\omega$ is energy transfer in meV, with positive $E$
  denoting neutron energy loss.
- $N$ is the number of magnetic sites in the crystallographic cell used for the
  model; $a$ and $b$ label those sites, and $\mathbf{r}_a$ is the fractional
  position of site $a$ in that cell.
- $\chi_0(\omega)$ is the complex single-site dynamic susceptibility.
  $\chi_0$ without the argument is its static amplitude in meV$^{-1}$, and
  $\Gamma_0$ is the bare local relaxation energy in meV.
- $J(\mathbf{Q})$ is the $N \times N$ exchange matrix obtained by Fourier
  transforming real-space bonds. A **bond** connects site $a$ to site $b$ in a
  cell displaced by the integer lattice vector $\mathbf{n}$.
- A **bond orbit** is a symmetry-equivalent set of bonds that share one fitted
  exchange value, labeled `J1`, `J2`, and so on. $J_{\text{bond}}$ is the
  exchange value assigned to the orbit containing a particular bond.
- $U$ is the unitary eigenvector matrix of $J(\mathbf{Q})$, meaning
  $U^\dagger U = \mathbf{1}$, where $\dagger$ denotes complex-conjugate
  transpose and $\mathbf{1}$ is the identity matrix. $\Lambda$ is the diagonal
  eigenvalue matrix; $\lambda_\nu$ is the eigenvalue of mode $\nu$.
- $w_\nu(\mathbf{Q})$ is the neutron structure-factor weight of mode $\nu$.
  It tells how strongly that mode contributes to the measured unpolarized
  magnetic intensity before the common form-factor, Bose-factor, and scale
  terms from the top of this page are applied.

The single-site dynamic susceptibility is

$$
\chi_0(\omega) = \frac{\chi_0}{1 - i\omega/\Gamma_0},
$$

and the RPA sums the exchange to all orders:

$$
\chi(\mathbf{Q}, \omega) =
\left[\mathbb{1} - \chi_0(\omega)\, J(\mathbf{Q})\right]^{-1} \chi_0(\omega).
$$

### Exchange Fourier transform and phase convention

With the definitions above, $J(\mathbf{Q})$ is the $N \times N$ Hermitian
matrix

$$
J(\mathbf{Q})_{ab} = \sum_{\text{bonds } (a, b, \mathbf{n})}
J_{\text{bond}}\,
e^{2\pi i\, \mathbf{Q} \cdot (\mathbf{r}_b + \mathbf{n} - \mathbf{r}_a)},
$$

This is the "extended zone" convention: the sublattice offsets
$\mathbf{r}_b-\mathbf{r}_a$ are included inside the phase factors, as in
Sunny.jl. Each physical bond is stored once; its Hermitian conjugate, the
complex-conjugate transpose needed to make $J(\mathbf{Q})$ Hermitian, is added
automatically.

### Mode decomposition

Diagonalizing $J(\mathbf{Q}) = U \Lambda U^\dagger$ decouples the RPA into
relaxational modes:

$$
\chi''(\mathbf{Q}, E) = \sum_\nu w_\nu(\mathbf{Q})\,
\chi_{\mathbf{Q}\nu}\, \frac{\Gamma_\nu E}{E^2 + \Gamma_\nu^2},
$$

$$
\chi_{\mathbf{Q}\nu} = \frac{\chi_0}{1 - \lambda_\nu(\mathbf{Q})\, \chi_0},
\qquad
\Gamma_\nu = \Gamma_0 \left[1 - \lambda_\nu(\mathbf{Q})\, \chi_0\right],
$$

where $\chi_{\mathbf{Q}\nu}$ is the static susceptibility of mode $\nu$ at
$\mathbf{Q}$ and $\Gamma_\nu$ is that mode's exchange-renormalized relaxation
energy. The entry $U_{a\nu}$ is the amplitude of mode $\nu$ on magnetic site
$a$. The neutron structure-factor weights are

$$
w_\nu(\mathbf{Q}) = \frac{1}{N}
\left| \sum_a U_{a\nu}(\mathbf{Q}) \right|^2,
\qquad \sum_\nu w_\nu(\mathbf{Q}) = 1 .
$$

The weight uses the *uniform* sublattice sum, **not** a site-phase-weighted
sum: because the extended-zone $J(\mathbf{Q})$ above already carries the full
physical pair phases $e^{2\pi i \mathbf{Q}\cdot(\mathbf{r}_b+\mathbf{n}-\mathbf{r}_a)}$,
adding site phases $e^{2\pi i \mathbf{Q}\cdot\mathbf{r}_a}$ in the weight would
double-count the sublattice offsets. With this pairing the observable is
**exactly independent of the cell description**: the same physical lattice
described with a doubled cell (twice the sites, half the r.l.u. unit) or a
shifted origin yields identical $\chi''$. The test suite locks this with a
cell-equivalence test; the weight sum rule alone does *not* (it holds for any
phase vector whose entries all have magnitude one), which is why cell
invariance is the decisive phase-convention check. The sum rule itself holds
because $U$ is unitary.

### Physics and conventions

- **Sign convention:** positive $J$ favors ordering at the wavevector where
  $\lambda_{\max}(\mathbf{Q})$, the largest eigenvalue of $J(\mathbf{Q})$, is
  largest. For example, a single positive nearest-neighbor $J$ on a Bravais
  lattice (one lattice point per primitive cell) favors $\mathbf{Q} = 0$,
  ferromagnetism, where spins align uniformly; negative $J$ favors the zone
  boundary, the edge of the first Brillouin zone.
- **Stoner-like criterion:** the static susceptibility of mode $\nu$ diverges
  when $\lambda_\nu(\mathbf{Q})\, \chi_0 \to 1$. Evaluation raises an error on
  the ordered side ($1 - \lambda \chi_0 \le 0$); during optimization such
  trial parameters are mapped to a huge misfit so the fit stays in the
  paramagnetic region, where no long-range magnetic order has formed.
- **Critical slowing:** $\Gamma_\nu = \Gamma_0 (1 - \lambda_\nu \chi_0)$
  softens toward the incipient ordering vector — the standard relaxational
  phenomenology of nearly ordered itinerant magnets [Moriya 1985; Bernhoeft &
  Lonzarich 1995]. In the limit of all $J = 0$ the model reduces exactly to
  `local_relaxational`.

### Limiting cases

Several simple limits are useful for checking a fit or choosing starting
parameters.

**No exchange.** If every exchange parameter is zero, $J(\mathbf{Q}) = 0$ and
all RPA denominators are one. The model reduces exactly to the local
relaxational form

$$
\chi''(E) =
\chi_0\, \frac{\Gamma_0 E}{E^2 + \Gamma_0^2}.
$$

The only remaining $\mathbf{Q}$ dependence in measured intensity is from the
magnetic form factor and any experimental coverage or masking.

**One magnetic site per primitive cell.** For a Bravais-lattice model,
$J(\mathbf{Q})$ is a scalar rather than a matrix, so the response is one
renormalized relaxational mode:

$$
\chi''(\mathbf{Q}, E) =
\chi_{\mathbf{Q}}\,
\frac{\Gamma_{\mathbf{Q}} E}{E^2 + \Gamma_{\mathbf{Q}}^2},
$$

$$
\chi_{\mathbf{Q}} =
\frac{\chi_0}{1 - J(\mathbf{Q})\chi_0},
\qquad
\Gamma_{\mathbf{Q}} =
\Gamma_0 \left[1 - J(\mathbf{Q})\chi_0\right].
$$

For a one-dimensional nearest-neighbor chain in this convention,
$J(H) = 2J\cos(2\pi H)$.

**Approach to magnetic order.** As the largest mode approaches the
Stoner-like criterion from the paramagnetic side,

$$
\lambda_{\max}(\mathbf{Q})\chi_0 \to 1^-,
$$

the corresponding static susceptibility diverges and its relaxation energy
softens to zero. At or beyond the instability
($1 - \lambda_\nu \chi_0 \le 0$), direct evaluation raises an error; during a
fit, those trial parameters are assigned a large finite misfit so the
optimizer is steered back to the paramagnetic side.

**Zero and negative energy transfer.** Because the model computes
$\chi''$, the susceptibility is odd in energy:

$$
\chi''(\mathbf{Q}, 0) = 0,
\qquad
\chi''(\mathbf{Q}, -E) = -\chi''(\mathbf{Q}, E).
$$

The measured intensity also includes the Bose detailed-balance factor, so the
small-$E$ intensity can remain finite even though $\chi''$ itself vanishes at
$E = 0$.

**Low-energy paramagnetic response.** For small positive $E$ while all RPA
denominators remain positive,

$$
\chi''(\mathbf{Q}, E) \approx
\frac{\chi_0 E}{\Gamma_0}
\sum_\nu
\frac{w_\nu(\mathbf{Q})}
{\left[1 - \lambda_\nu(\mathbf{Q})\chi_0\right]^2}.
$$

This squared denominator is why low-energy cuts emphasize the incipient
ordering vector so strongly.

**High-energy tail.** When $|E|$ is much larger than every renormalized
relaxation energy $\Gamma_\nu$, the leading tail becomes

$$
\chi''(\mathbf{Q}, E) \sim \frac{\chi_0 \Gamma_0}{E},
$$

independent of exchange to leading order. This follows from
$\chi_{\mathbf{Q}\nu}\Gamma_\nu = \chi_0\Gamma_0$ and the weight sum rule
$\sum_\nu w_\nu(\mathbf{Q}) = 1$.

**Equivalent cell descriptions.** A shifted origin, a doubled cell, or the
automatic primitive-cell reduction should not change the observable
$\chi''$ when the bond network represents the same physical lattice. The
extended-zone phase convention and uniform sublattice weight above are chosen
to make this invariance exact.

### Symmetry-distinct bond orbits

Exchange constants are defined per **bond orbit**: the set of bonds mapped
onto each other by the **space group**, the crystal's rotational, mirror,
inversion, and translational symmetries. Orbits are generated from the crystal
up to a cutoff distance, the maximum bond length included in the model, and
labeled `J1`, `J2`, ... by increasing bond length. The required crystal
information is the lattice, meaning the periodic translation vectors, the space
group, and the magnetic **Wyckoff sites**, which are symmetry-generated
positions occupied by the magnetic ion. When symmetry-inequivalent orbits,
orbits not related by any space-group operation, occur at the *same* distance
they get letter suffixes — on the pyrochlore lattice the twelve third neighbors
split into `J3a` and `J3b` (six bonds each per site), which are independent fit
parameters, following the Sunny.jl convention. Each orbit label becomes one fit
parameter of the component, with the same sharing/limit/constraint machinery as
any other parameter.

### Primitive-cell reduction (automatic, exact)

For a centered lattice, the conventional cell is larger than the primitive cell
and repeats magnetic sites under **centering translations**: fractional lattice
translations that map the lattice onto itself. The pyrochlore in the
conventional cubic $Fd\bar{3}m$ cell has 16 sites, but only 4 are
translationally distinct. Under such a translation $J(\mathbf{Q})$
block-diagonalizes into sectors, independent matrix blocks, at
$\mathbf{Q}, \mathbf{Q}+\mathbf{G}_1, \dots$, where each $\mathbf{G}_i$ is a
reciprocal vector of the reduced primitive description. The uniform neutron
weight vector lies **entirely in the untranslated sector**; the folded sectors,
the other blocks created by reducing the cell, carry exactly zero weight.
Evaluating with one representative site per translation class, the set of sites
related by centering translations, and fractional bond offsets such as
$(\tfrac12,\tfrac12,0)$ therefore gives bit-identical $\chi''$ at a fraction of
the cost: the per-$\mathbf{Q}$ eigendecomposition, or diagonalization of
$J(\mathbf{Q})$, scales as $N^3$. Reducing 16 sites to 4 sites is therefore a
$\sim$64× reduction in the dominant kernel, the eigendecomposition step that
dominates runtime, plus 16× less phase-array memory, the stored complex phase
factors for all bonds and $\mathbf{Q}$ points.

This happens automatically at evaluation time. The reduction is detected
empirically from the site and bond lists (candidate translations are pairwise
site differences that map both the site set and every labeled bond orbit onto
themselves) — never from the space-group symbol — so a hand-edited network that
breaks the translation symmetry, or a genuinely primitive cell, is left
untouched. **Everything the user sees stays in the specified cell:** $\mathbf{Q}$
axes, positions, bond tables, $J$ labels, and fitted parameters are unchanged;
only the internal evaluation uses the smaller basis of magnetic sites
(`nfit.spin_fluctuations.reduce_site_network`).

### Analytic Jacobian (resolvent form)

The **Jacobian** is the matrix of model derivatives with respect to fitted
parameters that the least-squares optimizer uses. The mode sum above is
equivalently the **resolvent** contraction, meaning the same response written
through the inverse RPA denominator matrix rather than through explicit
eigenmodes:

$$
\chi(\mathbf{Q}, \omega) = \frac{\chi_0(\omega)}{N}\,
\boldsymbol{\phi}^\dagger \big[\mathbf{1} - \chi_0(\omega) J(\mathbf{Q})\big]^{-1}
\boldsymbol{\phi},
\qquad \chi_0(\omega) = \frac{\chi_0}{1 - i\omega/\Gamma_0},
$$

Here $\boldsymbol{\phi}$ is the uniform sublattice vector (one equal entry for
each magnetic site), $\operatorname{Im}$ means imaginary part, and
$\chi'' = \operatorname{Im}\chi$. Writing
$A = \mathbf{1} - \chi_0(\omega)J$, where $A$ is the RPA denominator matrix,
$x = A^{-1}\boldsymbol{\phi}$, and
$z = A^{-\dagger}\boldsymbol{\phi}$, where $-\dagger$ means inverse Hermitian
conjugate, every parameter derivative is a closed-form sandwich that reuses the
single eigendecomposition and contains **no eigenvector derivatives** — so it is
exact even where $J(\mathbf{Q})$ has degenerate (flat) bands, which the
pyrochlore has generically:

$$
\frac{\partial \chi}{\partial J_o} =
\frac{\chi_0(\omega)^2}{N}\, z^\dagger P_o(\mathbf{Q})\, x,
\qquad P_o = \frac{\partial J}{\partial J_o},
$$

Here $J_o$ is the fitted exchange value for orbit $o$, and $P_o(\mathbf{Q})$ is
that orbit's precomputed structure matrix, i.e. the part of $J(\mathbf{Q})$
multiplied by $J_o$. The derivatives
$\partial\chi/\partial\chi_0$ and $\partial\chi/\partial\Gamma_0$ follow from
the chain rule through $\chi_0(\omega)$ and $A$. Because every tensor extension
of the model (single-ion anisotropy, anisotropic/tensor exchange,
dipole–dipole, Zeeman coupling to a field; see
[Tensor (anisotropic) interactions](#tensor-anisotropic-interactions)) enters
$J(\mathbf{Q}) = \sum_p \theta_p P_p(\mathbf{Q})$ as a parameter $\theta_p$
times a precomputed structure matrix $P_p(\mathbf{Q})$, the
$\partial/\partial J_o$ formula generalizes to those terms verbatim (in tensor
mode the current implementation uses central-difference gradients; the scalar
path keeps the analytic Jacobian). When every
model component on a dataset supplies its gradients, the optimizer uses the
exact Jacobian (`heisenberg_rpa_chipp_and_gradients`) instead of finite
differences, the numerical derivative method that perturbs one parameter at a
time. This cuts a least-squares iteration from $1 + n_{\text{param}}$
evaluations to $\sim$2 and improves convergence; otherwise it falls back
silently.

### Units

| Quantity | Unit |
| --- | --- |
| $E$, $\Gamma_0$, $\omega_{\mathrm{sf}}$, $J_i$ | meV |
| $\chi_0$, $\chi_{\mathrm{loc}}$, $\chi_{\mathrm{pk}}$ | meV$^{-1}$ (up to the intensity normalization) |
| $\xi$ | Å |
| $H, K, L$ | r.l.u. |
| $T$ | K |

$J \chi_0$ is dimensionless, so the instability criterion is unit-free.

### Tensor (anisotropic) interactions

The scalar Heisenberg model above generalizes to the full anisotropic
interaction set by promoting $J(\mathbf{Q})$ to a $3N\times 3N$ Hermitian matrix
carrying Cartesian spin indices $\alpha,\beta$ on the $N$ magnetic sublattices:

$$
\mathbb{J}(\mathbf{Q})_{(a\alpha),(b\beta)} = \sum_p \theta_p\,
P_p(\mathbf{Q})_{(a\alpha),(b\beta)},
\qquad
\chi(\mathbf{Q},\omega) = \bigl[\mathbb 1 - \chi_0(\omega)\,\mathbb{J}(\mathbf{Q})\bigr]^{-1}\chi_0(\omega).
$$

Each contribution is a fitted coefficient $\theta_p$ times a precomputed
structure matrix. When no tensor section is configured the evaluator runs the
*exact* scalar path unchanged (bit-identical, no cost). The terms are:

- **Anisotropic exchange** — for each bond orbit the symmetry engine projects
  the rank-2 spin–spin tensor onto the invariant subspace of the representative
  bond's stabilizer (`symmetry_allowed_exchange_basis`), removes the isotropic
  part (that stays the scalar Heisenberg parameter), and classifies the
  remainder as symmetric–traceless (`S1, S2, …`) or antisymmetric/
  Dzyaloshinskii–Moriya (`D1, …`). Each bond $b$ in the orbit contributes
  $R_g\,T_c\,R_g^{\mathsf T}$ (transposed when the generating op reverses the
  bond), with $R_g$ the Cartesian rotation of the op mapping the representative
  bond to $b$. Rank-2 tensors transform with $R$ even for improper ops (the two
  axial-vector $\det R$ factors cancel). Pyrochlore nearest neighbours allow one
  isotropic + two symmetric + one DM component (Ross *et al.* 2011).
- **Single-ion anisotropy (SIA)** — a $\mathbf{Q}$-independent on-site diagonal
  block per site class, from the symmetric-traceless invariants of the site
  point group (`symmetry_allowed_sia_basis`, coefficients `K1, …`), rotated to
  each site by its recorded generator. Cubic site symmetry allows none; only
  the rank-2 (quadratic) part is kept — higher-order anisotropy is beyond
  soft-spin RPA at Gaussian level.
- **Dipole–dipole** — one strength $D_{\mathrm{dip}}$ times the Ewald-summed
  dipole tensor $\mathfrak D(\mathbf{Q})$ (`nfit.dipole.ewald_dipole_tensor`;
  real-space erfc + reciprocal Gaussian + self term, tinfoil boundary). The
  default of $D_{\mathrm{dip}}$ is the physical $(\mu_0/4\pi)(g\mu_B)^2$ in
  meV·Å³; pin it (vary off) to keep the physical value or fit it. The tensor is
  cached densely per dataset geometry.
- **Zeeman (applied field)** — see below; makes $\chi_0(\omega)$ a per-site
  gyrotropic $3\times3$ tensor in the field frame.

**Intensity and polarization.** The evaluator forms the dissipative tensor
$\chi''_{\alpha\beta} = (\chi_{\alpha\beta} - \chi^*_{\beta\alpha})/2i$ and
contracts it against a per-channel weight matrix — it never collapses to a
scalar internally, so polarized channels can be added later without a kernel
rewrite. The only user-facing channel now is unpolarized,
$W = \tfrac13(\delta_{\alpha\beta} - \hat Q_\alpha \hat Q_\beta)$. The $1/3$
normalization makes the isotropic limit **exactly** equal the scalar model's
$\tfrac23\chi''$, so enabling an $\varepsilon$-small anisotropy produces no
intensity jump (locked by tests). See
[physics_conventions](physics_conventions.md) for the sign and frame
conventions.

**Two evaluation tiers.** With $B=0$ the local propagator is scalar and
$\chi(\mathbf{Q},\omega)$ follows from one Hermitian eigendecomposition of the
$3N\times3N$ $\mathbb{J}(\mathbf{Q})$ per unique $\mathbf{Q}$ (Tier A, covers
exchange + SIA + dipole). With the Zeeman term on, $X_0(\omega)$ is gyrotropic
and no longer commutes with $\mathbb{J}$'s eigenbasis, so each fitted point takes
a batched LU solve of $\mathbb 1 - X_0(\omega)\mathbb{J}(\mathbf{Q})$ (Tier B).
Tier B at $B\to0$ reduces to Tier A to machine precision (tested).

**Zeeman propagator.** In the field frame $\hat z = \hat B$ (uniform field), the
per-site local response is longitudinal $\chi_\parallel/(1 - i\omega/\Gamma_\parallel)$
along $\hat z$ and transverse circular $\chi_\perp/(1 - i(\omega \mp \omega_L)/\Gamma_\perp)$
in the plane, with Larmor frequency $\omega_L = g\,\mu_B\,B$ and
$\mu_B = 0.05788\,\text{meV/T}$. Fitted parameters are `g_factor` (default 2),
`chi_perp_ratio`, and `gamma_perp_ratio` (both default 1). The term needs a
valid per-dataset field (`Dataset details → Sample environment`, or
`dataset.parameters["magnetic_field"]`); a missing field raises a clear error.

**Primitive-cell reduction.** Pure lattice translations do not rotate spins, so
bond-resolved anisotropic exchange and on-site SIA fold onto the primitive cell
*exactly* (pyrochlore 16 → 4; tested to machine precision) carrying their
rotations and generators. The dipole Ewald sum depends on the Bravais lattice
and would need the primitive lattice vectors, so reduction is declined whenever
the dipole term is active.

**Enabling in the GUI.** In the `heisenberg_rpa` structured editor, generate the
symmetry bond orbits, then use the **Interactions** box to toggle anisotropic
exchange, single-ion anisotropy, dipole–dipole, and Zeeman. Enabling a term
snapshots the symmetry-allowed tensor basis into the component config and adds
the corresponding fit parameters (`J1_S1`, `J1_D1`, `K1_<class>`, `D_dip`,
`g_factor`, …). The per-dataset field lives in the `Sample environment` panel of
the dataset details.

## Temperature-dependent fitting

Model parameters are shared across datasets through the standard sharing
modes. To fit a temperature series, set each dataset's temperature (GUI
dataset details or `dataset.parameters["temperature"]`), then choose
`per_dataset` (or `grouped`) sharing for the parameters expected to vary with
temperature — typically `chi0` and `gamma0` — while keeping the exchange
constants and `scale` global:

```python
from nfit import FitDatasetInput, ModelComponentSpec, compile_fit_problem

component = ModelComponentSpec(
    name="rpa",
    type="heisenberg_rpa",
    parameters={"scale": 1.0, "chi0": 0.3, "gamma0": 4.0, "J1": 0.0},
    fit_parameters={"chi0": True, "gamma0": True, "J1": True},
    sharing={
        "chi0": {"mode": "per_dataset"},
        "gamma0": {"mode": "per_dataset"},
    },
    config={
        "ion": "Yb3",
        "site_positions": [[0.0, 0.0, 0.0]],
        "orbits": [{"label": "J1", "bonds": [
            {"site_i": 0, "site_j": 0, "offset": [1, 0, 0]},
            {"site_i": 0, "site_j": 0, "offset": [0, 1, 0]},
            {"site_i": 0, "site_j": 0, "offset": [0, 0, 1]},
        ]}],
    },
)
compiled = compile_fit_problem(
    [component],
    [FitDatasetInput("T2K", points_2K), FitDatasetInput("T50K", points_50K)],
)
```

The compiled problem then contains `rpa.chi0[T2K]`, `rpa.chi0[T50K]`, one
global `rpa.J1`, and each dataset's Bose factor uses its own temperature.

This per-temperature parametrization is deliberately unconstrained — the
fitted $\chi_0(T)$, $\Gamma_0(T)$ trajectories are themselves the physics
output. [Theory notes](theory_notes.md) discusses the sum rules and
self-consistency closures (spherical/Onsager, Moriya SCR, Takahashi) that can
replace the per-temperature values with one or two global parameters, and the
diagnostics for choosing between them.

For scripted orbit generation from a CIF file:

```python
from nfit import crystal_from_cif, generate_bond_orbits, orbits_to_config, sites_to_config

crystal = crystal_from_cif("pyrochlore.cif")
sites, orbits = generate_bond_orbits(crystal, ["Yb1"], cutoff_angstrom=7.2)
component.config["site_positions"] = sites_to_config(sites)
component.config["orbits"] = orbits_to_config(orbits)  # J1, J2, J3a, J3b, ...
```

## References

1. T. Moriya, *Spin Fluctuations in Itinerant Electron Magnetism*, Springer
   Series in Solid-State Sciences 56 (Springer, Berlin, 1985).
2. A. J. Millis, H. Monien, and D. Pines, "Phenomenological model of nuclear
   relaxation in the normal state of YBa₂Cu₃O₇", Phys. Rev. B **42**, 167
   (1990). <https://doi.org/10.1103/PhysRevB.42.167>
3. P. Monthoux and D. Pines, "YBa₂Cu₃O₇: A nearly antiferromagnetic Fermi
   liquid", Phys. Rev. B **47**, 6069 (1993).
   <https://doi.org/10.1103/PhysRevB.47.6069>
4. D. S. Inosov et al., "Normal-state spin dynamics and temperature-dependent
   spin-resonance energy in optimally doped BaFe₁.₈₅Co₀.₁₅As₂", Nat. Phys.
   **6**, 178 (2010). <https://doi.org/10.1038/nphys1483>
5. N. R. Bernhoeft and G. G. Lonzarich, "Scattering of slow neutrons from
   long-wavelength magnetic fluctuations in UPt₃", J. Phys.: Condens. Matter
   **7**, 7325 (1995). <https://doi.org/10.1088/0953-8984/7/37/006>
6. P. J. Brown, "Magnetic form factors", *International Tables for
   Crystallography*, Vol. C, Section 4.4.5; ILL tables:
   <https://www.ill.eu/sites/ccsl/ffacts/>
7. D. Dahlbom et al., Sunny.jl — symmetry-distinct bond convention:
   <https://github.com/SunnySuite/Sunny.jl>
8. K. A. Ross, L. Savary, B. D. Gaulin, and L. Balents, "Quantum excitations in
   quantum spin ice", Phys. Rev. X **1**, 021002 (2011); and J. D. Thompson
   *et al.* / K. A. Ross *et al.*, Phys. Rev. B **84**, 064430 (2011) —
   symmetry-allowed anisotropic exchange on the pyrochlore lattice.
   <https://doi.org/10.1103/PhysRevB.84.064430>
9. M. Enjalran and M. J. P. Gingras, "Theory of paramagnetic scattering in
   highly frustrated magnets with long-range dipole–dipole interactions: The
   case of the Tb₂Ti₂O₇ pyrochlore", Phys. Rev. B **70**, 174426 (2004) — RPA
   with Ewald-summed dipoles. <https://doi.org/10.1103/PhysRevB.70.174426>
10. T. Moriya, "Anisotropic superexchange interaction and weak ferromagnetism",
    Phys. Rev. **120**, 91 (1960) — Dzyaloshinskii–Moriya rules.
    <https://doi.org/10.1103/PhysRev.120.91>
