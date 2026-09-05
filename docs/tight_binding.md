# Tight-binding model

`tight_binding` represents a periodic one-electron Hamiltonian in an arbitrary
finite orthonormal basis. It can be built from a crystal and orbital model,
specified directly as real-space matrices, or imported from Wannier90. It does
not assume a particular material, lattice, orbital count, or magnetic ordering
wavevector.

Use this model when a static band Hamiltonian is an adequate starting point
for the electronic structure. Correlation effects that merely renormalize a
few bands can be absorbed into fitted onsite and hopping coefficients.
Frequency-dependent self-energies, incoherent spectral weight, nonorthogonal
bases, and superconducting Nambu Hamiltonians are not part of the implemented
model.

## Real-space definition

The ansatz retains a finite orthonormal set of localized orbitals (or Wannier
states), with $\langle\mathbf R,a|\mathbf R',b\rangle=
\delta_{\mathbf R\mathbf R'}\delta_{ab}$. A translation-invariant,
energy-independent one-electron matrix approximates the normal-state bands;
interactions beyond this fitted static matrix enter only a separate response
model. Here $\hat H$ is the one-electron Hamiltonian, $a,b=1,\ldots,N_b$
label the retained basis, and all its matrix elements have units meV.

Let $\lvert\mathbf R,b\rangle$ be basis state $b$ in the cell translated by
the integer vector $\mathbf R$. nfit defines

$$
H_{ab}(\mathbf R)
=\langle\mathbf 0,a\vert\hat H\vert\mathbf R,b\rangle .
$$

Rows therefore label destination states $a$ in the home cell and columns label
source states $b$ in cell $\mathbf R$. The cases have useful interpretations:

- $H_{aa}(\mathbf0)$ is an onsite energy;
- $H_{ab}(\mathbf0)$ with $a\ne b$ is same-cell hybridization; and
- $H_{ab}(\mathbf R\ne\mathbf0)$ is intercell hopping or hybridization.

Hermiticity requires

$$
H(-\mathbf R)=H(\mathbf R)^\dagger .
$$

nfit validates this relation. The manual constructor can create a missing
$-\mathbf R$ partner, but an explicitly inconsistent pair is an error.

Each basis state also has a fractional center $\boldsymbol\tau_a$. The
physical displacement associated with $H_{ab}(\mathbf R)$ is

$$
\mathbf d_{ab}(\mathbf R)
=A\left(\mathbf R+\boldsymbol\tau_b-\boldsymbol\tau_a\right),
$$

where the columns of $A$ are direct-lattice vectors in Å. Centers determine
hopping distances, symmetry actions, model geometry, and the position phases
of neutron-scattering operators.

## Reciprocal-space Hamiltonian and bands

With $N_c$ periodic cells, the Bloch basis is
$|a,\mathbf k\rangle=N_c^{-1/2}\sum_{\mathbf R}
 e^{2\pi i\mathbf k\cdot\mathbf R}|\mathbf R,a\rangle$.
The band coefficients obey
$|n,\mathbf k\rangle=\sum_a u_{an}(\mathbf k)|a,\mathbf k\rangle$ and
$u_n^\dagger u_m=\delta_{nm}$. For independent electrons the many-body ansatz is
$\hat H_{\rm el}=\sum_{\mathbf k,ab}c^\dagger_{a\mathbf k}H_{ab}(\mathbf k)c_{b\mathbf k}$.
$c^\dagger_{a\mathbf k}$ creates an electron in that basis state, $c_{a\mathbf k}$
annihilates it, and $\{c_{a\mathbf k},c^\dagger_{b\mathbf k'}\}
=\delta_{ab}\delta_{\mathbf k\mathbf k'}$, with other anticommutators zero.
For implicit spin an additional identical copy is understood for each of
$\uparrow,\downarrow$. This defines the independent-particle starting point
of the linked Lindhard response.

nfit uses the Wannier gauge

$$
H_{ab}(\mathbf k)=
\sum_{\mathbf R}w_{\mathbf R}H_{ab}(\mathbf R)
\exp(2\pi i\,\mathbf k\cdot\mathbf R),
$$

where $i^2=-1$, $\mathbf k$ is in dimensionless reduced coordinates and
$w_{\mathbf R}>0$ is a dimensionless interpolation weight (no unit-sum constraint). Orbital centers do not appear
again in this Fourier phase. A Wannier90 import applies the reported
Wigner--Seitz degeneracies while constructing its canonical blocks; those
imported blocks then use unit interpolation weights.

At each $\mathbf k$,

$$
H(\mathbf k)\lvert u_{n\mathbf k}\rangle
=\varepsilon_n(\mathbf k)\lvert u_{n\mathbf k}\rangle .
$$

The eigenvalues $\varepsilon_n$ are band energies in meV, $n$ is the band
index, and $|u_{n\mathbf k}\rangle$ is the normalized coefficient column
in the Bloch basis above. If
$C_{an}=\langle a\vert u_{n\mathbf k}\rangle$, the projected weight of a
basis-index group $G$ is the dimensionless fraction

$$
W_{nG}(\mathbf k)=\sum_{a\in G}|C_{an}(\mathbf k)|^2 .
$$

Projected bands, projected density of states, and projected Fermi surfaces
therefore derive their orbital character from eigenvectors, not from band
order or an energy label.

## Parameters

The immutable model separates fixed blocks from named linear terms:

$$
H(\mathbf R;\boldsymbol\theta)
=H_0(\mathbf R)+\sum_p\theta_pP_p(\mathbf R).
$$

$H_0$ is the fixed meV block, $p$ labels named coefficients, and
$P_p=\partial H/\partial\theta_p$ is a derivative matrix; it need not be a
projection operator.

`parameter_values[p]` is $\theta_p$ and `parameter_blocks[p]` is
$P_p(\mathbf R)`. The builder uses an energy-valued coefficient in meV and a
dimensionless matrix basis. Onsite, hopping, and spin--orbit terms expose
these coefficients through the common bounds, fit-selection, and
dataset-sharing machinery.

A `tight_binding` component does not directly contribute an additive dataset
observable. Its parameters enter a fit when a linked electronic-response
component calculates $\chi(\mathbf Q,E)$ or a bulk response.

## Canonical model fields

`ElectronicModel` is the resolved, immutable numerical representation.

| Field | Meaning | Example |
| --- | --- | --- |
| `direct_lattice` | $3\times3$ matrix with direct-lattice vectors as columns, in Å | `np.diag([4.0, 4.0, 12.0])` |
| `basis` | ordered nonempty `BasisState` sequence | `[BasisState("M1_d_xy")]` |
| `translations` | integer cell vectors $\mathbf R$ | `[[0,0,0], [1,0,0], [-1,0,0]]` |
| `hamiltonian_blocks` | one complex $N\times N$ $H(\mathbf R)$ in meV per translation | array of shape `(3, N, N)` |
| `interpolation_weights` | positive $w_{\mathbf R}$, one per block | `[1.0, 1.0, 1.0]` |
| `orbital_centers` | fractional direct-lattice center of each basis state | `[[0,0,0], [0.5,0.5,0]]` |
| `periodic_axes` | ordered periodic direct-lattice axes | `(0,)`, `(0,1)`, or `(0,1,2)` |
| `parameter_values` | named canonical coefficients in meV | `{"t_nn": -80.0}` |
| `parameter_blocks` | dimensionless derivative blocks with the same keys | `{"t_nn": dH_dt}` |
| `spin_operators` | optional Hermitian $S_x,S_y,S_z$ matrices | array of shape `(3, N, N)` |
| `energy_zero_meV` | recorded reference energy; it does not shift $H$ | `0.0` |
| `provenance` | JSON-compatible source and conversion record | `{"source": "manual"}` |
| `fourier_gauge` | Fourier convention, currently `"wannier"` | `"wannier"` |

Each `BasisState` has a unique `label` and optional `site`, `species`,
`orbital`, `correlated_shell`, `spin`, and JSON-compatible `metadata`.
`correlated_shell` selects local subspaces for the Hubbard--Hund dressing; it
does not itself add an interaction.

The canonical energy unit is meV. User-facing electronic inputs and plots
normally use eV and are converted once at the construction boundary. See
[Manual models and external interfaces](tight_binding_imports.md) for the unit
policy.

## Calculable data

The model supplies:

- band energies and optional basis projections along an arbitrary path;
- total and projected Gaussian-broadened density of states;
- one-dimensional constant-energy points, two-dimensional contours, and
  three-dimensional triangulated Fermi surfaces;
- the first Brillouin zone and configured high-symmetry path;
- resolved Hamiltonian, parameter, spin, onsite, and hopping matrices; and
- the band energies and eigenvectors required by a linked Lindhard response.

These are electronic-model calculations, not neutron cross sections. The path
from $H(\mathbf k)$ to an experimental prediction is described in
[From bands to magnetic response](electronic_response_models.md).

## Construction and validity checklist

Before interpreting a fit, check that:

- the basis is orthonormal and physically meaningful over the fitted energy
  window;
- the included hopping range reproduces the relevant bands and Fermi surface;
- the spin representation can express every included term;
- the mesh, broadening, and backend are numerically converged; and
- a bare-bubble or RPA response is appropriate for the correlation regime.

Agreement with neutron data does not by itself make an orbital decomposition
unique. Different tight-binding parameterizations can generate similar low-
energy bands, so band data, filling, symmetry, and chemically reasonable
bounds remain valuable constraints.

## References

- J. C. Slater and G. F. Koster, *Phys. Rev.* **94**, 1498 (1954),
  [doi:10.1103/PhysRev.94.1498](https://doi.org/10.1103/PhysRev.94.1498).
- A. A. Mostofi *et al.*, *Comput. Phys. Commun.* **178**, 685 (2008),
  [doi:10.1016/j.cpc.2007.11.016](https://doi.org/10.1016/j.cpc.2007.11.016).
- G. Pizzi *et al.*, *J. Phys.: Condens. Matter* **32**, 165902 (2020),
  [doi:10.1088/1361-648X/ab51ff](https://doi.org/10.1088/1361-648X/ab51ff).
