# Tight-binding electronic structure

`tight_binding` represents a material-independent orthonormal electronic
Hamiltonian. It accepts manual real-space models or Wannier90 Hamiltonians and
calculates bands, orbital projections, density of states, and Fermi surfaces.
It does not assume a particular lattice, material, orbital count, or spin
layout.

## Response

Let $\lvert\mathbf R,b\rangle$ be orthonormal basis orbital $b$ in the cell
translated by the integer vector $\mathbf R$, and let
$\lvert\mathbf 0,a\rangle$ be orbital $a$ in the home cell. The real-space
Hamiltonian is

$$
H_{ab}(\mathbf R)
=\langle\mathbf 0,a\vert\hat H\vert\mathbf R,b\rangle
$$

in meV. It is the matrix element from the translated copy of orbital $b$ to
orbital $a$ in the home cell. $H_{aa}(\mathbf 0)$ is an onsite energy;
off-diagonal elements of $H(\mathbf 0)$ are same-cell hybridizations; and
$H_{ab}(\mathbf R\ne\mathbf 0)$ contains intercell hopping and hybridization.
A named hopping parameter $\theta_p$ may multiply one element, a
symmetry-related group of elements, or a complete dimensionless matrix basis
$P_p(\mathbf R)$:

$$
H(\mathbf R)=H_0(\mathbf R)+\sum_p\theta_pP_p(\mathbf R).
$$

If the fractional centers of the orbitals are
$\boldsymbol\tau_a$ and $\boldsymbol\tau_b$, their physical separation for
this matrix element is

$$
\mathbf d_{ab}(\mathbf R)
=A\left(\mathbf R+\boldsymbol\tau_b-\boldsymbol\tau_a\right),
$$

where $A$ is the direct-lattice matrix whose columns are lattice vectors in Å.
The centers therefore determine orbital locations, hopping distances,
symmetry actions, and later neutron-scattering position phases. They are
stored separately and do not enter the Fourier phase below because nfit's
canonical Hamiltonian uses the Wannier gauge.

The real-space model must satisfy

$$
H(-\mathbf R)=H(\mathbf R)^\dagger.
$$

Element by element, this is
$H_{ab}(\mathbf R)=H_{ba}^*(-\mathbf R)$.

nfit validates this relation and the Hermiticity of interpolated
$H(\mathbf k)$. It does not silently repair inconsistent imported data.

The reciprocal-space Hamiltonian is

$$
H_{ab}(\mathbf k)=
\sum_{\mathbf R}w_{\mathbf R}H_{ab}(\mathbf R)
\exp(2\pi i\,\mathbf k\cdot\mathbf R),
$$

where $w_{\mathbf R}$ is an interpolation weight and the reduced wavevector
$\mathbf k$ is dimensionless. This is the Wannier gauge.

At each wavevector, the band problem is the Hermitian eigenproblem

$$
H(\mathbf k)\lvert u_{n\mathbf k}\rangle
=\varepsilon_n(\mathbf k)\lvert u_{n\mathbf k}\rangle.
$$

The eigenvalues $\varepsilon_n(\mathbf k)$ are the band energies. If
$C_{an}(\mathbf k)=\langle a\vert u_{n\mathbf k}\rangle$ is the component of
band $n$ on basis state $a$, the weight of a requested orbital group
$G$ is

$$
W_{nG}(\mathbf k)=\sum_{a\in G}|C_{an}(\mathbf k)|^2.
$$

Thus the orbital character shown on a projected band plot comes from the
eigenvectors of the same Hamiltonian whose eigenvalues define the bands. It is
not assigned from band order or energy.

## Parameters

Stage 3 has no optimizer-facing fit parameters. It does support named linear
Hamiltonian terms, but fitting those terms begins when a dataset observable is
available. All energies below are in meV.

### Basis-state fields

Each `BasisState` describes one ordered orthonormal basis vector.

| Field | Meaning | Acceptable input example |
| --- | --- | --- |
| `label` | unique basis-state name | `"M1_d_xy"` |
| `site` | crystallographic or user-defined site label | `"M1"` |
| `species` | chemical species metadata | `"Fe"` |
| `orbital` | orbital or effective-orbital name | `"d_xy"` or `"effective_1"` |
| `correlated_shell` | grouping for later interactions | `"M1_3d"` |
| `spin` | spin or spinor-component label | `"up"`; empty for a spinless basis |
| `metadata` | additional JSON-compatible annotations | `{"irrep": "t2g"}` |

Labels identify array indices and must be unique. Orbital names and shell
labels are descriptive in Stage 3; nfit does not infer degeneracy or symmetry
from their spelling.

### Canonical electronic-model fields

| Field | Meaning and convention | Acceptable input example |
| --- | --- | --- |
| `direct_lattice` | $3\times3$ matrix whose columns are direct-lattice vectors in Å | `np.diag([4.0, 4.0, 12.0])` |
| `basis` | ordered nonempty sequence of `BasisState`, dictionaries, or labels | `[BasisState("M1_d")]` |
| `translations` | canonical integer cell translations $\mathbf R$ | `[[0, 0, 0], [1, 0, 0], [-1, 0, 0]]` |
| `hamiltonian_blocks` | one complex $N\times N$ matrix $H(\mathbf R)$ per translation | an array with shape `(3, N, N)` |
| `interpolation_weights` | positive $w_{\mathbf R}$ multiplying each block | `[1.0, 1.0, 1.0]` |
| `orbital_centers` | one fractional direct-lattice position per basis state | `[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]` |
| `periodic_axes` | ordered periodic lattice-axis indices | `(0,)`, `(0, 1)`, or `(0, 1, 2)` |
| `parameter_values` | current values of named linear Hamiltonian terms | `{"t_nn": -80.0}` |
| `parameter_blocks` | derivative blocks paired one-to-one with `parameter_values` | `{"t_nn": dH_dt}` |
| `spin_operators` | optional Hermitian $S_x,S_y,S_z$ matrices | complex array with shape `(3, N, N)` |
| `energy_zero_meV` | recorded reference energy; it does not shift $H$ automatically | `0.0` |
| `provenance` | JSON-compatible source and conversion record | `{"source": "manual"}` |
| `fourier_gauge` | Fourier convention; currently fixed to the Wannier gauge | `"wannier"` |

The resolved Hamiltonian is

$$
H(\mathbf R;\boldsymbol\theta)
=H_0(\mathbf R)+\sum_p\theta_pP_p(\mathbf R),
$$

where `parameter_values[p]` is $\theta_p$ and `parameter_blocks[p]` is
$P_p(\mathbf R)$. Their product must have units of meV. The recommended
convention for an onsite or hopping energy is to store $\theta_p$ in meV and
use a dimensionless selector matrix for $P_p$. The two dictionaries must have
identical keys.

### Registered component configuration

These settings appear in a `tight_binding` model component. JSON lists and
dictionaries can be entered directly in the model editor.

| Setting | Meaning | Default | Acceptable input example |
| --- | --- | --- | --- |
| `source_path` | Wannier90 `*_hr.dat` or `*_tb.dat` filesystem path; the GUI stores an absolute path; empty for a stored manual model | `""` | `"/data/run/model_tb.dat"` |
| `model_digest` | expected SHA-256 digest of the canonical model; source reload fails if it differs | `""` | `"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"` when that is the model's actual digest |
| `model_data` | portable dictionary returned by `ElectronicModel.to_dict()` | `{}` | `model.to_dict()` |
| `crystal` | editable lattice, space group, crystallographic sites, and optional CIF provenance used by the structure-first builder | `P 1` cell with no sites | `{"lattice": {"a": 4, "b": 4, "c": 6, "alpha": 90, "beta": 90, "gamma": 90}, "spacegroup": "P 1", "sites": []}` |
| `periodic_axes` | periodic lattice axes; empty asks a Wannier import to infer them from nonzero translations | `[]` | `[0]`, `[0, 1]`, or `[0, 1, 2]` |
| `chemical_potential_meV` | chemical potential subtracted on band and DOS plots | `0.0` | `12.5` |
| `projection_groups` | plot labels mapped to zero-based basis indices | `{}` | `{"d": [0, 1, 2], "p": [3, 4]}` |
| `band_path` | ordered nodes with labels and reduced coordinates | $\Gamma$–X–M–$\Gamma$ | `[{"label": "G", "k": [0, 0, 0]}, {"label": "X", "k": [0.5, 0, 0]}]` |
| `band_points_per_segment` | interpolation intervals in each path segment | `60` | `80` |
| `dos_mesh` | uniform mesh sizes, one per periodic axis or one per lattice axis | `[40, 40, 40]` | `[80, 80]` for a two-dimensional model |
| `dos_energy_min_meV` | lower absolute energy sampled for the DOS | `-500.0` | `-250.0` |
| `dos_energy_max_meV` | upper absolute energy sampled for the DOS | `500.0` | `250.0` |
| `dos_energy_points` | number of DOS energy samples, at least two | `600` | `1000` |
| `dos_broadening_meV` | Gaussian standard deviation used for the DOS | `5.0` | `2.0` |
| `fermi_mesh` | extraction-grid sizes, each at least two | `[100, 100, 40]` | `[200, 200]` for a two-dimensional model |
| `fermi_energy_meV` | absolute target energy of the extracted constant-energy surface | `0.0` | `12.5` to extract the Fermi surface when $\mu=12.5$ meV |

For a reduced-dimensional model, a mesh may contain one size per periodic
axis. A three-entry mesh instead gives sizes in lattice-axis order, and nfit
selects the declared periodic axes. `chemical_potential_meV` changes the
displayed energy origin; it does not solve for a filling. Set
`fermi_energy_meV` equal to the chosen chemical potential for a true Fermi
surface.

## Calculable data

The model calculates:

- band energies along an ordered path;
- orbital-, site-, shell-, or spin-projected band weights;
- total and projected electronic density of states;
- one-dimensional Fermi points;
- two-dimensional Fermi contours; and
- three-dimensional triangulated Fermi surfaces.

Paths carry symmetry labels and cumulative physical distance in Å$^{-1}$.
Meshes carry normalized integration weights, dimensions, and shifts. Fermi
surface results retain vertices in both reduced and physical reciprocal
coordinates.

This stage does not yet calculate neutron intensity, bulk susceptibility, or a
fit observable. The generalized Lindhard response and its magnetic projections
are Stage 4; interaction dressings are Stage 5.

## Manual construction

`build_electronic_model` accepts arbitrary complex onsite and hopping matrices.
For convenience, it can add a missing Hermitian-conjugate $-\mathbf R$ block.
If both partners are supplied, inconsistency is an error.
Optional `parameter_values` and `parameter_hoppings` define named linear
Hamiltonian terms. `model.with_parameters(...)` returns a new immutable model,
which provides the parameter boundary needed by later fitting stages.

| Argument | Meaning | Acceptable input example |
| --- | --- | --- |
| `direct_lattice` | lattice-vector matrix described above | `np.diag([4.0, 4.0, 12.0])` |
| `basis` | ordered basis states | `[BasisState("M1_d"), BasisState("X1_p")]` |
| `hoppings` | map from integer $\mathbf R$ to $N\times N$ $H(\mathbf R)$ in meV | `{(0, 0, 0): [[0.0]], (1, 0, 0): [[-80.0]]}` |
| `orbital_centers` | fractional center of each basis state | `[[0, 0, 0], [0.5, 0.5, 0]]` |
| `periodic_axes` | periodic lattice axes | `(0, 1)` |
| `interpolation_weights` | optional map from $\mathbf R$ to positive weight | `{(0, 0, 0): 1.0}` |
| `parameter_values` | named linear-term values | `{"t_nn": -80.0}` |
| `parameter_hoppings` | named maps from $\mathbf R$ to selector matrices | `{"t_nn": {(1, 0, 0): [[1.0]]}}` |
| `spin_operators` | optional three spin matrices | `np.asarray([Sx, Sy, Sz])` |
| `energy_zero_meV` | recorded reference energy | `0.0` |
| `add_hermitian_conjugates` | add a missing $-\mathbf R$ partner | `True` |
| `provenance` | source description | `{"source": "manual", "author": "user"}` |

```python
import numpy as np

from nfit import BasisState, build_electronic_model

model = build_electronic_model(
    direct_lattice=np.diag([4.0, 4.0, 12.0]),
    basis=[
        BasisState("d_xy", site="M", orbital="d_xy"),
        BasisState("p_x", site="X", orbital="p_x"),
    ],
    orbital_centers=[[0, 0, 0], [0.5, 0.5, 0]],
    periodic_axes=(0, 1),
    hoppings={
        (0, 0, 0): [[-20, 0], [0, 30]],
        (1, 0, 0): [[-80, 0], [0, -40]],
        (0, 1, 0): [[-80, 0], [0, -40]],
    },
    parameter_values={"hybridization": 15.0},
    parameter_hoppings={
        "hybridization": {
            (0, 0, 0): [[0, 1], [1, 0]],
        },
    },
)
```

## Wannier90 import

`import_wannier90` reads `seedname_hr.dat` or `seedname_tb.dat` without a
Wannier90 runtime dependency. It converts eV to meV, applies the reported
Wigner--Seitz degeneracies, and uses `seedname_wsvec.dat` when present.

An `hr.dat` file does not contain the lattice or orbital centers. Supply them
explicitly or retain the associated `seedname.win` and
`seedname_centres.xyz`. A `tb.dat` file contains the lattice and position
matrix from which the centers are read. Imported files, sizes, SHA-256
digests, unit conversions, replica treatment, and Fourier gauge are retained
as provenance.

| Argument | Meaning | Acceptable input example |
| --- | --- | --- |
| `path` | input Hamiltonian | `"seedname_hr.dat"` or `"seedname_tb.dat"` |
| `direct_lattice` | optional lattice override required when `hr.dat` has no associated `.win` | `np.diag([4.0, 4.0, 4.0])` |
| `orbital_centers` | optional fractional centers required when `hr.dat` has no associated centers file | `[[0, 0, 0], [0.5, 0.5, 0.5]]` |
| `basis` | optional basis metadata replacing generated `w1`, `w2`, ... labels | `[BasisState("M1_d_xy"), BasisState("M1_d_xz")]` |
| `periodic_axes` | optional explicit periodic directions | `(0, 1, 2)` |
| `wsvec_path` | optional replica-correction file | `"seedname_wsvec.dat"` |

## Bands, density of states, and Fermi surfaces

```python
import numpy as np

from nfit import (
    band_path,
    calculate_bands,
    density_of_states,
    fermi_surface,
    k_mesh,
)

path = band_path(
    model,
    [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]],
    labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"],
)
bands = calculate_bands(model, path, projections={"d": [0], "p": [1]})

mesh = k_mesh(model, [80, 80])
dos = density_of_states(
    model,
    mesh,
    np.linspace(-400, 400, 1000),
    broadening_meV=3.0,
    projections={"d": [0], "p": [1]},
)
surface = fermi_surface(model, [200, 200], target_energy_meV=0.0)
```

The calculation arguments beyond the `model` itself are:

| Argument | Meaning | Acceptable input example |
| --- | --- | --- |
| `band_path.nodes` | two or more reduced-coordinate path nodes | `[[0, 0, 0], [0.5, 0, 0]]` |
| `band_path.labels` | optional label for each node | `["G", "X"]` |
| `band_path.points_per_segment` | positive interpolation-interval count | `80` |
| `k_mesh.shape` | positive size for each periodic axis, or three lattice-axis sizes | `[80, 80]` |
| `k_mesh.shift` | optional offset in mesh steps for each periodic axis | `[0.5, 0.5]` for a half-step shift |
| `calculate_bands.sampling` | `WavevectorSampling` path or mesh | `path` or `mesh` from the functions above |
| `calculate_bands.chemical_potential_meV` | energy stored as the plotting reference | `12.5` |
| `calculate_bands.projections` | optional named zero-based basis-index groups | `{"d": [0, 1]}` |
| `calculate_bands.include_eigenvectors` | retain the complete eigenvectors in the result | `True` |
| `density_of_states.mesh` | mesh-valued `WavevectorSampling` | `k_mesh(model, [80, 80])` |
| `density_of_states.energy_meV` | one-dimensional absolute energy grid | `np.linspace(-250, 250, 1000)` |
| `density_of_states.broadening_meV` | positive Gaussian standard deviation | `2.0` |
| `density_of_states.chemical_potential_meV` | plotting reference energy | `12.5` |
| `density_of_states.projections` | optional named basis-index groups | `{"d": [0, 1]}` |
| `density_of_states.max_chunk_bytes` | positive temporary-kernel memory budget | `67108864` for 64 MiB |
| `fermi_surface.mesh_shape` | extraction-grid size for each periodic axis | `[200, 200]` |
| `fermi_surface.target_energy_meV` | absolute constant-energy target | `12.5` |
| `fermi_surface.projections` | optional named basis-index groups evaluated on the surface | `{"d": [0, 1]}` |

The DOS calculation is chunked under an explicit temporary-memory budget.
Production backend selection and symmetry-reduced meshes remain part of the
later optimization stage.

## Fitting and identifiability

Stage 3 electronic models are calculation-only components. They can be saved
in projects and used by the model-owned plots, but are not passed to the
optimizer until an electronic response supplies a dataset observable. Later
fit results will retain the electronic model digest and calculation
provenance.

The current named linear terms are therefore parameter-ready but not yet shown
as fit checkboxes in the GUI. In this documentation, **onsite energy** means a
static diagonal or symmetry-allowed onsite Hamiltonian term. A frequency-
dependent many-body self-energy $\Sigma(\mathbf k,E)$ is a separate future
extension and should not be conflated with an onsite energy.

## Scripting and export

The **Tight-binding electronic structure** model editor supports three
structure inputs:

- import a CIF and retain its path, size, and SHA-256 digest;
- edit the lattice, space group, element labels, and fractional site
  positions manually; or
- import a complete Wannier90 Hamiltonian.

CIF import initializes `periodic_axes` to `[0, 1, 2]`. The structure editor
stores candidate orbital locations but does not yet assign orbital manifolds
or construct $H(\mathbf R)$ from them; that is Stage 3.2. Complete manual
Hamiltonians can still be constructed through the public scripting API and
stored as `model_data`. The remaining workflow is specified in the
[Tight-binding model-builder plan](tight_binding_builder_plan.md).

Each plot action has a **Copy script** button that exports editable GUI-free
Python using the same calculation and rendering functions. **Copy structure
script** exports CIF reload and digest verification, or embeds a manually
entered crystal, then reconstructs the data group, model component, and
periodic axes without Qt.

`ElectronicModel.to_dict()` is a portable, digest-protected representation.
`save_electronic_model` and `load_electronic_model` write and validate that
representation as JSON. Imported-model scripts reload the original source and
verify the stored canonical digest.

## References

- A. A. Mostofi *et al.*, *Comput. Phys. Commun.* **178**, 685 (2008),
  [doi:10.1016/j.cpc.2007.11.016](https://doi.org/10.1016/j.cpc.2007.11.016).
- G. Pizzi *et al.*, *J. Phys.: Condens. Matter* **32**, 165902 (2020),
  [doi:10.1088/1361-648X/ab51ff](https://doi.org/10.1088/1361-648X/ab51ff).
- [Wannier90 file-format documentation](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/).

```{toctree}
:maxdepth: 1
:hidden:

tight_binding_builder_plan
```
