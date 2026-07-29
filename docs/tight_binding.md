# Tight-binding electronic structure

`tight_binding` represents a material-independent orthonormal electronic
Hamiltonian. It accepts manual real-space models or Wannier90 Hamiltonians and
calculates bands, orbital projections, density of states, and Fermi surfaces.
It does not assume a particular lattice, material, orbital count, or spin
layout.

## Response

The canonical Hamiltonian uses the Wannier gauge

$$
H_{ab}(\mathbf k)=
\sum_{\mathbf R}w_{\mathbf R}H_{ab}(\mathbf R)
\exp(2\pi i\,\mathbf k\cdot\mathbf R).
$$

$a$ and $b$ are arbitrary orthonormal basis states, $\mathbf R$ is an integer
cell translation, and $w_{\mathbf R}$ is an interpolation weight. Energies and
matrix elements are stored in meV. Reduced wavevectors are dimensionless.
Orbital centers are stored separately and do not enter this Fourier phase.

The real-space model must satisfy

$$
H(-\mathbf R)=H(\mathbf R)^\dagger.
$$

nfit validates this relation and the Hermiticity of interpolated
$H(\mathbf k)$. It does not silently repair inconsistent imported data.

## Parameters

An electronic model records:

| Quantity | Meaning | Unit or convention |
| --- | --- | --- |
| direct lattice $A$ | lattice vectors stored as columns | Å |
| basis states | ordered site, species, orbital, shell, and spin metadata | arbitrary finite basis |
| orbital centers | fractional position of each basis state | direct-lattice coordinates |
| $H(\mathbf R)$ | onsite and hopping matrices | meV |
| $w_{\mathbf R}$ | interpolation weights | dimensionless |
| named parameter terms | linear onsite or hopping matrices and their current values | meV-compatible |
| periodic axes | one, two, or three periodic lattice directions | axis indices |
| spin operators | optional $S_x,S_y,S_z$ matrices | dimensionless |

The registered model component also stores the chemical potential, manual
band path, DOS and Fermi-surface meshes, DOS energy grid and broadening, and
named orbital-projection groups. These are fixed calculation settings in
Stage 3 rather than fitted parameters.

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
        (0, 0, 0): [[-20, 15], [15, 30]],
        (1, 0, 0): [[-80, 0], [0, -40]],
        (0, 1, 0): [[-80, 0], [0, -40]],
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

The DOS calculation is chunked under an explicit temporary-memory budget.
Production backend selection and symmetry-reduced meshes remain part of the
later optimization stage.

## Fitting and identifiability

Stage 3 electronic models are calculation-only components. They can be saved
in projects and used by the model-owned plots, but are not passed to the
optimizer until an electronic response supplies a dataset observable. Later
fit results will retain the electronic model digest and calculation
provenance.

## Scripting and export

The **Tight-binding electronic structure** model editor imports Wannier90
sources and provides **Band structure**, **Density of states**, and
**Fermi surface** actions. Each action has a **Copy script** button that
exports editable GUI-free Python using the same calculation and rendering
functions.

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
