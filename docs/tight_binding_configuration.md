# Configuration and performance

This page defines every registered `tight_binding` component setting. Onsite,
hopping, and spin--orbit coefficients are dynamic fit parameters described on
their construction pages; the fields below are fixed scientific or execution
configuration.

## Source and builder state

| Setting | Default | Meaning |
| --- | --- | --- |
| `source_path` | `""` | Wannier90 `*_hr.dat` or `*_tb.dat` source. Empty means the component uses stored manual or builder data. |
| `model_digest` | `""` | Expected SHA-256 digest of the resolved canonical model. A source reload fails if it differs. |
| `model_data` | `{}` | Portable `ElectronicModel.to_dict()` payload. |
| `model_stale` | `false` | Derived flag indicating that builder edits require one canonical rebuild. It is not a physical option. |
| `use_primitive_cell` | `true` | Exactly fold a compatible GUI-built conventional-cell model before evaluation. |
| `crystal` | cubic `P 1` cell with no sites | Editable lattice, space group, sites, and optional CIF provenance. |
| `orbital_manifolds` | `[]` | Site-attached basis definitions, symmetry choices, and local frames. |
| `onsite_terms` | `[]` | Generated onsite matrices and mirrored parameter state. |
| `hopping_parameterization` | `"slater_koster"` | Candidate basis: compact two-centre integrals or `"general"` symmetry matrices. |
| `hopping_cutoff_angstrom` | `0.0` | Largest bond distance examined by the hopping generator; zero generates none. |
| `spatial_orbits` | `[]` | Generated symmetry-distinct geometric pathways. |
| `hopping_candidates` | `[]` | Complete generated suggestions. Candidates do not enter $H$. |
| `hopping_terms` | `[]` | Selected active hopping coefficients. |
| `spin_treatment` | `"auto"` | `"auto"`, `"implicit"`, `"collinear"`, or `"spinor"`. |
| `soc_terms` | `[]` | Enabled manifold-resolved onsite $\lambda\mathbf L\cdot\mathbf S$ terms. |
| `periodic_axes` | `[]` | Periodic direct-lattice axes. Empty lets a Wannier import infer them from nonzero translations. |

Structured fields contain JSON-compatible dictionaries or lists. Their
detailed record schemas are defined in
[Lattice, sites, and orbitals](tight_binding_lattice_orbitals.md),
[Hoppings](tight_binding_hoppings.md), and
[Spin and spin--orbit coupling](tight_binding_spin.md).

## Units, projections, and execution

| Setting | Default | Meaning and example |
| --- | --- | --- |
| `electronic_energy_unit` | `"eV"` | Unit used by electronic entry fields and plots; choose `"eV"` or `"meV"`. Canonical state remains meV. |
| `chemical_potential_meV` | `0.0` | Canonical plotting reference and source chemical potential. `12.5` means $0.0125$ eV. |
| `projection_groups` | `{}` | Labels mapped to zero-based basis indices, for example `{"d":[0,1,2],"p":[3,4]}`. |
| `electronic_backend` | `"auto"` | `"auto"`, `"numpy"`, `"threaded"`, or explicit `"cupy"`. |
| `electronic_workers` | `0` | Maximum CPU workers. Zero follows nfit's allocation and `NFIT_NUM_THREADS`. |
| `electronic_max_batch_mb` | `256.0` | Temporary Hamiltonian/eigensystem memory target in MiB. |

`chemical_potential_meV` changes the energy origin shown on plots; it does not
solve for electron filling. A linked Lindhard component can instead solve the
chemical potential from `filling_per_cell`.

Projection groups are basis-index sums. A projected calculation must retain
eigenvectors and is therefore more expensive than an eigenvalue-only band or
total-DOS calculation.

## Band path

| Setting | Default | Meaning and example |
| --- | --- | --- |
| `band_path` | $\Gamma$--X--M--$\Gamma$ nodes | Ordered labels and primitive reduced coordinates, for example `[{"label":"G","k":[0,0,0]},{"label":"X","k":[0.5,0,0]}]`. |
| `band_path_convention` | `"hinuma"` | `"hinuma"` for a generated HPKOT path or `"manual"`. |
| `band_path_metadata` | `{}` | Provider, version, convention, and symmetry tolerance of an automatic path. |
| `band_points_per_inv_angstrom` | `80.0` | Positive interpolation-interval density per Å$^{-1}$ of physical path length. |

For a three-dimensional crystal,
`set_tight_binding_standard_path(component)` uses Seek-path's
Hinuma/HPKOT convention and converts the standardized path to nfit's primitive
reciprocal basis. Manual paths remain available for nonstandard or
reduced-dimensional models. Labels alone have no coordinate meaning: a point
called X must carry coordinates appropriate to the selected convention.
Connected segments are sampled at a uniform physical momentum density, so
longer segments receive proportionally more interpolation points.

## Density of states

| Setting | Default | Meaning and example |
| --- | --- | --- |
| `dos_mesh` | `[40,40,40]` | Uniform integration mesh. A two-dimensional model may use `[80,80]`. |
| `dos_symmetry` | `"full"` | `"full"`, certified `"auto"` reduction with fallback, or required `"reduced"` sampling. |
| `dos_energy_min_meV` | `-500.0` | Lower absolute energy sampled, in canonical meV. |
| `dos_energy_max_meV` | `500.0` | Upper absolute energy sampled, in canonical meV. |
| `dos_energy_points` | `600` | Number of energy samples, at least two. |
| `dos_broadening_meV` | `5.0` | Positive Gaussian standard deviation in meV. |

The total DOS is normalized per primitive cell and per energy. Rendering in eV
converts both the energy axis and states/meV to states/eV, preserving the
integrated number of states.

Symmetry reduction is certified only for a three-dimensional uniform mesh and
a model built with nfit's known orbital representations. `auto` records a
full-mesh fallback when it cannot prove equivalence; `reduced` raises instead.
The component setting applies only to total DOS because an arbitrary orbital
projection need not be symmetry invariant.

## Constant-energy and Fermi surfaces

| Setting | Default | Meaning and example |
| --- | --- | --- |
| `fermi_mesh` | `[64,64,64]` | Extraction grid with at least two points per periodic axis. |
| `fermi_energy_meV` | `0.0` | Absolute target energy in canonical meV. |

Set `fermi_energy_meV` equal to the chemical potential for a Fermi surface.
Another value produces a general constant-energy surface. nfit extracts
crossing points in one dimension, contours in two dimensions, and triangulated
surfaces in three dimensions.

## Making calculations efficient

The following changes preserve the model but alter cost:

1. **Use the primitive cell.** Matrix diagonalization scales approximately as
   the cube of basis dimension. An exact conventional-to-primitive fold often
   gives the largest speedup.
2. **Keep spin implicit when valid.** Explicit collinear or spinor treatment
   doubles matrix dimension.
3. **Avoid unused eigenvectors.** Total bands and total DOS use eigenvalues
   only; projections require eigenvectors.
4. **Converge meshes from below.** Increase band-path density, DOS mesh, or
   Fermi mesh until the feature of interest is stable.
5. **Use a bounded memory target.** A larger
   `electronic_max_batch_mb` reduces dispatch overhead but increases temporary
   memory. It does not change the sampled points or numerical formula.
6. **Choose the backend deliberately.** `auto` uses NumPy and bounded CPU
   threading when the workload is large enough. `threaded` requests that path
   explicitly. `cupy` is an optional GPU backend and the low-level electronic
   service falls back to NumPy if it is unavailable.

Threaded evaluation assembles each bounded wave of Hamiltonians serially, then
diagonalizes completed matrices in parallel. This avoids overlap between
complex Hamiltonian assembly and platform LAPACK calls. Results remain
float64/complex128 and retain input order.

Backend choice and batching are recorded in result provenance. GPU arithmetic
can differ at roundoff level; for a response fit, the Lindhard component can
validate a deterministic probe against serial NumPy before using a threaded or
GPU backend.

Three-dimensional Fermi-surface extraction retains a complete regular grid
because contouring requires its topology. It is therefore not symmetry
reduced.

## Lazy rebuilding and fitting

Editing a crystal, manifold, onsite term, hopping term, or SOC term invalidates
the resolved canonical model. The next band, DOS, Fermi-surface, matrix,
response, or script calculation rebuilds it once. Repeated calculations reuse
the immutable result until a scientific input changes.

Every active term identifier is shared by `component.parameters`,
`component.limits`, `component.fit_parameters`, `component.sharing`, and the
resolved `ElectronicModel.parameter_values`. The tight-binding component is
calculation-only; a linked response component supplies the dataset residual.
Fit reports then include fitted electronic coefficients and their
uncertainties.

## Scripts and reproducibility

Every electronic plot has **Copy script** support. **Copy builder script**
reconstructs the crystal or source import, orbital manifolds, generated terms,
values, bounds, fit selections, sharing, and expected model digest without
constructing Qt widgets.

Low-level calculations expose corresponding `backend`, `workers`, and
`max_batch_bytes` arguments. Process defaults can be set with
`set_electronic_backend`, `NFIT_ELECTRONIC_BACKEND`, and
`NFIT_NUM_THREADS`.

## Reference

- Y. Hinuma *et al.*, *Comput. Mater. Sci.* **128**, 140 (2017),
  [doi:10.1016/j.commatsci.2016.10.015](https://doi.org/10.1016/j.commatsci.2016.10.015).
- [Seek-path documentation](https://seekpath.readthedocs.io/).
