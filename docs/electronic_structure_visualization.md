# Electronic-structure visualization

The visualization tools inspect different parts of one resolved electronic
model. Their calculations are available through public APIs, and GUI actions
can export editable scripts.

## Model geometry

**View model in 3D** shows the direct-space construction shared by tight
binding and Heisenberg RPA. A tight-binding scene can display:

- the full unit cell or only electronically active sites;
- inactive atoms as translucent ghosts;
- every orbital token and its local frame;
- one representative hopping pathway; or
- every symmetry-equivalent member of a spatial orbit.

Atoms use element-dependent colors and display radii. Orbital tokens are
separated colored markers so several orbitals can be seen on one site; they are
not real-space wavefunction isosurfaces.

Selecting an atom, orbital token, or pathway updates the **Selected object**
panel with its expanded site, element, orbital, manifold, or orbit label. When
a hopping matrix term is selected, the scene shows its geometric pathway; use
the matrix inspector for the actual orbital-to-orbital amplitudes.

`model_geometry_scene` returns renderer-independent immutable geometry.
`model_geometry_script` exports a standalone viewer.

## First Brillouin zone

**View Brillouin zone in 3D** constructs the Wigner--Seitz cell of the
primitive reciprocal lattice. For a centered conventional crystal, the
space-group centering translations define the primitive direct vectors.

Configured `band_path` coordinates use that primitive reciprocal basis. A
node `[0.5,0,0]` means $\mathbf b_1/2$; the label X alone does not define its
location. Standard paths can follow the Hinuma/HPKOT convention from
Seek-path or the Setyawan--Curtarolo convention from ASE.

The scene contains the zone polyhedron, labelled path, and reciprocal vectors
$\mathbf b_1,\mathbf b_2,\mathbf b_3$ in Å$^{-1}$. Each reciprocal vector
extends to the neighboring reciprocal-lattice point and therefore passes
through the zone surface.

`BrillouinZoneViewOptions` exposes:

| Field | Default | Meaning |
| --- | --- | --- |
| `basis_vector_color_mode` | `"single"` | one color or `"rgb"` |
| `basis_vector_color` | `"#496A9B"` | color used in single-color mode |
| `basis_vector_thickness` | `0.006` | reciprocal-vector shaft radius |
| `basis_vector_inside_style` | `"solid"` | `"solid"`, `"dashed"`, or `"hidden"` inside the cell |
| `path_color` | `"#7A1F1F"` | high-symmetry path color |
| `path_thickness` | `4.0` | path width |
| `label_font_size` | `14` | high-symmetry label size |
| `cell_surface_color` | `"#B8C7D9"` | Wigner--Seitz face color |
| `cell_surface_opacity` | `0.10` | face opacity from 0 to 1 |
| `cell_outline_color` | `"#202020"` | zone-edge color |
| `cell_outline_thickness` | `4.0` | zone-edge width |
| `show_basis_vectors` | `true` | show reciprocal vectors |
| `show_path` | `true` | show connected path segments |
| `show_path_labels` | `true` | show node labels |
| `show_compass` | `false` | show the Cartesian XYZ compass |
| `projection` | `"orthographic"` | `"orthographic"` or `"perspective"` |

The viewer also copies or saves the current viewport. Presentation changes
redraw the cached geometry rather than rebuilding the Wigner--Seitz cell.

`brillouin_zone_scene` builds a component scene.
`build_brillouin_zone_scene` accepts explicit direct and primitive lattices,
and `show_brillouin_zone_scene` renders it with scriptable options.

## Band structure

The band viewer diagonalizes $H(\mathbf k)$ along `band_path`. The horizontal
axis is cumulative physical distance in Å$^{-1}$, with disconnected sections
kept separate. Energies are shown relative to the configured chemical
potential.

`calculate_bands` returns eigenvalues, optional eigenvectors, and any requested
basis-index projections. `render_band_structure` handles the energy-unit
conversion. Disconnected endpoint labels sharing one horizontal position are
combined as, for example, `U|K`. Increase `band_points_per_inv_angstrom` until
curvature and crossings are visually stable.

## Density of states

The DOS viewer offers two integrations:

- **Gaussian** replaces each sampled eigenvalue with a normalized Gaussian of
  standard deviation `dos_broadening_meV`. It supports every model dimension
  and the certified total-DOS symmetry reduction.
- **Linear tetrahedron** uses ASE to interpolate each band inside the
  tetrahedra of a complete uniform three-dimensional mesh. It introduces no
  artificial broadening and supports the same basis-index projections, but it
  cannot use a symmetry-reduced mesh.

`density_of_states` returns states per meV per primitive cell. Plotting in eV
converts the ordinate to states per eV per cell. Mesh density controls
integration accuracy. For Gaussian DOS, broadening also controls displayed
energy resolution and should be converged separately.

## Constant-energy and Fermi surfaces

The surface viewer extracts $\varepsilon_n(\mathbf k)=E_{\rm target}$ for
every crossing band. The result is:

- points in one periodic dimension;
- line segments in two dimensions; or
- triangulated sheets in three dimensions.

Set the target equal to the chemical potential for a Fermi surface.
`fermi_surface` retains vertices in reduced and physical reciprocal
coordinates, connectivity, band index, and optional projected weights.

Three-dimensional results use a PyVista renderer for responsive rotation.
One- and two-dimensional results use Matplotlib.
`render_fermi_surface` remains available for a static scripted figure.

The regular extraction grid includes its periodic boundary and is not
symmetry reduced. Increase all periodic mesh dimensions until topology and
small pockets are stable.

## Calculation API inputs

The viewer calculations are GUI independent. Their main inputs are:

| Function and argument | Meaning | Example |
| --- | --- | --- |
| `band_path.nodes` | two or more reduced-coordinate nodes | `[[0,0,0],[0.5,0,0]]` |
| `band_path.labels` | optional node labels | `["G","X"]` |
| `band_path.break_before` | suppress connection from the preceding node | `[False,False,True,False]` |
| `band_path.points_per_inv_angstrom` | positive interpolation-interval density per Å$^{-1}$ | `80.0` |
| `band_path.coordinate_reciprocal_lattice` | optional reciprocal basis in which node coordinates are expressed | primitive reciprocal $3\times3$ matrix |
| `k_mesh.shape` | positive size per periodic axis, or three lattice-axis sizes | `[80,80]` |
| `k_mesh.shift` | offset in mesh steps | `[0.5,0.5]` |
| `k_mesh.symmetry` | `"full"`, certified `"auto"`, or required `"reduced"` | `"auto"` |
| `calculate_bands.sampling` | path or mesh `WavevectorSampling` | result of `band_path(...)` |
| `calculate_bands.chemical_potential_meV` | energy stored as the plotting reference | `12.5` |
| `calculate_bands.projections` | named zero-based basis-index groups | `{"d":[0,1]}` |
| `calculate_bands.include_eigenvectors` | retain complete eigenvectors in the result | `True` |
| `density_of_states.mesh` | mesh-valued sampling with weights | result of `k_mesh(...)` |
| `density_of_states.energy_meV` | one-dimensional absolute energy grid | `np.linspace(-250,250,1000)` |
| `density_of_states.broadening_meV` | positive Gaussian standard deviation | `2.0` |
| `density_of_states.method` | `"gaussian"` or three-dimensional `"tetrahedron"` integration | `"tetrahedron"` |
| `density_of_states.projections` | optional basis-index groups | `{"d":[0,1]}` |
| `density_of_states.max_chunk_bytes` | temporary Gaussian-kernel memory target | `67108864` |
| `fermi_surface.mesh_shape` | at least two points per periodic axis | `[64,64,64]` |
| `fermi_surface.target_energy_meV` | absolute constant-energy target | `12.5` |
| `fermi_surface.projections` | optional weights evaluated on each sheet | `{"d":[0,1]}` |
| `backend`, `workers`, `max_batch_bytes` | shared eigensystem execution controls | `"threaded"`, `8`, `268435456` |

`electronic_energy_to_meV` and `electronic_energy_from_meV` are the explicit
unit boundary for low-level scripts. Result objects retain model digest,
sampling, execution backend, precision, and projection definitions.

```python
import numpy as np

from nfit import band_path, calculate_bands, density_of_states, k_mesh

path = band_path(
    model,
    [[0, 0, 0], [0.5, 0, 0]],
    labels=[r"$\Gamma$", "X"],
    points_per_inv_angstrom=80.0,
)
bands = calculate_bands(model, path, projections={"d": [0, 1]})

mesh = k_mesh(model, [64, 64, 64], symmetry="full")
dos = density_of_states(
    model,
    mesh,
    np.linspace(-250.0, 250.0, 1000),
    broadening_meV=2.0,
    method="gaussian",
)
```

## Matrix and subspace inspector

**Inspect matrices** provides exact access to:

- the resolved complex $H(\mathbf k)$;
- each named parameter selector and its coefficient-weighted contribution;
- $S_x,S_y,S_z$ for explicit-spin models; and
- representative onsite and hopping matrix bases.

The heatmap can show real part, imaginary part, magnitude, or phase. A table
shows exact complex entries with ordered row and column labels. For a hopping
$T_{ij}(\mathbf R)$, rows are destination states and columns are source
states.

The subspace panel reports nonzero blocks

$$
P_\alpha A P_\beta
$$

grouped by site, manifold, and spin, including block shape, Frobenius norm, and
largest element. This is the appropriate tool for distinguishing geometric
pathways from their multiorbital matrix content.

`electronic_matrix_catalog` is renderer independent,
`show_electronic_matrix_catalog` opens the viewer, and
`electronic_matrix_script` exports the inspection.

## Viewer window convention

The band, DOS, Fermi-surface, and Brillouin-zone viewers use a plot area on the
left and a fixed **Settings** panel on the right. Not every panel has
interactive scientific controls yet; the shared layout reserves one stable
place for them. Standard close shortcuts use Command-W on macOS and Control-W
elsewhere.
