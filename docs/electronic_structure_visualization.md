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
through the zone surface. When a path revisits the same reciprocal-space
point, the viewer draws each repeated label only once. Distinct labels assigned
to one point are combined with a vertical bar, such as `X|Y`.

`BrillouinZoneViewOptions` exposes:

| Field | Default | Meaning |
| --- | --- | --- |
| `basis_vector_color_mode` | `"single"` | one color or `"rgb"` |
| `basis_vector_color` | `"#496A9B"` | color used in single-color mode |
| `basis_vector_thickness` | `0.006` | reciprocal-vector shaft radius |
| `basis_vector_inside_style` | `"solid"` | `"solid"`, `"dashed"`, or `"hidden"` inside the cell |
| `path_color` | `"#7A1F1F"` | high-symmetry path color |
| `path_thickness` | `4.0` | path width |
| `path_point_size` | `16.0` | high-symmetry point diameter in screen pixels |
| `label_font_size` | `18` | high-symmetry and reciprocal-vector label size |
| `label_bold` | `false` | use bold label text |
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

Its **Calculation** panel edits the path convention, manual labelled nodes,
and sampling density. Applying a standard convention regenerates its
coordinates; applying a manual path uses the entered nodes directly.

`calculate_bands` returns eigenvalues, optional eigenvectors, and any requested
basis-index projections. `render_band_structure` handles the energy-unit
conversion and labels the relative energy as $E-\mu$. Disconnected endpoint
labels sharing one horizontal position are
combined as, for example, `U|K`. Increase `band_points_per_inv_angstrom` until
curvature and crossings are visually stable.

## Density of states

For band energies $\varepsilon_{n\mathbf k}$ in meV and normalized mesh
weights $\sum_{\mathbf k}w_{\mathbf k}=1$, the total density of states is
$D(\varepsilon)=d_s\sum_{\mathbf k,n}w_{\mathbf k}
\delta(\varepsilon-\varepsilon_{n\mathbf k})$ in states/(meV model cell).
$\delta$ is the Dirac distribution with unit integral and inverse-energy units;
$d_s=2$ for implicit spin and 1 when spin is explicit. Band index $n$ spans the
represented basis. A Gaussian width $\sigma_E$ replaces $\delta(x)$ by
$e^{-x^2/(2\sigma_E^2)}/(\sqrt{2\pi}\sigma_E)$, with $x,\sigma_E$ in meV.
Filling is $n_e=\int D(\varepsilon)f(\varepsilon)d\varepsilon$ electrons per
model cell, where $f$ is the Fermi occupation at temperature $T$ and chemical
potential $\mu$, defined in [Lindhard](lindhard.md#complex-susceptibility).

The DOS viewer offers two integrations:

- **Gaussian** replaces each sampled eigenvalue with a normalized Gaussian of
  standard deviation `dos_broadening_meV`. It supports every model dimension
  and the certified total-DOS symmetry reduction. Because symmetry-equivalent
  wavevectors carry identical eigenvalues, the Gaussian sum is evaluated
  directly on the irreducible mesh with orbit multiplicities as weights. That
  is an exact rewrite of the full-mesh sum — it reproduces it to machine
  precision — and it saves the broadening kernel as well as the
  diagonalization.
- **Linear tetrahedron** uses ASE to interpolate each band inside the
  tetrahedra of a complete uniform three-dimensional mesh. It introduces no
  artificial broadening. For certified models, nfit can diagonalize the
  symmetry-unique points and expand the eigenvalues back onto the complete
  ordered grid before integration. Total DOS and complete-basis projections
  use this exact acceleration; arbitrary orbital projections use the full
  eigensystem unless their symmetry transformation is known.

An exactly flat band is a delta function in the DOS and makes the usual
tetrahedron denominators singular. nfit represents its full normalized weight
on the nearest energy-grid sample, or splits it between the two neighboring
samples when necessary. Refining the energy grid makes the displayed peak
narrower and taller without changing its integrated number of states.

`density_of_states` returns states per meV per primitive cell, **counting both
spin states**. An implicit-spin model is multiplied by its two-fold degeneracy,
so the total integrates to $2N_{\rm basis}$ states per cell and agrees with the
electron count from `electron_filling`; a collinear or spinor model already
carries spin in its basis and uses a degeneracy of one. The applied factor is
recorded as `provenance["spin_degeneracy"]`. $E_{\rm F}$ denotes the
zero-temperature chemical potential. The Sommerfeld relation uses this
both-spin DOS after converting its energy and formula-unit normalization;
the scalar Stoner criterion uses $D_\uparrow=D/2$ for a spin-degenerate model,
not the total $D$. See [Low-temperature heat capacity](low_temperature_heat_capacity.md)
and [Stoner RPA](stoner_rpa.md#complex-susceptibility).

Plotting in eV converts the ordinate to states per eV per cell. Mesh density
controls integration accuracy. For Gaussian DOS, broadening also controls
displayed energy resolution and should be converged separately.

The viewer side panel owns the integration method, mesh, certified symmetry
policy, energy window, energy-point count, and Gaussian width. **Automatic
range** derives the minimum and maximum from the eigenvalues on the selected
DOS mesh. Gaussian integration adds four standard deviations of padding;
tetrahedron integration adds a small band-span margin.

## Band and DOS presentation

The band and DOS viewers share compact, live presentation controls adapted
from the slice viewer's one-dimensional plotter. Both viewers control curve
color and width, marker shape (none by default), marker size and fill (none by
default), a common axes/tick/legend font size, border width, and legend
visibility. **Show orbital projections** hides projection weights and projected
DOS curves without hiding the total bands or total DOS. The Fermi-level
reference has independent color, width, and line style controls. The band
viewer also gives the vertical high-symmetry guides their own color, width, and
style.

Ticks point inward and appear on all four sides. Legends use an opaque black
outline with square corners; border width also sets the legend-outline and
tick thickness. The default border width is 1.5 pt; Fermi-level and
high-symmetry guides default to 1 pt. `ElectronicPlotStyle` holds these
settings for scripts, and
`apply_electronic_plot_style` applies them to an existing electronic figure.
`render_band_structure` and `render_density_of_states` accept the same object
through their `style` argument.

## Constant-energy and Fermi surfaces

The surface viewer extracts $\varepsilon_n(\mathbf k)=E_{\rm target}$ for
every crossing band. The result is:

- points in one periodic dimension;
- line segments in two dimensions; or
- triangulated sheets in three dimensions.

Set the target equal to the chemical potential for a Fermi surface.
`fermi_surface` retains vertices in reduced and physical reciprocal
coordinates, connectivity, band index, and optional projected weights.

The isosurface grid repeats the zone face so that the surface closes, but
those points are periodic images of the interior. nfit diagonalizes only the
distinct wavevectors, further reduced to the irreducible wedge when the model
carries certified reciprocal symmetry, then gathers the eigenvalues back onto
the full grid. Both steps are exact — the extracted vertices are identical —
and the applied reduction is recorded in
`provenance["symmetry_reduction"]`. Bands that never reach the target energy
are skipped entirely, and eigenvectors for projected weights are computed only
at the surface vertices.
By default the viewer accepts one target grid spacing in Å$^{-1}$ and resolves
the nearest grid size separately along each reciprocal basis vector. This
keeps the physical sampling density comparable in anisotropic cells. Select
**Explicit grid size** to enter the full three-dimensional grid directly.

Three-dimensional results use a PyVista renderer for responsive rotation.
One- and two-dimensional results use Matplotlib.
`render_fermi_surface` remains available for a static scripted figure.
The 3D side panel controls band-sheet opacity, reciprocal-grid line width,
axes and legend text size, legend visibility, and shading. Smooth shading is
the default and interpolates vertex normals across the extracted triangles;
flat shading deliberately preserves visible facets. These settings affect
only rendering, not the surface vertices or connectivity. Fermi-surface
viewers omit a title so the viewport remains focused on the geometry.

Scripts can configure the same presentation with
`FermiSurfaceViewOptions`:

```python
from nfit.qt_fermi_surface_viewer import (
    FermiSurfaceViewOptions,
    show_fermi_surface_result,
)

options = FermiSurfaceViewOptions(
    band_opacity=0.5,
    text_size=18,
    grid_line_width=1.5,
    show_legend=False,
    shading="smooth",
)
window = show_fermi_surface_result(result, view_options=options)
```

The regular extraction grid includes its periodic boundary and is not
symmetry reduced. Increase all periodic mesh dimensions until topology and
small pockets are stable.

The viewer side panel owns the extraction policy and constant-energy target.
These remain visualization settings rather than certified convergence claims.

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

where $A$ is the inspected matrix and
$P_\alpha=\sum_{a\in G_\alpha}|a\rangle\langle a|$ selects basis-index group
$G_\alpha$ (similarly for $\beta$). These $\alpha,\beta$ are subspace labels,
not Cartesian components, and $P$ is dimensionless. The block has the units of
$A$; its Frobenius norm is $\sqrt{\sum_{ab}|(P_\alpha A P_\beta)_{ab}|^2}$.
Blocks are
grouped by site, manifold, and spin, including block shape, Frobenius norm, and
largest element. This is the appropriate tool for distinguishing geometric
pathways from their multiorbital matrix content.

`electronic_matrix_catalog` is renderer independent,
`show_electronic_matrix_catalog` opens the viewer, and
`electronic_matrix_script` exports the inspection.

## Viewer window convention

The band, DOS, Fermi-surface, and Brillouin-zone viewers use a plot area on the
left and a fixed, scrollable **Settings** panel on the right. Band, DOS, and Fermi-surface
calculation settings are stored only after **Apply and recalculate** succeeds.
Successful recalculation replaces the plot inside the existing window, keeping
its position, size, and presentation controls. The three-dimensional
Fermi-surface viewer also preserves its camera. Band and DOS presentation
settings update immediately without recalculation. Band, DOS, and
Fermi-surface viewers provide **Copy figure** and **Save figure** actions in
their Output panels.
The Brillouin-zone panel controls presentation without rebuilding its cached
geometry. Standard close shortcuts use Command-W on macOS and Control-W elsewhere.
