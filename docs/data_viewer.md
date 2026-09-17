# Data viewer and saved plots

Composites can include [discrete metadata dimensions](data_import.md#metadata-dimensions),
such as temperature in kelvin. Select these as X/Y, waterfall, or tiled-slice
axes. Hidden-axis controls select one coordinate or integrate a range. In tiled
slices, **Auto** creates one panel per metadata coordinate, labeled with its
actual value; manual step sizes can combine neighboring coordinates. Uneven coordinates retain
their exact values; no data are interpolated between conditions. Stored plot
recipes preserve their metadata-dimension settings along with the spatial grid.

Zero-valued reference slices remain visible after self-subtraction. If a full
metadata tile range gains coordinates during a refresh, its limits expand to
include them. A deliberately restricted range stays restricted.
Numeric range controls accept values beyond their limits while typing, then
clamp to the nearest allowed bound on Enter or focus loss. For example, entering
50 when the maximum is 49.8 selects 49.8. Hidden-axis selections then snap to a
measured coordinate as usual.

The data viewer displays point lists and gridded datasets without changing
their stored values. It supports channel selection, masks, cuts, maps,
waterfalls, model comparisons, and volumetric rendering.
Histogram slices process coverage only inside the selected region. Appearance
changes in the ordinary slice view reuse the displayed numerical slice;
changing axes, selections, masks, or coverage settings recomputes it. No
performance settings are needed in the viewer.

Every **View in data viewer** action opens a new window. When opened from a
dataset collection, the initial selection is that collection's
composite or its first effective descendant dataset. Duplicate composite names
include their collection path to distinguish the series. Multiple viewers may
display the same workspace or different workspaces at once; dataset selection,
axes, ranges, visualization mode, styling, and other controls are independent
in each window. Project changes refresh every open viewer without merging their
view settings. **Open new viewer**, beside the visualization selector, creates
an independent viewer initialized from the current dataset and view settings.
**Store plot**, beside it, saves the current view to the workspace's **Plots**
branch and reports the stored recipe name in the viewer.
Closing one viewer does not affect the others.

The controls are ordered **Dataset**, **Binning**, then **Channel**. Dataset
selects the source dataset or composite; Binning selects one of its enabled
named grids without duplicating that source in the Dataset menu. Only the
configuration marked **Use for fitting** contributes during optimization.
Visualization configurations can use different axes, limits, or resolutions,
and receive model and residual channels from the final fitted parameters.

## Composite neutron representations

In a collection's **Composite dataset → Physics** tab, enable paired channels
and choose **Plot and fit** to select scattering cross section or dynamical
susceptibility χ″ (the imaginary part of the susceptibility). The viewer also
retains the imported signal and both named representations. Settings belong to
the selected composite binning and are saved with the project. Conversion follows
rebinning and background subtraction; parent composites use underlying child
signals so child display settings cannot apply the conversion twice. Materialized
composites retain the underlying signal and the conversion recipe.

A temperature metadata dimension in kelvin takes precedence and supplies the
conversion temperature separately at each coordinate. Preserve this dimension
for temperature scans: averaging different temperatures first cannot recover
their individual detailed-balance corrections. Without this dimension, nfit uses
automatic source temperatures only when every source agrees within 0.1% (absolute
tolerance 10⁻⁶ K). Otherwise set **Nominal temperature (K)** deliberately for a
collection measured at one nominal temperature, or add a temperature dimension
for a series. Missing or ambiguous temperatures stop conversion with an explanation.
A nominal temperature is an explicit approximation for small temperature drifts.

Review the imported quantity, units, and corrections before enabling conversion.
Common source conventions and fixed incident/final energies are proposed when
unambiguous. If the imported cross section includes the neutron wavevector ratio
k_f/k_i, supply the fixed incident energy Ei or final energy Ef in meV appropriate
to the whole composite. Different instrument-energy settings should remain in
separate composites. The shared normalization, magnetic form factor, polarization,
and susceptibility units follow [physics conventions](physics_conventions.md).

The scripting API uses the same saved recipe:

```python
from nfit import composite_dataset_data
from nfit.project_data import data_group_composite_config

config = data_group_composite_config(group)
config["spectral_channels"] = {
    "enabled": True,
    "fit_representation": "chi_double_prime",
    "temperature_K": 5.0,  # K; omit when using a temperature metadata dimension
    "kf_ki_state": "removed",
}
data = composite_dataset_data(group)
```

**Export composite workflow** includes these settings in the editable
`REBIN_CONFIG`. Dataset Physics pages group crystal information side by side and
separate representation selection from correction conventions. The Metadata tab
uses its full height for the scrollable metadata tree.

## Slices, lines, and maps

Choose the displayed axes and use the remaining-axis controls to select one bin
or integrate a range. Channel labels and units come from the dataset's declared
physical quantities. **Apply masks** hides the combined file and nfit mask in
the figure; fitting always excludes masked data. When an integrated auxiliary
channel has no finite contributors, the corresponding displayed pixel remains
blank rather than producing a numerical value.

Each displayed axis has a **Step** control beside its minimum and maximum.
The default is the smallest native bin spacing. Increasing it combines adjacent
display bins in integer multiples of that spacing using inverse-variance
weighting and propagated uncertainties; event and normalization counts are
summed. This is a reversible presentation setting: it does not alter the source
data, project rebin configuration, or cached bins. Stored plots and exported
scripts retain the selected `x_step` and `y_step` values. In waterfall mode the
x step coarsens each trace, while the existing **Bin width** remains responsible
for grouping the vertical waterfall axis.

In **Figure**, enable **Show other-axis binning above plot** to add the current
non-displayed-axis selections to the title. Projected labels and physical units
are retained, for example `[K,-K,0]=[0.9,1.1] r.l.u., ΔE=[1,2] meV`. The option
is preserved independently for each dataset view and included in saved or
copied figure scripts.

### Gridlines

The **Gridlines** panel offers two mutually exclusive overlays. **Major-tick
gridlines** draws conventional horizontal and vertical lines at the visible
major ticks. For a two-dimensional HKL map, **Brillouin-zone boundaries**
instead overlays the intersections of repeated first Brillouin zones with the
displayed reciprocal-space plane. The latter follows oblique projected axes
and hidden-axis coordinates, and tiled views recompute it for each tile. Both
are presentation layers only; they do not alter data, masks, integration, or
fitting.

The reciprocal metric comes from the dataset's UB matrix or lattice parameters.
The space-group symbol supplies the conventional-cell centering (`P`, `I`, `F`,
`A`, `B`, `C`, or `R`). A complete Hermann--Mauguin symbol is accepted, but
point symmetry does not otherwise change the Wigner--Seitz boundary. Set this
information in **Crystal orientation**. If either the metric or centering is
missing when the Brillouin-zone overlay is enabled, the viewer prompts for it
and stores the result on the data group. The two gridline modes share their
color, line thickness, and opacity controls. Defaults are 1.5 points and full
opacity.

Copied scripts and saved plots preserve the selected overlay through
`show_brillouin_zone_boundaries`, `show_major_gridlines`,
`brillouin_zone_spacegroup`,
`brillouin_zone_lattice_parameters`, `brillouin_zone_color`,
`brillouin_zone_linewidth`, and `brillouin_zone_alpha` arguments to
`plot_mdhisto_slice` or `plot_mdhisto_tiled_slices`.

### Momentum-path plots

For a histogram with three independent HKL axes and one energy-transfer axis,
choose **K-path viewer** in the data viewer toolbar. This opens a dedicated
non-fitting visualization. The source dataset and its rebin configuration are
not changed.

Path nodes use absolute conventional-cell `(H K L)` coordinates. They are not
folded into the first Brillouin zone, so a custom path may deliberately pass
through higher zones such as `(1 2 2)`. **Use standard path** constructs a
Setyawan--Curtarolo path for the Bravais lattice recorded under **Crystal
orientation**. The table can then be edited, extended, or replaced. **Break**
starts a disconnected section before a node.

**Path step** controls the longitudinal plotting resolution in inverse
angstroms (Å⁻¹). **Tube radius** selects source momentum voxels by their
shortest physical distance from each connected path segment. Within each
path-energy pixel, nfit reports the inverse-variance weighted intensity

$$
\bar I=\frac{\sum_i I_i/\sigma_i^2}{\sum_i1/\sigma_i^2},\qquad
\sigma_{\bar I}=\frac{1}{\sqrt{\sum_i1/\sigma_i^2}}.
$$

**Minimum voxels** can mask poorly sampled pixels. The viewer can label nodes
with high-symmetry names, absolute HKL coordinates, or both, and can show or
hide independently styled dashed vertical guides. **View path in 3D** reuses
the Brillouin-zone viewer and displays the same absolute path without folding
higher-zone nodes. **Copy script** exports the complete calculation through
the public `prepare_mdhisto_kpath` and `plot_mdhisto_kpath` APIs.

**Minimum coverage** applies independently to viewer reductions. A displayed
pixel is masked when the measured fraction of its requested hidden-axis
integration volume is below the cutoff. The same cutoff is recomputed over
each box selected by the histogram tool before its horizontal and vertical
profiles are drawn. Coverage is available as its own channel and in the cursor
readout; it does not alter pixel opacity or the intensity colormap.

For a histogram box cut, data are combined with inverse-variance weights:

$$
\bar y=\frac{\sum_i y_i/\sigma_i^2}{\sum_i1/\sigma_i^2},
\qquad
\sigma_{\bar y}=\frac{1}{\sqrt{\sum_i1/\sigma_i^2}}.
$$

$y_i$ is a contributing bin value, $\sigma_i$ is its one-sigma uncertainty,
and $\bar y$ is the reduced value with uncertainty $\sigma_{\bar y}$.
Masked bins and invalid uncertainties are excluded.

The selected rectangle also shows its total integrated sum above the plot,
left-aligned beneath the optional other-axis binning title:

$$
Y_{\mathrm{box}}=\sum_i y_i,
\qquad
\sigma_{Y_{\mathrm{box}}}=\sqrt{\sum_i\sigma_i^2}.
$$

The sum uses the finite, displayed bins inside the rectangle. Its one-sigma
uncertainty assumes that their errors are independent. Copied figure scripts
produce the same annotation when histogram axes and rectangle extents are
included.

Plot smoothing is specified in displayed-bin widths. It affects only the
rendered figure and exported figure recipe, not fitting, rebinning, or numerical
data exports. With **Fill adjacent NaN bins** checked, Gaussian smoothing uses
nearby finite display pixels to fill adjacent NaN pixels, including pixels
hidden by a mask. This option is off by default, so smoothing preserves empty
pixels as gaps until it is enabled. Slice, line, waterfall, and tiled-slice
views retain the choice independently for each dataset and in saved plot
recipes. The volume viewer offers the same choice in its own smoothing panel.
Mask channels still show the unchanged masks, and the stored histogram, masks,
and uncertainties are never changed.

## Model and residual channels

**Show model** evaluates the current compatible model or reuses stored
fit-result channels. **Unmask model** extends model evaluation over finite
coordinates excluded from the fit while leaving the data masked. This is useful
for inspecting interpolation or extrapolation, but can be expensive for large
volumes.

For multidimensional data, the live model channel has the same full shape as
the prepared dataset. Slice selection, integration ranges, and reductions are
then applied identically to data and model in the viewer, so changing a viewed
or integrated axis does not require a new model evaluation. A model failure is
isolated to its dataset; compatible datasets retain their overlays, and the
failure text is recorded in the **Current state** metadata.

Two-dimensional comparisons show linked data, fit, and optional residual
panels. Box cuts through a comparison apply the same weighted reduction to the
data and model.

After fitting, nfit also attempts display-only model evaluation for disabled
and zero-weight datasets. Failure of this optional evaluation is recorded but
does not invalidate the fit or discard successful overlays from other
datasets.

Image, tiled-slice, waterfall, and volume colormap menus keep the original
Matplotlib choices first, followed by the complete standard Matplotlib
sequential and diverging families, then Colorcet's short named perceptually
uniform continuous maps, cmcrameri's sequential Scientific Colour Maps,
cmocean sequential maps, and selected Palettable collections.
The standard families include the ColorBrewer maps
used by DAVE/MSlice under `CB-*` names; select **Reverse** to reproduce DAVE's
default direction. `cubehelix` and `CMRmap` provide grayscale-friendly
sequential choices, while `gnuplot2` provides a higher-contrast specialized
ramp and `twilight` and `twilight_shifted` cover periodic phase or angle data.
Each menu row includes a preview swatch of the actual map.
The **alpha** color-shift control redistributes colors within the existing
color range without changing that range or moving colorbar ticks. For a
normalized color coordinate $x$ between zero and one, nfit samples the chosen
colormap at

$$
f_\alpha(x)=\frac{\exp(\alpha x)-1}{\exp(\alpha)-1},
$$

with $f_0(x)=x$. Positive alpha values concentrate more of the color change at
the high end of the range; negative values do the reverse. The scripting
equivalent is `color_alpha=` in `plot_mdhisto_slice` and
`plot_mdhisto_tiled_slices`.
Choose installation-local defaults for newly opened continuous image plots and
waterfall trace sequences under **File → Preferences… → Colormaps**. These
defaults apply across projects on the current computer and are not written to
`.nfit` files. A saved plot's explicit colormap still takes precedence.
The waterfall color menu adds Matplotlib's complete qualitative family and
Colorcet's named Glasbey categorical palettes after the continuous maps for
clearly distinguishing many traces. Separator
rows mark the original Matplotlib, sequential, diverging, specialized,
Colorcet continuous, cmcrameri sequential, cmocean sequential, MyCarta,
CartoColors, nfit, Matplotlib qualitative, and Colorcet categorical sections;
Colorcet names use the `cet_` prefix.
The Matplotlib choices include `Spectral` (the ColorBrewer/CB-Spectral map),
`turbo`, and `bwr`; the latter replaces nfit's former identical
`bluewhitered` menu entry. Existing stored plots and scripts using the old name
remain compatible. The `young_rdbu`, `young_ylbkcy`, and `young_quadratic` maps
reproduce the three control-point sets in `YoungColorMap.m` at its centered
50/50 inflection.

The cmcrameri section contains all 21 sequential maps in cmcrameri 1.10:
`batlow`, `batlowW`, `batlowK`, `devon`, `lajolla`, `bamako`, `davos`,
`bilbao`, `nuuk`, `oslo`, `grayC`, `hawaii`, `lapaz`, `tokyo`, `buda`,
`acton`, `turku`, `imola`, `glasgow`, `lipari`, and `navia`.
Their menu and script names use the `cmc.` prefix, for example `cmc.batlow`.
They use the original 256-color tables and support reversal with `_r` in
scripts or the viewer's reversal control. They also support continuous color
sampling for waterfall traces. The multi-sequential terrain maps, categorical
variants, cyclic maps, and diverging maps are separate upstream families and
are not included in this section. See
[Scientific Colour Maps](https://www.fabiocrameri.ch/colourmaps/) for their design.

The cmocean section offers 14 sequential maps with the `cmo.` prefix:
`thermal`, `haline`, `solar`, `ice`, `gray`, `deep`, `dense`, `algae`,
`matter`, `turbid`, `speed`, `amp`, `tempo`, and `rain`. It uses cmocean's
original 256-color tables. The oxygen-threshold and terrain maps are omitted
because their transitions have specialized meanings.

The MyCarta section supplies `mycarta.Cube1`, `mycarta.CubeYF`, and
`mycarta.LinearL` using all 256 RGB entries provided by Palettable. The
CartoColors section supplies `carto.SunsetDark`, `carto.BluYl`, and
`carto.TealGrn`, interpolated from their seven-color palettes. All these maps
support the same reversal, saved-plot, script, and waterfall sampling controls
as other continuous maps. Their original directions are preserved; use
Reverse if you prefer low values dark and high values light.

Stored plots and duplicated viewers restore the color controls to match the
rendered plot, including the reversed colormap, scale, alpha color shift,
automatic-limit method and its numeric parameters, power exponent, and
automatic or manual limits.
Older recipes without numeric color parameters use the viewer defaults.

### Custom colormap files

In the main window, choose **File → Preferences… → Colormaps → Open folder…**.
The page also shows the folder path, loaded custom palettes, and file-format
instructions. Alternatively, right-click any Qt colormap selector and choose
**Open custom colormap folder…**. Both open nfit's application-data folder
(`~/Library/Application Support/nfit/colormaps` on macOS, `%APPDATA%/nfit/colormaps`
on Windows, or `$XDG_CONFIG_HOME/nfit/colormaps` on Linux), or the directory set
by `NFIT_COLORMAP_DIR`. Opening this folder moves an existing
`~/nfit_colormaps` folder there when possible.
Drop in a `.csv`, `.txt`, or `.rgb` file and restart nfit. Custom maps appear
in a final section with preview swatches in slice, tiled-slice, volume, and
waterfall menus. Malformed files are skipped with a warning in the console.

Use at least two rows of three RGB values, ordered from low to high, separated
by commas or whitespace. Values may be floats in 0–1 or integers in 0–255;
lines beginning with `#` are comments. There is no column header. For example:

```text
# RGB, 0–1
0.0 0.0 0.3
0.0 0.6 0.6
1.0 1.0 0.5
```

The filename determines the name: `parula.csv` becomes `user_parula`.
Use letters, digits, underscores, or hyphens; do not end the name with `_r`,
which is reserved for reversal. Export a Parula table from MATLAB with
`writematrix(parula(256), 'parula.csv')` and place it in this folder.

Scripts discover the same folder on import. To load an explicit file elsewhere:

```python
from nfit.colormaps import load_colormap_file
name = load_colormap_file("/path/to/parula.csv")
# Pass cmap=name to a plotting function, or cmap=name + "_r" to reverse it.
```

Saved plots and exported scripts reference the colormap by name. Include the
RGB file when sharing them with another machine; RGB tables are not embedded
in project files. Replacing a table changes future renderings of that name.

## Waterfall plots

Choose **Visualization > Waterfall** to stack one-dimensional traces.

- For multidimensional data, choose the horizontal and waterfall axes; other
  dimensions retain their ordinary point/range controls.
- For a one-dimensional dataset, compatible datasets in the same immediate
  project group provide the traces. Choosing any sibling dataset switches to
  that sibling's immediate group.
- **Bin width** controls coarsening along the waterfall axis; **Auto** targets
  roughly ten traces.
- **Minimum coverage** is specific to waterfall trace bins. It is independent
  of the slice and histogram cutoff.

Trace offset, colors, marker fill, zero references, labels, and model overlays
are configurable. Saved plot recipes and generated scripts retain these
settings and all contributing dataset references. Trace markers are unfilled by
default.

The axis **Reset** buttons and Matplotlib **Home** button use the current trace
extent. That extent is recomputed when the dataset group, displayed axes, or
waterfall binning changes.

## Tiled 2D slices

Choose **Visualization > Tiled slices** for gridded data with at least three
dimensions containing more than one bin. Choose the horizontal and vertical
dimensions as for a normal map, then choose a distinct **Third dimension** in
the tiled-slice controls. Any further dimensions retain the ordinary
point/range integration controls.

**Range low** and **Range high** specify the first and maximum panel centers.
**Step size** sets both center spacing and integration-window width. For example,
`-0.2` to `1.4` with step `0.2` gives nine panels centered on those values,
with windows from `-0.3` to `-0.1`, `-0.1` to `0.1`, and so on. Windows include
source-bin centers, with shared boundaries assigned to the later panel; labels
show the requested center even when coverage is asymmetric. Empty windows are
omitted. The horizontal slider
spans one native bin through the selected range. **Auto** makes nine panels
when at least nine third-axis values are available, and otherwise makes one
panel per available value.

By default, every panel is labeled at its bottom-right corner with its
representative third-axis value, backed by a translucent square-cornered box so
that it remains legible over the data. Clear **Show tiled-slice value labels**
in **Figure** to hide these annotations.

Tiled-slice labels can be customized in **Figure**: **Decimals** selects 0–10
decimal places (default 1), **Label text** replaces the text before the value,
and **Unit** replaces its unit string. Use `{axis} = ` and `{unit}` for the
automatic axis name and unit, or empty text to omit either. **SI prefix**
rescales only the displayed number and prepends the prefix to the unit: for
an axis in kelvin, selecting `m` displays millikelvin. Scaling is relative to
the original axis unit, including any prefix it already contains; changing
the unit text alone does not convert values. These options do not affect data,
binning, or integration and are retained in saved plots and exported scripts.

All panels share their x and y limits. The default global color scale also
shares one normalization and one colorbar at the far right. With **Autoscale**
enabled, **Local scale per tiled plot** gives every panel independently computed
limits and its own adjacent colorbar. Local scales are unavailable with manual
color limits; turning off **Autoscale** returns the view to the global scale.
Only local colorbars in the rightmost grid column carry the channel or
cross-section label, avoiding repeated labels between neighboring panels.
The mode intentionally displays data only; model and residual panels and
histogram box cuts are not shown. Saved plot recipes and generated scripts
preserve the three dimensions, range, step, label visibility, hidden-axis
selections, and global/local color setting.

Hovering over any panel updates the coordinate inspector with that panel's
horizontal and vertical bin centers, tiled-axis coordinate, intensity,
uncertainty, and coverage. Colorbars and unused grid space do not change the
readout.

For slice and tiled plots, editing either **vmin** or **vmax** turns off
**Autoscale** and keeps the currently displayed value for the other bound.
Thus either field can be changed independently and takes effect immediately.
Turning off **Autoscale** also locks both displayed limits while using the
Matplotlib Home, Back, or Forward buttons; those buttons change the data view
without silently restoring an older color normalization.
When the optional other-axis binning title is visible, it summarizes only
dimensions not represented by the horizontal, vertical, or tiled axes. For a
single selected slice, the title shows the selected bin's edges; for an
integrated range, it shows the selected lower and upper bin centers.

## Volumetric mode

Gridded datasets with at least three dimensions can use **Volumetric** mode.
Point-list data must first be rebinned.

Before PyVista is initialized, nfit warns when the initial three-dimensional
volume exceeds 2,000,000 bins. The count covers only the three axes that will
be rendered; remaining dimensions begin as selected slices. Choose **Proceed**
to render anyway or **Cancel** to remain in the current viewer mode. A coarser
rebin or narrower axis range reduces rendering work.

Select three distinct displayed axes. Remaining dimensions use the same bin or
range integration controls as the slice viewer. Axis limits crop by bin center;
**Equal data units** preserves physical axis ratios, while **Custom** enables
visual stretching without changing coordinates or scalar values.

**Volume** performs direct volume rendering. **Isosurface** extracts a surface
at the selected opacity level. Color and opacity channels may be linked or
controlled independently with editable transfer curves. Masked and non-finite
cells remain transparent.

Exports include:

- PNG for the current camera view;
- VTK PolyData, PLY, STL, or glTF for an isosurface;
- VTK rectilinear grid for volumetric scalar data; and
- MP4 for a camera orbit around a selected displayed axis.

Smoothing and axis scaling affect rendered images and movies. Numerical volume
and surface exports use unsmoothed channel values.

## Saved plots

When editing a stored plot in the viewer, **Save plot** updates that recipe;
**Save new plot** creates an independently named copy, preserving the original
and its rebin settings. The viewer then edits the new copy, so subsequent
**Save plot** actions update it. Fresh data viewers retain **Store plot** rather
than these two editing actions. Plot recipes are saved to disk with the project;
these actions do not export an image file.

Each workspace has a **Plots** branch. In the viewer, **Store plot** stores the
current figure as an editable recipe and selects the resulting tree entry. Each
recipe snapshots the complete rebin configuration of every source dataset,
including coordinate vectors, bounds, step sizes or bin counts, symmetry,
fractional weighting, coverage thresholds, and batching choices. Later dataset
rebin edits therefore do not change an existing plot, and multiple plots can
use different projected coordinate bases from the same source data. Composite
views, including tiled slices, retain the composite scope and its complete
rebin recipe while continuing to reference the underlying datasets. A saved
plot can be:

- reopened in a presentation window;
- restored into the data viewer;
- copied as a script; or
- saved as a standalone script.

Scripts require a saved project so dataset references have a stable path. They
use the public plotting API and can run without constructing the project
explorer.

See [Saved plots](plotting.md) for the serialization and scripting interface.
