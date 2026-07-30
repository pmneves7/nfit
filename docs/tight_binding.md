# Tight-binding electronic structure

`tight_binding` represents a material-independent orthonormal electronic
Hamiltonian. It accepts manual real-space models or Wannier90 Hamiltonians and
calculates bands, orbital projections, density of states, and Fermi surfaces.
It does not assume a particular lattice, material, orbital count, or spin
layout.

## Energy units

Electronic inputs and plots default to eV. The canonical `ElectronicModel`
stores every Hamiltonian, eigenvalue, chemical potential, and electronic
interaction energy in meV so it can feed neutron-response calculations without
another unit boundary. Conversion occurs once when a model is built or
imported and is recorded in provenance.

The GUI has one **Electronic energy unit** selector for coherent electronic
settings. Changing it between eV and meV changes entry fields and electronic
plot labels, but not stored physical values. Low-level calculation arguments
whose names end in `_meV` are canonical and are never interpreted in the
display unit. Unit-neutral package adapters must declare their input unit;
manual scripts should pass it explicitly whenever the source convention may
not be eV. nfit does not guess from magnitude.

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

The builder exposes named linear Hamiltonian terms through the common value,
bounds, sharing, and fit-selection machinery. An optimizer consumes them only
after an electronic-response component supplies a dataset observable.
Canonical-model fields below are in meV; manual-builder inputs use their
declared `energy_unit`.

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
| `metadata` | additional JSON-compatible annotations | `{"manifold": "M1_d"}` |

Labels identify array indices and must be unique. Orbital names and shell
labels are descriptive in Stage 3; nfit does not infer degeneracy or symmetry
from their spelling.

### Canonical electronic-model fields

| Field | Meaning and convention | Acceptable input example |
| --- | --- | --- |
| `direct_lattice` | $3\times3$ matrix whose columns are direct-lattice vectors in Å | `np.diag([4.0, 4.0, 12.0])` |
| `basis` | ordered nonempty sequence of `BasisState`, dictionaries, or labels | `[BasisState("M1_d")]` |
| `translations` | canonical integer cell translations $\mathbf R$ | `[[0, 0, 0], [1, 0, 0], [-1, 0, 0]]` |
| `hamiltonian_blocks` | one complex $N\times N$ canonical matrix $H(\mathbf R)$ in meV per translation | an array with shape `(3, N, N)` |
| `interpolation_weights` | positive $w_{\mathbf R}$ multiplying each block | `[1.0, 1.0, 1.0]` |
| `orbital_centers` | one fractional direct-lattice position per basis state | `[[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]` |
| `periodic_axes` | ordered periodic lattice-axis indices | `(0,)`, `(0, 1)`, or `(0, 1, 2)` |
| `parameter_values` | canonical meV values of named linear Hamiltonian terms | `{"t_nn": -80.0}` |
| `parameter_blocks` | derivative blocks paired one-to-one with `parameter_values` | `{"t_nn": dH_dt}` |
| `spin_operators` | optional Hermitian $S_x,S_y,S_z$ matrices | complex array with shape `(3, N, N)` |
| `energy_zero_meV` | recorded reference energy; it does not shift $H$ automatically | `0.0` |
| `canonical_energy_unit` | serialized energy unit, fixed by the model contract | `"meV"` |
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
| `model_stale` | derived cache state; builder edits set it and the next calculation or script export rebuilds `model_data` | `false` | `true` |
| `use_primitive_cell` | fold equivalent conventional-cell orbitals and hoppings onto the primitive translation cell before diagonalization | `true` | `false` for diagnostic comparison |
| `crystal` | editable lattice, space group, crystallographic sites, and optional CIF provenance used by the structure-first builder | `P 1` cell with no sites | `{"lattice": {"a": 4, "b": 4, "c": 6, "alpha": 90, "beta": 90, "gamma": 90}, "spacegroup": "P 1", "sites": []}` |
| `orbital_manifolds` | editable site-attached basis definitions and local frames | `[]` | `[orbital_manifold_preset("M1", "d").to_dict()]` |
| `spin_treatment` | requested spin representation; `auto` remains implicit unless SOC requires spinors | `"auto"` | `"auto"`, `"implicit"`, `"collinear"`, or `"spinor"` |
| `soc_terms` | optional manifold-resolved onsite $\lambda\mathbf L\cdot\mathbf S$ terms | `[]` | `[spin_orbit_term("M1_d", value_meV=25).to_dict()]` |
| `onsite_terms` | generated Hermitian onsite matrix bases and mirrored canonical parameter state | `[]` | `[term.to_dict() for term in generate_onsite_terms(crystal, manifolds)]` |
| `hopping_cutoff_angstrom` | maximum representative-bond distance used by the symmetry hopping generator; zero disables generated hoppings | `0.0` | `4.2` |
| `hopping_parameterization` | generated hopping basis: compact two-centre integrals or the complete symmetry-allowed real matrix basis | `"slater_koster"` | `"slater_koster"` or `"general"` |
| `spatial_orbits` | generated symmetry-equivalent bond families retained for editing and visualization | `[]` | `[orbit.to_dict() for orbit in generation.orbits]` |
| `hopping_candidates` | complete generated list of symmetry-allowed matrix terms; candidates do not affect $H(\mathbf k)$ | `[]` | `[term.to_dict() for term in generation.terms]` |
| `hopping_terms` | selected active hopping terms with mirrored canonical parameter state | `[]` | `[selected_term.to_dict()]` |
| `periodic_axes` | periodic lattice axes; empty asks a Wannier import to infer them from nonzero translations | `[]` | `[0]`, `[0, 1]`, or `[0, 1, 2]` |
| `electronic_energy_unit` | input and electronic-plot unit; changing it does not alter canonical values | `"eV"` | `"eV"` or `"meV"` |
| `electronic_backend` | execution policy for bands, DOS, and Fermi surfaces | `"auto"` | `"auto"`, `"numpy"`, `"threaded"`, or `"cupy"` |
| `electronic_workers` | maximum CPU workers; zero uses nfit's allocation and `NFIT_NUM_THREADS` | `0` | `8` |
| `electronic_max_batch_mb` | target temporary memory per Hamiltonian/eigensystem batch | `256.0` | `512.0` |
| `chemical_potential_meV` | canonical chemical potential subtracted on band and DOS plots; displayed in `electronic_energy_unit` | `0.0` | `12.5` for 0.0125 eV |
| `projection_groups` | plot labels mapped to zero-based basis indices | `{}` | `{"d": [0, 1, 2], "p": [3, 4]}` |
| `band_path` | ordered nodes with labels and primitive reduced reciprocal coordinates | $\Gamma$–X–M–$\Gamma$ | `[{"label": "G", "k": [0, 0, 0]}, {"label": "X", "k": [0.5, 0, 0]}]` |
| `band_path_convention` | origin of the configured path | `"hinuma"` | `"hinuma"` or `"manual"` |
| `band_path_metadata` | provider, version, convention, and symmetry tolerance for an automatic path | `{}` until a path is generated | `{"provider": "seekpath", "provider_version": "2.2.1", "convention": "HPKOT", "symprec": 1e-5}` |
| `band_points_per_segment` | interpolation intervals in each path segment | `60` | `80` |
| `dos_mesh` | uniform mesh sizes, one per periodic axis or one per lattice axis | `[40, 40, 40]` | `[80, 80]` for a two-dimensional model |
| `dos_symmetry` | total-DOS mesh policy: full mesh, certified automatic reduction, or required certified reduction | `"full"` | `"full"`, `"auto"`, or `"reduced"` |
| `dos_energy_min_meV` | canonical lower absolute energy sampled for the DOS | `-500.0` | `-250.0` |
| `dos_energy_max_meV` | canonical upper absolute energy sampled for the DOS | `500.0` | `250.0` |
| `dos_energy_points` | number of DOS energy samples, at least two | `600` | `1000` |
| `dos_broadening_meV` | canonical Gaussian standard deviation used for the DOS | `5.0` | `2.0` |
| `fermi_mesh` | extraction-grid sizes, each at least two | `[64, 64, 64]` | `[200, 200]` for a two-dimensional model |
| `fermi_energy_meV` | canonical absolute target energy of the extracted constant-energy surface | `0.0` | `12.5` to extract the Fermi surface when $\mu=12.5$ meV |

For a reduced-dimensional model, a mesh may contain one size per periodic
axis. A three-entry mesh instead gives sizes in lattice-axis order, and nfit
selects the declared periodic axes. `chemical_potential_meV` changes the
displayed energy origin; it does not solve for a filling. Set
`fermi_energy_meV` equal to the chosen chemical potential for a true Fermi
surface.

## Calculable data

The model calculates:

- the first Brillouin zone, labelled reciprocal vectors, and the configured
  high-symmetry path;
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

The tight-binding component itself remains an electronic-structure provider,
not an additive dataset observable. A separate
[bare Lindhard response](lindhard.md) can reference it to calculate neutron
intensity or bulk susceptibility and to fit the electronic coefficients
through that response. Separate scalar Stoner, user-matrix, and
Hubbard--Hund RPA components can dress the bare response.

## Structure-first orbital, onsite, and hopping builder

Stages 3.2 and 3.3 construct a static tight-binding Hamiltonian from a CIF or
manually entered crystal. `OrbitalManifold` attaches an ordered basis to a
crystallographic representative site. The space group expands it to every
equivalent site. Each generated basis state retains its site, element,
orbital, manifold, correlated-shell label, and fractional center.

### Orbital-manifold fields

| Field | Meaning | Acceptable input example |
| --- | --- | --- |
| `site_label` | representative crystal site receiving the manifold | `"M1"` |
| `label` | unique editable manifold identifier | `"M1_d"` |
| `basis_kind` | transformation convention | `"real_harmonic"`, `"complex_harmonic"`, `"effective_scalar"`, `"custom"`, or `"wannier"` |
| `orbitals` | ordered basis labels | `["d_xy", "d_yz", "d_zx", "d_x2_y2", "d_z2"]` |
| `l` | angular-momentum quantum number for a harmonic basis | `2` |
| `irrep` | descriptive site-symmetry subspace label | `"4/mmm subspace 2 (dimension 2; d_yz, d_zx)"` |
| `degeneracy_groups` | optional groups constrained to share a diagonal onsite energy | `[["d_yz", "d_zx"]]` |
| `local_frame` | right-handed orthonormal local axes as columns in crystal Cartesian coordinates | `[[1,0,0], [0,1,0], [0,0,1]]` |
| `spin_basis` | basis spin convention; Stage 3.2 supports spinless | `"spinless"` |
| `correlated_shell` | label used by later interaction dressings | `"M1_3d"` |
| `symmetry_mode` | automatic analytic representation or no inferred symmetry | `"automatic"` or `"none"` |
| `harmonic_transform` | orthonormal columns mapping selected orbitals into the complete $l$ shell | a site-symmetry projector basis with shape $5\times2$ |
| `preset` | convenience recipe that created the record | `"d"` or `"site_symmetry"` |
| `site_point_group` | identified point group when a site-symmetry subspace was selected | `"4/mmm"` |
| `submanifold_id` | stable identifier of that calculated subspace | `"d:4/mmm:8705c5723841"` |

The always-available basis choices are `effective`, complete `s`, `p`, `d`,
and `f` shells, and `custom`. The real $p$ order is $(p_x,p_y,p_z)$ and the
real $d$ order is
$(d_{xy},d_{yz},d_{zx},d_{x^2-y^2},d_{z^2})$. Complete shells do not impose
accidental degeneracy: the site symmetry determines their allowed splitting,
unless the user adds a `degeneracy_groups` constraint.
For `complex_harmonic`, rows of `harmonic_transform` follow
$m=-l,-l+1,\ldots,l$.

The local frame defaults to the crystal Cartesian frame. The GUI identifies
the selected site's crystallographic point group from its stabilizer. For a
complete $s$, $p$, $d$, or $f$ shell, the user may then opt into one of the
symmetry-closed subspaces calculated from that point-group representation.
This is a general decomposition: nfit does not assume a particular cubic,
trigonal, or other named crystal-field scheme. A reported subspace gives its
dimension and dominant complete-shell orbitals for identification. It does
not claim a conventional irrep name when nfit has not established one.
For the supported trigonal site groups, nfit assigns conventional labels such
as $A_{1g}$ and $E_g$ only after checking the representation characters.
Repeated copies of the same irrep receive deterministic `copy 1`, `copy 2`,
and orbital-weight descriptions. These copy numbers identify the displayed
basis choice; they are not extra symmetry quantum numbers. Symmetry permits
different copies of the same irrep to mix, so two displayed $E_g$ subspaces
remain physically the same irrep.

For site groups that act only as a scalar on the selected shell, including
the trivial group, nfit returns the complete shell rather than inventing
arbitrary one-dimensional crystal-field levels.

For a site operation $g$, nfit evaluates the spherical-harmonic
representation $D(g)$ in the declared local frame. A selected subspace with
coefficient matrix $C$ is valid for automatic symmetry only when
$D(g)C$ lies in the span of $C$ for every operation in the site stabilizer.
A non-closed subspace is rejected with a diagnostic; it is never silently
projected. A custom or imported numerical basis remains usable with
`symmetry_mode="none"`, but nfit does not invent its symmetry character.

### Onsite-invariant fields

The builder finds the complete Hermitian matrix space satisfying

$$
D(g)P_pD(g)^\dagger=P_p
$$

for every site-stabilizer operation, together with any declared diagonal
degeneracies. The static onsite block is

$$
H_{\mathrm{onsite}}=\sum_p\epsilon_pP_p.
$$

| Field | Meaning | Acceptable input example |
| --- | --- | --- |
| `identifier` | stable hash of site, ordered basis, and invariant matrix | `"M1:onsite:6054a1a6e2ad"` |
| `label` | readable term label | `"M1 epsilon_1"` |
| `site_label` | representative site on which the term acts | `"M1"` |
| `basis_labels` | ordered local basis addressed by the matrix | `["M1_d:d_xy", "M1_d:d_yz", "M1_d:d_zx"]` |
| `matrix` | unit-Frobenius Hermitian invariant, serialized as real/imaginary pairs | a $3\times3$ identity-like selector |
| `kind` | diagonal onsite energy or allowed onsite hybridization | `"onsite_energy"` or `"onsite_hybridization"` |
| `value_meV` | canonical coefficient | `25.0` |
| `bounds_meV` | canonical fit bounds mirrored from `component.limits` | `[-100.0, 100.0]` |
| `fit` | fit selection mirrored from `component.fit_parameters` | `false` |
| `source` | origin of the matrix constraints | `"site_symmetry"` or `"declared_degeneracy"` |

The GUI displays values and bounds in `electronic_energy_unit` and converts
them immediately to canonical meV. It updates the compact builder and marks
the derived canonical model stale. The next band, DOS, Fermi-surface, matrix,
response, or script calculation rebuilds `model_data` once and caches the
immutable result. Repeated calculations reuse that result until another
scientific input changes. Fit selections and sharing rules update without
rebuilding the Hamiltonian. With no hopping terms, the resulting bands are
flat.

```python
from nfit import (
    add_tight_binding_orbital_manifold,
    create_model_component,
    orbital_manifold_from_site_symmetry,
    set_model_crystal,
    set_tight_binding_onsite_term,
    site_symmetry_harmonic_submanifolds,
)

model = create_model_component(group, "electrons", type="tight_binding")
set_model_crystal(model, crystal, group=group)
subspaces = site_symmetry_harmonic_submanifolds(crystal, "M1", "d")
add_tight_binding_orbital_manifold(
    model,
    orbital_manifold_from_site_symmetry(
        crystal,
        "M1",
        "d",
        subspaces[0].identifier,
        local_frame=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        correlated_shell="M1_3d",
    ),
)
term_id = model.config["onsite_terms"][0]["identifier"]
set_tight_binding_onsite_term(
    model,
    term_id,
    value=0.025,
    energy_unit="eV",
    lower=-0.2,
    upper=0.2,
)
```

### Hopping-invariant fields

The **Hoppings** section offers two parameterizations. **Slater--Koster
integrals** is the compact default for analytic `s`, `p`, `d`, and `f`
manifolds, including symmetry-selected subspaces. **General symmetry
matrices** finds a complete real matrix basis $B_p$ allowed by the
representative bond stabilizer:

$$
T_{\mathrm{rep}}=\sum_p t_p B_p.
$$

For Slater--Koster hopping, nfit makes a bond frame with local $z$ along the
representative bond and constructs

$$
T_{\mathrm{rep}}
=\sum_{\mu=\sigma,\pi,\delta,\phi}
V_{l_i l_j\mu}\,
C_i^\dagger D_i^\dagger P_\mu D_j C_j.
$$

$P_\mu$ selects equal bond-axis magnetic quantum numbers with
$|m|=0,1,2,3$, respectively. $D_i$ and $D_j$ rotate the endpoint local frames
into the bond frame, and $C_i$ and $C_j$ project complete harmonic shells into
the selected orbital subspaces. A coefficient such as $V_{pd\pi}$ is the
axial two-centre matrix element itself, not a unit-Frobenius rescaling. Terms
whose projection is identically zero are omitted. Effective scalar orbitals
participate as $l=0$ states. The current generator uses nfit's real-harmonic
convention. Complex-harmonic, custom, and Wannier bases require the general
convention because nfit does not infer or silently change their phase
conventions.
Distinct images of each Slater--Koster seed under the bond stabilizer are
combined into one coefficient. This ties endpoint-reversed blocks when
required and prevents the compact convention from violating space-group
covariance.

If a space-group operation $g$ maps the representative bond to another orbit
member, its hopping matrix is

$$
T_{g(i)g(j)}(\mathbf R_g)
=D_i(g)T_{\mathrm{rep}}D_j(g)^\dagger.
$$

Reversing the directed bond transposes the current real matrix. The resolved
Hamiltonian contains both directions and therefore obeys
$H(-\mathbf R)=H(\mathbf R)^\dagger$ exactly.
Generation populates `hopping_candidates`. The user selects which candidates
to add to `hopping_terms`; only those active terms enter the Hamiltonian.
General-matrix names retain the compact orbit and basis index while
summarizing active orbital support, for example
`B2 t1: V1_d[d_xy,d_yz,+3] ← Li1_s[s]`. Slater--Koster names state the
integral and manifolds, for example
`B2 V_pdπ: V1_d ← O1_p`. The arrow follows the convention that
$T_{ij}(\mathbf R)$ maps orbitals on site $j$ in cell $\mathbf R$ to site $i$
in the home cell.

| Field | Meaning | Acceptable input example |
| --- | --- | --- |
| `identifier` | stable hash of the orbit, ordered endpoint bases, and invariant matrix | `"B1:hopping:2a9374d7d0f1"` |
| `label` | readable parameter label | `"B1 t_1"` |
| `orbit_label` | spatial bond-orbit identifier | `"B1"` |
| `distance_angstrom` | representative center-to-center distance in Å | `3.9` |
| `representative_bond` | endpoint indices and integer cell offset for the canonical representative | `{"site_i": 0, "site_j": 0, "offset": [1, 0, 0]}` |
| `basis_i` | ordered basis labels at the receiving endpoint | `["M1_d:d_xy", "M1_d:d_yz"]` |
| `basis_j` | ordered basis labels at the sending endpoint | `["M1_d:d_xy", "M1_d:d_yz"]` |
| `matrix` | real hopping selector, serialized as real/imaginary pairs; unit-Frobenius for the general basis and literal axial normalization for Slater--Koster | `[[[0.7071, 0], [0, 0]], [[0, 0], [0.7071, 0]]]` |
| `value_meV` | canonical coefficient $t_p$ | `-80.0` |
| `bounds_meV` | canonical fit bounds mirrored from `component.limits` | `[-200.0, 20.0]` |
| `fit` | fit selection mirrored from `component.fit_parameters` | `false` |
| `source` | origin of the matrix constraints | `"slater_koster"` or `"spinless_time_reversal_space_group"` |

The spatial hopping generator is orbital-only, real, and time-reversal
symmetric. Collinear and spinor models lift each generated hopping as
$B_p\otimes I_2$, so adding spin does not duplicate its coefficient.
Spin-dependent hopping remains an advanced opt-in extension rather than part
of the default generated basis. Manual and Wannier90 models may contain
general complex matrices. A custom numerical basis can use the general
generator only when every required site mapping is the identity; otherwise
explicit representation matrices remain a later extension. Changing
parameterization regenerates suggestions. Active terms survive only when
their stable identifiers also exist in the new convention.

```python
from nfit import (
    add_tight_binding_hopping_term,
    regenerate_tight_binding_hopping_terms,
    set_tight_binding_hopping_parameterization,
    set_tight_binding_hopping_term,
)

set_tight_binding_hopping_parameterization(model, "slater_koster")
generation = regenerate_tight_binding_hopping_terms(
    model,
    cutoff_angstrom=4.2,
)
hopping_id = generation.terms[0].identifier
add_tight_binding_hopping_term(model, hopping_id)
set_tight_binding_hopping_term(
    model,
    hopping_id,
    value=-0.080,
    energy_unit="eV",
    lower=-0.2,
    upper=0.02,
)
```

### Primitive-cell resolution

For a three-dimensional GUI-built model, `use_primitive_cell=true` folds a
centered conventional construction onto the primitive translation lattice
before spin expansion and diagonalization. Orbitals are grouped by their
wrapped primitive-cell center, manifold, orbital, species, correlated shell,
and spin label. Hamiltonian and named-parameter blocks are translated and
summed into that basis. nfit requires each group to contain exactly the
crystallographic cell multiplicity; an incomplete or inequivalent basis is an
error rather than an approximate fold.

The resolved provenance records the cell multiplicity and input and output
basis sizes. Disable the option only when diagnosing the unreduced
construction. This reduces matrix dimension; it is distinct from the planned
optimization that will sample only the symmetry-unique part of a
Brillouin-zone mesh.

## Spin representations and spin–orbit coupling

Spin is optional computational structure, not another spatial-orbital preset.
The **Spin and SOC** section resolves one of three representations:

| Resolved treatment | Basis size | Use |
| --- | --- | --- |
| `implicit` | $N$ | default SU(2)-symmetric paramagnet; spin degeneracy and spin traces are applied analytically by later response calculations |
| `collinear` | $2N$ | explicit up/down blocks without spin mixing |
| `spinor` | $2N$ | SOC, transverse spin operators, and other terms that mix spin components |

`spin_treatment="auto"` resolves to `implicit` when no SOC term is active and
to `spinor` when one is added. Explicit `implicit` or `collinear` treatment is
rejected if SOC is active because neither can represent
$\mathbf L\cdot\mathbf S$. This makes the inexpensive $N\times N$
diagonalization the normal case.

Explicit bases use orbital-major ordering,

$$
(a\uparrow,a\downarrow,b\uparrow,b\downarrow,\ldots).
$$

Every pre-existing orbital Hamiltonian block and parameter selector is lifted
without creating new coefficients:

$$
H_{\mathrm{spin}}(\mathbf R)
  =H_{\mathrm{orb}}(\mathbf R)\otimes I_2,\qquad
P_{p,\mathrm{spin}}(\mathbf R)
  =P_{p,\mathrm{orb}}(\mathbf R)\otimes I_2.
$$

The generated dimensionless spin operators are

$$
S_\alpha=I_N\otimes\frac{\sigma_\alpha}{2},
$$

where $\sigma_\alpha$ are Pauli matrices. `BasisState.spin` is empty for an
implicit model and is `"up"` or `"down"` for an explicit basis. The 3D
geometry viewer continues to show one token per spatial orbital; it does not
draw two overlapping spheres for the two spin components.

### Onsite SOC

Each enabled manifold contributes a named linear term

$$
H_{\mathrm{SOC},m}
  =\lambda_m\sum_{\alpha=x,y,z}
  L_{m,\alpha}\otimes S_\alpha.
$$

$\lambda_m$ is entered in `electronic_energy_unit`, stored canonically in
meV, and uses the same bounds, fit selection, and dataset-sharing machinery as
onsite and hopping coefficients. `SpinOrbitTerm` defines:

| Field | Meaning | Acceptable input example |
| --- | --- | --- |
| `identifier` | stable parameter name | `"M1_d:soc:lambda"` |
| `label` | readable description | `"M1_d λ L·S"` |
| `manifold_label` | spatial manifold receiving SOC | `"M1_d"` |
| `value_meV` | canonical $\lambda$ | `25.0` |
| `bounds_meV` | canonical optional fit bounds | `[0.0, 100.0]` |
| `fit` | mirrored fit selection | `false` |
| `prescription` | construction of $\mathbf L$ | `"auto"`, `"atomic"`, `"projected"`, or `"effective"` |
| `orbital_operators` | explicit local $L_x,L_y,L_z$ for an effective/custom basis | complex array with shape `(3, n, n)` |

The prescriptions have distinct physical meanings:

- **Atomic** uses the exact angular-momentum matrices of a complete
  $(2l+1)$-orbital spherical-harmonic shell. It is rejected for a truncated
  shell.
- **Projected** uses $L_\alpha^{(P)}=P L_\alpha P$ in a selected
  site-symmetry subspace. It is internally consistent, but orbital angular
  momentum can be quenched and virtual coupling through excluded orbitals is
  absent.
- **Effective** uses explicitly supplied Hermitian $L_x,L_y,L_z$ matrices.
  This is the supported route for effective, custom numerical, and suitably
  characterized Wannier-like local bases; nfit does not infer them from
  orbital names.
- **Auto** selects atomic for a complete analytic shell and projected for an
  analytic subspace.

`harmonic_transform` maps the selected analytic orbitals into the documented
complete real- or complex-harmonic order. nfit constructs $\mathbf L$ in that
convention, projects it, and rotates its Cartesian components from the
manifold's local frame into the crystal frame. Symmetry-equivalent sites use
their generated local frames.

```python
from nfit import (
    set_tight_binding_soc_term,
    set_tight_binding_spin_treatment,
    spin_orbit_term,
)

# Efficient default: remains N dimensional until SOC is enabled.
set_tight_binding_spin_treatment(model, "auto")
set_tight_binding_soc_term(
    model,
    "M1_d",
    enabled=True,
    prescription="auto",
    value=0.025,
    energy_unit="eV",
    lower=0.0,
    upper=0.10,
)

# Equivalent serializable term for configure_tight_binding_spin.
soc = spin_orbit_term("M1_d", value_meV=25.0)
```

For a nonmagnetic spinor Hamiltonian, nfit validates

$$
H(\mathbf k)
=U_\Theta H(-\mathbf k)^*U_\Theta^\dagger,\qquad
U_\Theta=I_N\otimes i\sigma_y.
$$

`spinor_time_reversal_residual` exposes the normalized residual and
`validate_spinor_time_reversal` applies the validation threshold. For a
spatial operation $g$, the double-group representation is

$$
D_{\mathrm{spinor}}(g)
=D_{\mathrm{orbital}}(g)\otimes D_{1/2}(g).
$$

`spinor_rotation_representation` constructs $D_{1/2}$, treating spin as an
axial vector under improper spatial operations, while
`spinor_manifold_representation` combines it with the orbital action. The
default spin-independent lifting does not regenerate a large set of separate
up/down hopping invariants.

## Manual construction

`build_electronic_model` accepts arbitrary complex onsite and hopping matrices.
For convenience, it can add a missing Hermitian-conjugate $-\mathbf R$ block.
If both partners are supplied, inconsistency is an error.
Optional `parameter_values` and `parameter_hoppings` define named linear
Hamiltonian terms. `energy_unit` defaults to eV and applies to `hoppings`,
`parameter_values`, and `energy_zero`; dimensionless `parameter_hoppings`
select the affected matrix elements. `model.with_parameters(...)` returns a
new immutable model and also requires an explicit `energy_unit`.

| Argument | Meaning | Acceptable input example |
| --- | --- | --- |
| `direct_lattice` | lattice-vector matrix described above | `np.diag([4.0, 4.0, 12.0])` |
| `basis` | ordered basis states | `[BasisState("M1_d"), BasisState("X1_p")]` |
| `hoppings` | map from integer $\mathbf R$ to $N\times N$ $H(\mathbf R)$ in `energy_unit` | `{(0, 0, 0): [[0.0]], (1, 0, 0): [[-0.08]]}` in eV |
| `orbital_centers` | fractional center of each basis state | `[[0, 0, 0], [0.5, 0.5, 0]]` |
| `periodic_axes` | periodic lattice axes | `(0, 1)` |
| `interpolation_weights` | optional map from $\mathbf R$ to positive weight | `{(0, 0, 0): 1.0}` |
| `parameter_values` | named linear-term energies in `energy_unit` | `{"t_nn": -0.08}` in eV |
| `parameter_hoppings` | named maps from $\mathbf R$ to selector matrices | `{"t_nn": {(1, 0, 0): [[1.0]]}}` |
| `spin_operators` | optional three spin matrices | `np.asarray([Sx, Sy, Sz])` |
| `energy_unit` | unit of manual Hamiltonian energies; converted immediately to canonical meV | `"eV"` or `"meV"` |
| `energy_zero` | recorded reference energy in `energy_unit` | `0.0` |
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
    energy_unit="eV",
    hoppings={
        (0, 0, 0): [[-0.020, 0], [0, 0.030]],
        (1, 0, 0): [[-0.080, 0], [0, -0.040]],
        (0, 1, 0): [[-0.080, 0], [0, -0.040]],
    },
    parameter_values={"hybridization": 0.015},
    parameter_hoppings={
        "hybridization": {
            (0, 0, 0): [[0, 1], [1, 0]],
        },
    },
)

stronger_hybridization = model.with_parameters(
    hybridization=0.020,
    energy_unit="eV",
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
    electronic_energy_to_meV,
    fermi_surface,
    k_mesh,
)

# These manual nodes are the conventional path for this simple-cubic example.
path = band_path(
    model,
    [[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0, 0]],
    labels=[r"$\Gamma$", "X", "M", r"$\Gamma$"],
)
bands = calculate_bands(model, path, projections={"d": [0], "p": [1]})

mesh = k_mesh(model, [80, 80])
energy_unit = "eV"
energy = electronic_energy_to_meV(
    np.linspace(-0.4, 0.4, 1000),
    energy_unit,
)
dos = density_of_states(
    model,
    mesh,
    energy,
    broadening_meV=electronic_energy_to_meV(0.003, energy_unit),
    projections={"d": [0], "p": [1]},
)
surface = fermi_surface(model, [200, 200], target_energy_meV=0.0)
```

The GUI displays three-dimensional Fermi surfaces with PyVista's
GPU-accelerated mesh renderer, which remains responsive for meshes that are
slow to rotate in Matplotlib. One- and two-dimensional results continue to use
Matplotlib. `render_fermi_surface` also remains available when a static
Matplotlib figure is preferable in a script or notebook.

The calculation arguments beyond the `model` itself are:

| Argument | Meaning | Acceptable input example |
| --- | --- | --- |
| `band_path.nodes` | two or more reduced-coordinate path nodes | `[[0, 0, 0], [0.5, 0, 0]]` |
| `band_path.labels` | optional label for each node | `["G", "X"]` |
| `band_path.break_before` | optional flags that suppress interpolation from the preceding node | `[False, False, True, False]` |
| `band_path.points_per_segment` | positive interpolation-interval count | `80` |
| `k_mesh.shape` | positive size for each periodic axis, or three lattice-axis sizes | `[80, 80]` |
| `k_mesh.shift` | optional offset in mesh steps for each periodic axis | `[0.5, 0.5]` for a half-step shift |
| `k_mesh.symmetry` | choose `"full"`, certified `"auto"` reduction with fallback, or required `"reduced"` sampling | `"auto"` |
| `calculate_bands.sampling` | `WavevectorSampling` path or mesh | `path` or `mesh` from the functions above |
| `calculate_bands.chemical_potential_meV` | energy stored as the plotting reference | `12.5` |
| `calculate_bands.projections` | optional named zero-based basis-index groups | `{"d": [0, 1]}` |
| `calculate_bands.include_eigenvectors` | retain the complete eigenvectors in the result | `True` |
| `calculate_bands.backend` | execution backend; `None` uses the configured process default | `"threaded"` |
| `calculate_bands.workers` | bounded CPU worker count; `None` uses nfit's allocation | `8` |
| `calculate_bands.max_batch_bytes` | temporary Hamiltonian/eigensystem memory target | `268435456` |
| `density_of_states.mesh` | mesh-valued `WavevectorSampling` | `k_mesh(model, [80, 80])` |
| `density_of_states.energy_meV` | one-dimensional absolute energy grid | `np.linspace(-250, 250, 1000)` |
| `density_of_states.broadening_meV` | positive Gaussian standard deviation | `2.0` |
| `density_of_states.chemical_potential_meV` | plotting reference energy | `12.5` |
| `density_of_states.projections` | optional named basis-index groups | `{"d": [0, 1]}` |
| `density_of_states.max_chunk_bytes` | positive temporary-kernel memory budget | `67108864` for 64 MiB |
| `density_of_states.backend`, `workers`, `max_batch_bytes` | eigensystem execution settings, as for `calculate_bands` | `"numpy"`, `1`, `268435456` |
| `fermi_surface.mesh_shape` | extraction-grid size for each periodic axis | `[200, 200]` |
| `fermi_surface.target_energy_meV` | absolute constant-energy target | `12.5` |
| `fermi_surface.projections` | optional named basis-index groups evaluated on the surface | `{"d": [0, 1]}` |
| `fermi_surface.backend`, `workers`, `max_batch_bytes` | eigensystem execution settings, as for `calculate_bands` | `"cupy"`, `1`, `268435456` |

For three-dimensional crystals, `standard_band_path(crystal,
convention="hinuma")` uses Seek-path to apply the Hinuma *et al.* HPKOT
convention, then converts the standardized path into nfit's primitive
reciprocal basis. `set_tight_binding_standard_path(component)` installs that
path and explicit disconnected-section markers. CIF-installed tight-binding
models use this convention by default. Manual nodes remain available for
nonstandard paths and reduced-dimensional models.

`electronic_energy_to_meV` and `electronic_energy_from_meV` are the explicit
script boundary for these low-level functions. The electronic renderers
default to eV and accept `energy_unit="meV"` when a low-energy display is more
useful. When a DOS is rendered in eV, nfit converts states/meV to states/eV as
well as converting the horizontal axis.

Electronic calculations use float64/complex128 throughout. `auto` keeps the
NumPy reference path and may split sufficiently large work across bounded CPU
workers. The threaded path constructs each bounded wave of Hamiltonians
serially, then diagonalizes only completed matrices in parallel; this avoids
platform eigensolver calls overlapping Hamiltonian assembly. `threaded`
selects that split explicitly. `cupy` is an explicit GPU
opt-in and falls back to NumPy when no compatible CuPy device is available.
Set the process default with `set_electronic_backend()` or
`NFIT_ELECTRONIC_BACKEND`; `electronic_workers=0` follows `NFIT_NUM_THREADS`.
Result provenance records the requested and resolved backend, worker count,
batch size, precision, and that no approximation was used.

Batching changes neither the wavevectors nor the Hamiltonian. Unprojected
calculations use `eigvalsh` and avoid constructing eigenvectors. Projected
bands, projected DOS, and projected Fermi surfaces still calculate the
eigenvectors they require.

`k_mesh(..., symmetry="auto")` reduces only a three-dimensional uniform mesh
with reciprocal operations certified by nfit's symmetry-aware orbital
builder. It retains exact orbit multiplicities as integration weights.
Wannier90 imports and incompatible shifts safely return the full mesh in
automatic mode. `symmetry="reduced"` instead raises an error when that
certification is unavailable. The component-level option is limited to total
DOS because an arbitrary orbital projection need not be invariant under the
crystal symmetry. It defaults to `"full"`, preserving the full-mesh summation
order.

The band-structure, density-of-states, and Fermi-surface viewers use a common
two-column window: the interactive plot and its Matplotlib navigation toolbar
are on the left, and a fixed-width **Settings** panel is on the right. The
panel is intentionally empty in this release; it establishes one location for
later plot-specific controls without changing the viewer layout. Plot scripts
run as files open the same window.

## Electronic matrix inspector

**Inspect matrices** opens a separate viewer rather than adding permanent
columns to the model editor. Its catalog includes:

- the resolved complex $H(\mathbf k)$;
- every named parameter selector after Fourier summation and its current
  coefficient contribution;
- $S_x,S_y,S_z$ for explicit spin models; and
- representative onsite and hopping matrix bases from the builder.

The **Heatmap** tab displays the real part, imaginary part, magnitude, or
phase. **Matrix elements** gives the exact complex entries with ordered row
and column labels. A parameter can be viewed either as its dimensionless
matrix basis or as its coefficient times that basis. For a hopping
$T_{ij}(\mathbf R)$, rows are destination orbitals and columns are source
orbitals.

The right panel also reports every nonzero orbital-subspace block

$$
P_\alpha A P_\beta
$$

grouped by site, manifold, and spin. It gives the block shape, Frobenius norm,
and largest element. This makes onsite mixing, inter-manifold hopping, and
spinor structure visible without treating a multiorbital term as one scalar.
`electronic_matrix_catalog` is the renderer-independent API,
`show_electronic_matrix_catalog` is the Qt viewer, and
`electronic_matrix_script` exports an editable standalone inspection.

## Fitting and identifiability

Every active onsite, hopping, or SOC invariant is a first-class parameter whose
stable identifier is shared by:

- `component.parameters`, in canonical meV;
- `component.limits`, `component.fit_parameters`, and `component.sharing`;
- the mirrored high-level onsite, hopping, or SOC record;
- the resolved `ElectronicModel.parameter_values`; and
- project files, builder scripts, and fit reports.

The orbital-aware GUI tables display electronic energies in the selected
`electronic_energy_unit` and expose global, per-dataset, and grouped sharing.
`set_tight_binding_parameter_state` is the equivalent canonical-meV scripting
API. Regenerating symmetry terms preserves common parameter state for stable
identifiers and removes state belonging to deleted terms.

```python
from nfit import (
    electronic_model_from_component,
    set_tight_binding_parameter_state,
)

set_tight_binding_parameter_state(
    model,
    values_meV={term_id: 30.0},
    fit_parameters={term_id: True},
    limits_meV={term_id: [-100.0, 100.0]},
    sharing={
        term_id: {
            "mode": "grouped",
            "groups": {"scan1": "low_temperature", "scan2": "low_temperature"},
        }
    },
)
trial_model = electronic_model_from_component(
    model,
    parameter_values_meV={term_id: 31.5},
)
```

`electronic_model_from_component` replaces only the immutable model's named
coefficient values. Its symmetry-generated Hamiltonian blocks can therefore
be reused by a response calculation. The returned content digest includes the
resolved values, so caches can distinguish optimizer states.

The tight-binding component itself remains calculation-only: bands and
related electronic plots are not measured-dataset residuals. Its selected
parameters enter an optimizer when a compatible electronic-response model
supplies the susceptibility or other dataset observable. Fit reports then
include the readable term labels, canonical values and uncertainties, bounds,
sharing modes, basis count, and electronic-model digest.

In this documentation, **onsite energy** means a static diagonal or
symmetry-allowed onsite Hamiltonian term. A frequency-dependent many-body
self-energy $\Sigma_{ab}(\mathbf k,E)$ is a separate extension and should not
be conflated with an onsite energy.

## Scripting and export

The **Tight-binding electronic structure** model editor supports three
structure inputs:

- import a CIF and retain its path, size, and SHA-256 digest;
- edit the lattice, space group, element labels, and fractional site
  positions manually; or
- import a complete Wannier90 Hamiltonian.

CIF import initializes `periodic_axes` to `[0, 1, 2]` and generates the
Hinuma/HPKOT path. The **Orbitals**,
**Onsite terms**, and **Hoppings** sections construct and retain a high-level
editable builder specification as well as its resolved `model_data`. Complete
manual Hamiltonians can also be constructed through the lower-level scripting
API.

Each plot action has a **Copy script** button that exports editable GUI-free
Python using the same calculation and rendering functions. The exported plot
script states `energy_unit`, converts its editable values to canonical meV,
and passes the same unit to the renderer. **Copy builder script** exports CIF
reload and digest verification, or embeds a manually entered crystal, then
reconstructs the component, manifolds, regenerated onsite matrices, values,
bounds, generated hopping matrices, fit selections, sharing rules, and
canonical digest without Qt.

## Shared 3D model viewer

**View model in 3D** opens the same model-geometry viewer for tight binding and
Heisenberg RPA. It can show the complete unit cell, only active sites, or
inactive atoms as translucent ghosts. Tight-binding scenes show all orbitals
simultaneously as separated colored tokens and local frames as red, green, and
blue axis triads. Atoms are smooth-shaded sphere glyphs with distinct,
CPK-like element colors and element-scaled display radii. Geometry is batched
by visual role so building a scene does not create one rendering actor per
atom, orbital, frame arrow, or pathway. Tokens identify basis states; they are
not wavefunction isosurfaces. Heisenberg scenes can display one representative
exchange pathway or all symmetry-equivalent members. Tight-binding scenes
offer the same spatial-orbit selection and can restrict a pathway to one
symmetry-allowed hopping matrix term. A pathway depicts geometry, not a scalar
summary of a multiorbital hopping matrix.
Left-clicking an atom, orbital token, or hopping/exchange pathway updates the
**Selected object** panel with its expanded site name, element, orbital and
manifold labels, or bond-orbit label. This picking behavior is shared by the
interactive model editor and exported standalone viewer.

`model_geometry_scene` is the renderer-independent public API. It returns cell
edges, sites, orbital tokens, frames, and pathways as immutable records.
`model_geometry_script` exports an editable standalone viewer script.

## Three-dimensional Brillouin-zone viewer

**View Brillouin zone in 3D** constructs the first Brillouin zone as the
Wigner--Seitz cell of the **primitive** reciprocal translation lattice. For a
centered conventional crystal cell, including F-centered space groups, nfit
uses the space-group centering translations to construct primitive direct
vectors before finding the zone. Configured path nodes use that primitive
reciprocal basis. A coordinate such as `[0.5, 0, 0]` means
$\mathbf b_1/2$; it is not a universal definition of a point named X and need
not lie on a zone face for a non-orthogonal primitive basis. Use the generated
Hinuma/HPKOT path for conventional high-symmetry labels, or enter manual
coordinates deliberately. Before evaluating $H(\mathbf k)$, nfit converts
these physical wavevectors to the model's internal reduced coordinates. The
three-dimensional view and band-structure calculation therefore use the same
physical path.

The viewer overlays the configured `band_path`, including every node label,
and the primitive reciprocal vectors $\mathbf b_1$, $\mathbf b_2$, and
$\mathbf b_3$. Coordinates and vectors are in Å$^{-1}$. Each reciprocal
vector reaches the neighboring reciprocal-lattice point and therefore extends
past the intervening Brillouin-zone face. Path labels do not add point glyphs
at reciprocal-vector endpoints.

The right **Settings** panel controls:

- visibility of the reciprocal vectors, path, path labels, and XYZ compass;
- one-color or RGB reciprocal vectors and their thickness;
- path color and width, and label font size;
- Wigner--Seitz face color and opacity, outline color and width;
- orthographic or perspective camera projection; and
- copying or saving the current viewport image.

The defaults use neutral-blue reciprocal vectors, a dark-red path, transparent
flat faces, a dark outline, and orthographic projection. Each setting redraws
the existing scene without rebuilding the zone geometry.

`brillouin_zone_scene` is the renderer-independent component API.
`build_brillouin_zone_scene` accepts the model direct-lattice matrix, path
dictionaries, and an optional `primitive_lattice` matrix that defines both the
zone and path-coordinate basis. `BrillouinZoneViewOptions` contains the
renderer settings described above. `show_brillouin_zone_scene(scene,
options=...)` applies them without requiring project widgets, and
`brillouin_zone_script` exports both lattices and an editable options object.
For lower-level band calculations, `band_path(...,
coordinate_reciprocal_lattice=...)` accepts the same physical reciprocal basis
and converts nodes to the Hamiltonian basis.

```python
from nfit import BrillouinZoneViewOptions, brillouin_zone_scene
from nfit.qt_brillouin_zone_viewer import show_brillouin_zone_scene

scene = brillouin_zone_scene(model_component)
options = BrillouinZoneViewOptions(
    basis_vector_color_mode="rgb",
    path_color="#7A1F1F",
    cell_surface_opacity=0.15,
    projection="orthographic",
)
window = show_brillouin_zone_scene(scene, options=options)
```

`ElectronicModel.to_dict()` is a portable, digest-protected representation.
`save_electronic_model` and `load_electronic_model` write and validate that
representation as JSON. Imported-model scripts reload the original source and
verify the stored canonical digest.

## References

- A. A. Mostofi *et al.*, *Comput. Phys. Commun.* **178**, 685 (2008),
  [doi:10.1016/j.cpc.2007.11.016](https://doi.org/10.1016/j.cpc.2007.11.016).
- G. Pizzi *et al.*, *J. Phys.: Condens. Matter* **32**, 165902 (2020),
  [doi:10.1088/1361-648X/ab51ff](https://doi.org/10.1088/1361-648X/ab51ff).
- V. Vitale *et al.*, *npj Comput. Mater.* **6**, 66 (2020),
  [doi:10.1038/s41524-020-0312-y](https://doi.org/10.1038/s41524-020-0312-y).
- J. C. Slater and G. F. Koster, *Phys. Rev.* **94**, 1498 (1954),
  [doi:10.1103/PhysRev.94.1498](https://doi.org/10.1103/PhysRev.94.1498).
- Y. Hinuma *et al.*, *Comput. Mater. Sci.* **128**, 140 (2017),
  [doi:10.1016/j.commatsci.2016.10.015](https://doi.org/10.1016/j.commatsci.2016.10.015).
- [Seek-path documentation](https://seekpath.readthedocs.io/).
- [Wannier90 file-format documentation](https://wannier90.readthedocs.io/en/latest/user_guide/wannier90/files/).
- [ASE unit conventions](https://docs.ase-lib.org/ase/units.html).
- [pymatgen electronic-structure API](https://pymatgen.org/pymatgen.electronic_structure.html).
- [sisl internal unit conventions](https://sisl.readthedocs.io/en/latest/quickstart/overview.html).
- [PythTB hopping API](https://pythtb.readthedocs.io/en/latest/generated/pythtb/TBModel/pythtb.TBModel.set_hop.html).

```{toctree}
:maxdepth: 1
:hidden:

tight_binding_builder_plan
```
