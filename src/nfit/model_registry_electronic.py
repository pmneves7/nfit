"""Built-in electronic-structure model definitions.

The declarative Lindhard, electronic-RPA, and tight-binding schemas live here
so the core registry remains focused on its extension contract and lookup API.
Registration stays explicit and idempotence behavior remains owned by
:mod:`nfit.model_registry`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# These collaborators are installed from the core registry immediately before
# registration.  Declaring the names keeps this schema module independent of
# the registry module and therefore avoids a reverse import cycle.
ModelDefinition: Any = None
ModelPlotDefinition: Any = None
_config_field: Any = None
_fit_context_diagnostics: Any = None
_fit_context_factory: Any = None
_fit_dynamic_parameters: Any = None
_fit_factory: Any = None
_fit_validator: Any = None
_model_plot_calculator: Any = None
_model_plot_renderer: Any = None
_model_plot_script: Any = None
_parameter: Any = None
_report_sections: Any = None
_tight_binding_plot_script: Any = None
register_model_definition: Any = None

_CONTEXT_NAMES = (
    "ModelDefinition",
    "ModelPlotDefinition",
    "_config_field",
    "_fit_context_diagnostics",
    "_fit_context_factory",
    "_fit_dynamic_parameters",
    "_fit_factory",
    "_fit_validator",
    "_model_plot_calculator",
    "_model_plot_renderer",
    "_model_plot_script",
    "_parameter",
    "_report_sections",
    "_tight_binding_plot_script",
    "register_model_definition",
)


def _install_registry_context(context: Mapping[str, Any]) -> None:
    namespace = globals()
    for name in _CONTEXT_NAMES:
        namespace[name] = context[name]


def _register_lindhard_model() -> None:
    register_model_definition(
        ModelDefinition(
            key="lindhard",
            label="Bare Lindhard spin susceptibility",
            description=(
                "Generalized complex particle-hole response projected onto "
                "physical spin operators from a referenced tight-binding model."
            ),
            category="electronic_structure",
            data_types=(
                "single_crystal_inelastic",
                "powder_inelastic",
                "single_crystal_elastic",
                "powder_elastic",
                "magnetization",
            ),
            factory=_fit_factory("_lindhard_unbound_factory"),
            context_factory=_fit_context_factory("_lindhard_factory"),
            component_reference_fields=("electronic_component",),
            parameter_fields=(
                _parameter(
                    "broadening",
                    5.0,
                    "Positive particle-hole lifetime broadening eta.",
                    "Positive finite energy in meV.",
                    "meV",
                    "2.0",
                ),
            ),
            config_fields=(
                _config_field(
                    "electronic_component",
                    "",
                    (
                        "Name of the sibling tight-binding component supplying "
                        "the Hamiltonian and its fitted coefficients. An empty "
                        "value selects the only enabled compatible component."
                    ),
                    (
                        "Empty with exactly one enabled tight-binding component, "
                        "or the name of one enabled component."
                    ),
                    "str",
                    "Bands",
                ),
                _config_field(
                    "response_mesh",
                    [16, 16, 16],
                    "Full Brillouin-zone integration mesh for the bare response.",
                    "One positive size per periodic model direction.",
                    "list",
                    "[24, 24, 24]",
                ),
                _config_field(
                    "response_sampling_mode",
                    "automatic",
                    (
                        "Use a stored manually selected response mesh or certify "
                        "a concrete mesh before fitting."
                    ),
                    "Either automatic or manual.",
                    "str",
                    "automatic",
                ),
                _config_field(
                    "response_sampling_accuracy",
                    "standard",
                    (
                        "Observable-error profile for automatic response-mesh "
                        "certification."
                    ),
                    "One of preview, standard, high, or custom.",
                    "str",
                    "standard",
                ),
                _config_field(
                    "response_sampling_custom_rtol",
                    0.01,
                    (
                        "Relative error target used only by the custom response "
                        "sampling profile."
                    ),
                    "Positive finite number.",
                    "float",
                    "0.005",
                ),
                _config_field(
                    "response_sampling_max_refinements",
                    7,
                    "Maximum candidate meshes in one response certificate.",
                    "Integer of at least 3.",
                    "int",
                    "8",
                ),
                _config_field(
                    "response_sampling_max_mesh_points",
                    500000,
                    (
                        "Independent safety budget for full response-mesh "
                        "points; exceeding it reports an uncertified result."
                    ),
                    "Positive integer.",
                    "int",
                    "1000000",
                ),
                _config_field(
                    "response_sampling_points_per_dataset",
                    32,
                    (
                        "Maximum deterministic representative fit points from "
                        "each applicable dataset used by complete-pipeline "
                        "response-mesh certification."
                    ),
                    "Positive integer.",
                    "int",
                    "48",
                ),
                _config_field(
                    "response_sampling_certificate",
                    {},
                    (
                        "Derived serialized response-mesh certificate. It records "
                        "tested meshes and errors and is not a physical input."
                    ),
                    "JSON dictionary.",
                    "dict",
                    "{}",
                ),
                _config_field(
                    "response_mesh_shift",
                    [0.5, 0.5, 0.5],
                    (
                        "Mesh-step offsets for the full response mesh. The "
                        "default half shift reduces special-point and "
                        "Fermi-surface shell artifacts."
                    ),
                    "One finite offset per periodic model direction.",
                    "list",
                    "[0.0, 0.0, 0.0]",
                ),
                _config_field(
                    "response_symmetry",
                    "auto",
                    (
                        "Use a full response mesh, require certified little-group "
                        "reduction, or reduce automatically with a recorded fallback."
                    ),
                    "One of auto, full, or reduced.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "response_q_evaluation",
                    "auto",
                    (
                        "Evaluate arbitrary transferred wavevectors directly, "
                        "require exact commensurate mesh shifts, or permit "
                        "validated periodic interpolation. Auto remains exact "
                        "when both interpolation tolerances are zero."
                    ),
                    "One of auto, direct, commensurate, or interpolated.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "response_q_interpolation_rtol",
                    0.0,
                    (
                        "Maximum relative error accepted from automatic "
                        "wavevector interpolation; zero disables the relative "
                        "criterion."
                    ),
                    "Finite nonnegative number.",
                    "float",
                    "0.01",
                ),
                _config_field(
                    "response_q_interpolation_atol",
                    0.0,
                    (
                        "Maximum absolute complex-susceptibility error accepted "
                        "from automatic wavevector interpolation; zero disables "
                        "the absolute criterion."
                    ),
                    "Finite nonnegative susceptibility magnitude.",
                    "float",
                    "1e-5",
                    "meV^-1 cell^-1",
                ),
                _config_field(
                    "response_q_interpolation_mesh",
                    [],
                    (
                        "Optional initial commensurate interpolation mesh. An "
                        "empty list selects a small divisor of the integration "
                        "mesh and refines deterministically."
                    ),
                    (
                        "Empty list or one positive divisor of the response "
                        "mesh size per periodic direction."
                    ),
                    "list",
                    "[8, 8, 8]",
                ),
                _config_field(
                    "response_q_validation_points",
                    8,
                    (
                        "Number of deterministic off-mesh points evaluated "
                        "directly to certify wavevector interpolation."
                    ),
                    "Positive integer.",
                    "int",
                    "16",
                ),
                _config_field(
                    "response_q_interpolation_certificates",
                    {},
                    (
                        "Derived per-dataset interpolation certificates. Each "
                        "record identifies the electronic parameter state, "
                        "tested meshes, direct validation points, tolerances, "
                        "and measured errors."
                    ),
                    "JSON dictionary keyed by dataset name.",
                    "dict",
                    "{}",
                ),
                _config_field(
                    "chemical_potential_mode",
                    "source",
                    (
                        "Use the tight-binding component chemical potential or "
                        "solve it from a fixed electron filling."
                    ),
                    "Either source or filling.",
                    "str",
                    "source",
                ),
                _config_field(
                    "filling_per_cell",
                    1.0,
                    (
                        "Electron count per primitive cell when chemical "
                        "potential mode is filling."
                    ),
                    "Positive finite number below the basis capacity.",
                    "float",
                    "3.0",
                    "electrons/cell",
                ),
                _config_field(
                    "response_backend",
                    "auto",
                    (
                        "Execution backend for the response. Auto selects serial "
                        "or threaded CPU execution by workload. CuPy keeps "
                        "eigensystems and the Lindhard contraction on the GPU "
                        "and remains an explicit, validated choice."
                    ),
                    "One of auto, numpy, threaded, or cupy.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "response_workers",
                    0,
                    (
                        "Total CPU allocation shared between independent "
                        "wavevectors, electronic eigensystems, and fused "
                        "Lindhard contractions. Zero detects the current process, "
                        "affinity, cgroup, or scheduler allocation."
                    ),
                    "Nonnegative integer; zero selects automatically.",
                    "int",
                    "0",
                ),
                _config_field(
                    "response_transition_backend",
                    "auto",
                    (
                        "CPU particle-hole contraction implementation. Auto "
                        "factorizes certified ordered orbital pairs and uses "
                        "the exact fused Numba kernel for other sufficiently "
                        "large operator problems."
                    ),
                    "One of auto, numpy, or numba.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "response_validate_backend",
                    True,
                    (
                        "Compare a deterministic probe with serial NumPy before "
                        "using a threaded or GPU backend for a new model digest."
                    ),
                    "Boolean.",
                    "bool",
                    "true",
                ),
                _config_field(
                    "response_backend_probe_points",
                    8,
                    "Number of mesh points used for backend equivalence certification.",
                    "Positive integer.",
                    "int",
                    "12",
                ),
                _config_field(
                    "response_backend_rtol",
                    1.0e-10,
                    "Relative eigenvalue tolerance for backend certification.",
                    "Finite nonnegative number.",
                    "float",
                    "1e-9",
                ),
                _config_field(
                    "response_backend_atol_meV",
                    1.0e-8,
                    "Absolute eigenvalue tolerance for backend certification.",
                    "Finite nonnegative energy.",
                    "float",
                    "1e-7",
                    "meV",
                ),
                _config_field(
                    "response_max_batch_mb",
                    256.0,
                    "Temporary-memory target for electronic eigensystem batches.",
                    "Positive finite memory size.",
                    "float",
                    "512",
                    "MiB",
                ),
                _config_field(
                    "response_transition_max_batch_mb",
                    256.0,
                    (
                        "Temporary-memory target for particle-hole transition "
                        "matrix-element and denominator batches."
                    ),
                    "Positive finite memory size.",
                    "float",
                    "512",
                    "MiB",
                ),
                _config_field(
                    "response_cache_mb",
                    512.0,
                    (
                        "Maximum numerical-array memory retained for reusable "
                        "eigensystems, completed responses, and CuPy "
                        "Hamiltonian components on each host or GPU cache; "
                        "zero disables this response cache."
                    ),
                    "Finite nonnegative memory size.",
                    "float",
                    "1024",
                    "MiB",
                ),
                _config_field(
                    "response_cache_entries",
                    64,
                    (
                        "Maximum number of reusable response intermediates "
                        "retained by this compiled response component."
                    ),
                    "Nonnegative integer.",
                    "int",
                    "128",
                ),
                _config_field(
                    "powder_orientations",
                    50,
                    (
                        "Deterministic approximately equal-area sphere directions "
                        "used automatically when a powder dataset is evaluated."
                    ),
                    "Integer of at least 6.",
                    "int",
                    "96",
                ),
                _config_field(
                    "formula_units_mode",
                    "auto",
                    (
                        "Infer formula units in the actual electronic model cell "
                        "from the complete linked crystal, or use an explicit "
                        "override."
                    ),
                    "Either auto or manual.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "formula_units_per_cell",
                    1.0,
                    (
                        "Formula units represented by the electronic model cell "
                        "when formula_units_mode is manual. This converts both "
                        "spectral and bulk model-cell responses to a per-formula-"
                        "unit dataset basis."
                    ),
                    "Positive finite number.",
                    "float",
                    "2.0",
                    "f.u./cell",
                ),
                _config_field(
                    "magnetic_normalization_mode",
                    "auto",
                    (
                        "Count represented magnetic centers from unique tight-binding "
                        "sites, or use an explicit count per electronic model cell."
                    ),
                    "Either auto or manual.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "magnetic_normalization_species",
                    "",
                    (
                        "Represented tight-binding species used when a dataset is "
                        "normalized per magnetic ion. Empty selects it automatically "
                        "only when exactly one species is represented."
                    ),
                    "Empty or one represented basis species or element.",
                    "str",
                    "V4+",
                ),
                _config_field(
                    "magnetic_centers_per_model_cell",
                    1.0,
                    (
                        "Magnetic reference centers per electronic model cell when "
                        "magnetic_normalization_mode is manual."
                    ),
                    "Positive finite number.",
                    "float",
                    "4.0",
                    "centers/cell",
                ),
                _config_field(
                    "bulk_g_factor",
                    2.0,
                    (
                        "Lande g factor converting uniform spin response to "
                        "bulk susceptibility or moment."
                    ),
                    "Positive finite number.",
                    "float",
                    "2.0",
                ),
                _config_field(
                    "plot_q_reduced",
                    [0.5, 0.5, 0.5],
                    (
                        "Representative transferred wavevector Q for the "
                        "model-owned energy scan and default convergence "
                        "certificate, in the tight-binding reciprocal basis."
                    ),
                    "Three finite reduced coordinates.",
                    "list",
                    "[0.5, 0.5, 0.0]",
                    "r.l.u.",
                ),
                _config_field(
                    "plot_energy_min_meV",
                    -100.0,
                    "Lower energy transfer for the model-owned response plot.",
                    "Finite energy below plot_energy_max_meV.",
                    "float",
                    "-50.0",
                    "meV",
                ),
                _config_field(
                    "plot_energy_max_meV",
                    100.0,
                    "Upper energy transfer for the model-owned response plot.",
                    "Finite energy above plot_energy_min_meV.",
                    "float",
                    "50.0",
                    "meV",
                ),
                _config_field(
                    "plot_energy_points",
                    401,
                    "Number of samples in the model-owned response plot.",
                    "Integer of at least 2.",
                    "int",
                    "501",
                ),
                _config_field(
                    "plot_temperature_K",
                    10.0,
                    "Temperature used for the model-owned response plot.",
                    "Finite nonnegative temperature.",
                    "float",
                    "20.0",
                    "K",
                ),
                _config_field(
                    "convergence_mesh_scales",
                    [0.5, 0.75, 1.0],
                    (
                        "Factors applied to each response-mesh dimension for "
                        "the convergence plot; the last entry is the reference."
                    ),
                    "List of positive finite factors.",
                    "list",
                    "[0.5, 1.0, 1.5]",
                ),
                _config_field(
                    "convergence_broadening_scales",
                    [2.0, 1.0, 0.5],
                    (
                        "Factors applied to the current broadening for the "
                        "convergence plot; the last entry is the reference."
                    ),
                    "List of positive finite factors.",
                    "list",
                    "[2.0, 1.0, 0.5]",
                ),
                _config_field(
                    "convergence_energy_points",
                    9,
                    (
                        "Energy samples spanning the configured plot window "
                        "for mesh and broadening convergence."
                    ),
                    "Integer of at least 1.",
                    "int",
                    "17",
                ),
                _config_field(
                    "convergence_relative_floor",
                    1.0e-12,
                    "Denominator floor for reported relative convergence changes.",
                    "Positive finite susceptibility magnitude.",
                    "float",
                    "1e-10",
                    "meV^-1 cell^-1",
                ),
            ),
            validate_component=_fit_validator("_validate_lindhard_component"),
            report_sections=_report_sections("lindhard_report_sections"),
            plots=(
                ModelPlotDefinition(
                    key="complex_energy_scan",
                    label="Complex spin susceptibility versus energy",
                    description=(
                        "Plot the isotropic bare spin response at the configured "
                        "transferred wavevector."
                    ),
                    calculate=_model_plot_calculator(
                        "lindhard_energy_scan_unbound"
                    ),
                    context_calculate=_model_plot_calculator(
                        "lindhard_energy_scan"
                    ),
                    render=_model_plot_renderer(
                        "render_lindhard_energy_scan"
                    ),
                    script=_model_plot_script(
                        "lindhard_energy_scan_script_unbound"
                    ),
                    context_script=_model_plot_script(
                        "lindhard_energy_scan_script"
                    ),
                ),
                ModelPlotDefinition(
                    key="convergence",
                    label="Mesh and broadening convergence",
                    description=(
                        "Compare response meshes and broadenings independently "
                        "over representative Q-energy points."
                    ),
                    calculate=_model_plot_calculator(
                        "lindhard_convergence_scan_unbound"
                    ),
                    context_calculate=_model_plot_calculator(
                        "lindhard_convergence_scan"
                    ),
                    render=_model_plot_renderer(
                        "render_lindhard_convergence_scan"
                    ),
                    script=_model_plot_script(
                        "lindhard_convergence_scan_script_unbound"
                    ),
                    context_script=_model_plot_script(
                        "lindhard_convergence_scan_script"
                    ),
                ),
            ),
            default_lower_bounds=(("broadening", 0.0),),
            documentation="lindhard.md",
            citations=(
                "Lindhard, Kgl. Danske Videnskab. Selskab, Mat.-Fys. Medd. 28, no. 8 (1954)",
                "https://doi.org/10.1088/1367-2630/11/2/025016",
            ),
            metadata={
                "component_dependencies": True,
                "reference_types": {"electronic_component": ["tight_binding"]},
            },
        )
    )


def _register_electronic_rpa_models() -> None:
    rpa_data_types = (
        "single_crystal_inelastic",
        "powder_inelastic",
        "single_crystal_elastic",
        "powder_elastic",
        "magnetization",
    )
    rpa_reference_fields = (
        _config_field(
            "response_component",
            "",
            (
                "Enabled sibling Lindhard component supplying the bare response, "
                "mesh, broadening, electronic model, and dataset conversion."
            ),
            "Name of one enabled Lindhard component in this workspace.",
            "str",
            "Bare response",
        ),
        _config_field(
            "singular_tolerance",
            1.0e-12,
            "Relative singular-value threshold for an RPA pole.",
            "Positive finite number.",
            "float",
            "1e-10",
        ),
        _config_field(
            "near_pole_tolerance",
            1.0e-3,
            (
                "Relative minimum singular value below which an evaluated RPA "
                "denominator is reported as near a pole."
            ),
            "Finite number greater than singular_tolerance.",
            "float",
            "0.01",
        ),
        _config_field(
            "static_stability_warning_margin",
            0.05,
            (
                "Warn when the sampled static feedback leaves less than this "
                "margin before its RPA instability."
            ),
            "Finite nonnegative number.",
            "float",
            "0.1",
        ),
        _config_field(
            "reject_sampled_static_instability",
            False,
            (
                "Give unstable zero-energy trial parameters a finite fitting "
                "penalty when an evaluated wavevector crosses the sampled RPA "
                "stability boundary."
            ),
            "Boolean.",
            "bool",
            "true",
        ),
    )
    rpa_plot_definition = ModelPlotDefinition(
        key="complex_energy_scan",
        label="Complex interaction-dressed susceptibility versus energy",
        description=(
            "Plot the real and imaginary interaction-dressed spin response "
            "using the referenced Lindhard plot settings."
        ),
        calculate=_model_plot_calculator("electronic_rpa_energy_scan_unbound"),
        context_calculate=_model_plot_calculator("electronic_rpa_energy_scan"),
        render=_model_plot_renderer("render_lindhard_energy_scan"),
        script=_model_plot_script("electronic_rpa_energy_scan_script_unbound"),
        context_script=_model_plot_script("electronic_rpa_energy_scan_script"),
    )
    register_model_definition(
        ModelDefinition(
            key="stoner_rpa",
            label="Scalar Stoner RPA",
            description=(
                "Isotropic scalar interaction dressing of a referenced bare "
                "Lindhard spin susceptibility."
            ),
            category="electronic_structure",
            data_types=rpa_data_types,
            factory=_fit_factory("_rpa_unbound_factory"),
            context_factory=_fit_context_factory("_stoner_rpa_factory"),
            component_reference_fields=("response_component",),
            consumes_referenced_observables=True,
            parameter_fields=(
                _parameter(
                    "I",
                    0.1,
                    "Scalar Stoner interaction applied to each Cartesian spin channel.",
                    "Finite electronic interaction energy.",
                    "eV",
                    "0.35",
                ),
            ),
            config_fields=rpa_reference_fields,
            validate_component=_fit_validator("_validate_stoner_rpa_component"),
            context_diagnostics=_fit_context_diagnostics(
                "_electronic_rpa_component_diagnostics"
            ),
            report_sections=_report_sections("electronic_rpa_report_sections"),
            plots=(rpa_plot_definition,),
            default_lower_bounds=(("I", 0.0),),
            documentation="stoner_rpa.md",
            citations=(
                "https://doi.org/10.1007/978-3-642-82499-9",
            ),
            metadata={
                "component_dependencies": True,
                "reference_types": {"response_component": ["lindhard"]},
            },
        )
    )
    register_model_definition(
        ModelDefinition(
            key="matrix_rpa",
            label="Matrix RPA interaction",
            description=(
                "User-defined Hermitian spin-channel interaction matrix dressing "
                "of a referenced bare Lindhard response."
            ),
            category="electronic_structure",
            data_types=rpa_data_types,
            factory=_fit_factory("_rpa_unbound_factory"),
            context_factory=_fit_context_factory("_matrix_rpa_factory"),
            component_reference_fields=("response_component",),
            consumes_referenced_observables=True,
            parameter_fields=(
                _parameter(
                    "scale",
                    0.1,
                    "Fittable energy multiplying the configured dimensionless matrix.",
                    "Finite electronic interaction energy.",
                    "eV",
                    "0.2",
                ),
            ),
            config_fields=(
                *rpa_reference_fields,
                _config_field(
                    "vertex_matrix",
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    (
                        "Dimensionless Hermitian matrix in the ordered "
                        "(Sx, Sy, Sz) operator basis."
                    ),
                    "Finite Hermitian 3 by 3 nested list.",
                    "list",
                    "[[1,0,0],[0,0.8,0],[0,0,1.2]]",
                ),
                _config_field(
                    "channel",
                    "spin",
                    "Recorded physical channel of the user-defined vertex.",
                    "Nonempty descriptive string.",
                    "str",
                    "spin",
                ),
            ),
            validate_component=_fit_validator("_validate_matrix_rpa_component"),
            context_diagnostics=_fit_context_diagnostics(
                "_electronic_rpa_component_diagnostics"
            ),
            report_sections=_report_sections("electronic_rpa_report_sections"),
            plots=(rpa_plot_definition,),
            documentation="matrix_rpa.md",
            citations=(
                "https://doi.org/10.1088/1367-2630/11/2/025016",
            ),
            metadata={
                "component_dependencies": True,
                "reference_types": {"response_component": ["lindhard"]},
            },
        )
    )
    register_model_definition(
        ModelDefinition(
            key="hubbard_hund_rpa",
            label="Hubbard-Hund RPA",
            description=(
                "Local multiorbital spin-channel RPA dressing on selected "
                "site-attached correlated shells."
            ),
            category="electronic_structure",
            data_types=rpa_data_types,
            factory=_fit_factory("_rpa_unbound_factory"),
            context_factory=_fit_context_factory("_hubbard_hund_rpa_factory"),
            component_reference_fields=("response_component",),
            consumes_referenced_observables=True,
            parameter_fields=(
                _parameter(
                    "U",
                    1.0,
                    "Intraorbital Hubbard interaction.",
                    "Finite electronic interaction energy.",
                    "eV",
                    "2.0",
                ),
                _parameter(
                    "U_prime",
                    0.6,
                    (
                        "Interorbital direct interaction; derived as U - 2 J_H "
                        "when rotational invariance is enabled."
                    ),
                    "Finite electronic interaction energy.",
                    "eV",
                    "1.4",
                ),
                _parameter(
                    "J_H",
                    0.2,
                    "Hund exchange interaction.",
                    "Finite electronic interaction energy.",
                    "eV",
                    "0.3",
                ),
                _parameter(
                    "J_pair",
                    0.2,
                    (
                        "Pair-hopping interaction; derived as J_H when "
                        "rotational invariance is enabled."
                    ),
                    "Finite electronic interaction energy.",
                    "eV",
                    "0.3",
                ),
            ),
            config_fields=(
                *rpa_reference_fields,
                _config_field(
                    "correlated_shells",
                    [],
                    (
                        "Correlated-shell labels to dress; an empty list selects "
                        "all labelled correlated shells."
                    ),
                    "List of shell labels or a comma-separated string.",
                    "list",
                    '["M1_3d"]',
                ),
                _config_field(
                    "rotationally_invariant",
                    True,
                    "Enforce U_prime = U - 2 J_H and J_pair = J_H.",
                    "Boolean.",
                    "bool",
                    "true",
                ),
            ),
            validate_component=_fit_validator("_validate_hubbard_hund_rpa_component"),
            context_diagnostics=_fit_context_diagnostics(
                "_electronic_rpa_component_diagnostics"
            ),
            report_sections=_report_sections("electronic_rpa_report_sections"),
            plots=(rpa_plot_definition,),
            default_lower_bounds=(("U", 0.0), ("J_H", 0.0)),
            documentation="hubbard_hund_rpa.md",
            citations=(
                "https://doi.org/10.1088/1367-2630/11/2/025016",
                "https://doi.org/10.1146/annurev-conmatphys-020911-125045",
            ),
            metadata={
                "component_dependencies": True,
                "reference_types": {"response_component": ["lindhard"]},
            },
        )
    )


def _register_tight_binding_model() -> None:
    register_model_definition(
        ModelDefinition(
            key="tight_binding",
            label="Tight-binding electronic structure",
            description=(
                "Material-independent orthonormal electronic Hamiltonian for "
                "manual or Wannier90 models, with bands, orbital projections, "
                "density of states, and Fermi surfaces."
            ),
            category="electronic_structure",
            data_types=("electronic_structure",),
            factory=_fit_factory("_electronic_structure_factory"),
            validate_component=_fit_validator("_validate_tight_binding_config"),
            dynamic_parameters=_fit_dynamic_parameters(
                "tight_binding_parameter_names"
            ),
            dynamic_parameter_description=(
                "Named static onsite or hopping coefficient {name}. Values are "
                "canonical meV and multiply the symmetry-generated Hamiltonian basis."
            ),
            dynamic_parameter_unit="meV",
            config_fields=(
                _config_field(
                    "source_path",
                    "",
                    "Wannier90 *_hr.dat or *_tb.dat source; empty for a manual model.",
                    "Existing Wannier90 Hamiltonian path, or empty.",
                    "str",
                    "/path/to/material_tb.dat",
                ),
                _config_field(
                    "model_digest",
                    "",
                    "SHA-256 digest of the canonical electronic model.",
                    "A 64-character hexadecimal digest, or empty.",
                    "str",
                    "",
                ),
                _config_field(
                    "model_data",
                    {},
                    "Portable ElectronicModel.to_dict() data for a manual model.",
                    "Canonical model dictionary, or empty.",
                    "dict",
                    "{}",
                ),
                _config_field(
                    "model_stale",
                    False,
                    (
                        "Whether compact builder edits have invalidated the "
                        "cached canonical Hamiltonian."
                    ),
                    "Boolean derived-cache state; calculations resolve it lazily.",
                    "bool",
                    "false",
                ),
                _config_field(
                    "use_primitive_cell",
                    True,
                    (
                        "Fold GUI-built conventional-cell Hamiltonians onto "
                        "the primitive translation lattice before evaluation."
                    ),
                    "Boolean; disable only for diagnostic comparison.",
                    "bool",
                    "true",
                ),
                _config_field(
                    "hopping_parameterization",
                    "slater_koster",
                    (
                        "Basis used for GUI-generated hopping coefficients: "
                        "compact two-centre Slater-Koster channels or the full "
                        "space-group-allowed matrix basis."
                    ),
                    "Either slater_koster or general.",
                    "str",
                    "slater_koster",
                ),
                _config_field(
                    "crystal",
                    {
                        "lattice": {
                            "a": 5.0,
                            "b": 5.0,
                            "c": 5.0,
                            "alpha": 90.0,
                            "beta": 90.0,
                            "gamma": 90.0,
                        },
                        "spacegroup": "P 1",
                        "sites": [],
                    },
                    "Editable crystal geometry for CIF-based or manual construction.",
                    "Crystal dictionary with lattice, space group, and site records.",
                    "dict",
                    '{"lattice": {"a": 4, "b": 4, "c": 6, '
                    '"alpha": 90, "beta": 90, "gamma": 90}, '
                    '"spacegroup": "P 1", "sites": []}',
                ),
                _config_field(
                    "orbital_manifolds",
                    [],
                    (
                        "Editable site-attached orbital manifolds, local frames, "
                        "and symmetry conventions used by the structure-first builder."
                    ),
                    "A list of OrbitalManifold dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "spin_treatment",
                    "auto",
                    (
                        "Requested spin representation. Auto keeps spin "
                        "implicit unless active SOC requires a full spinor basis."
                    ),
                    "One of auto, implicit, collinear, or spinor.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "soc_terms",
                    [],
                    (
                        "Optional manifold-resolved onsite lambda L.S terms "
                        "with canonical meV coefficients."
                    ),
                    "A list of SpinOrbitTerm dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "onsite_terms",
                    [],
                    (
                        "Generated symmetry-allowed onsite energies and "
                        "hybridizations with canonical meV coefficients."
                    ),
                    "A list of OnsiteInvariant dictionaries generated from the crystal and orbitals.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "hopping_cutoff_angstrom",
                    0.0,
                    (
                        "Maximum real-space bond distance used by the "
                        "symmetry-aware hopping generator."
                    ),
                    "Zero before generation, otherwise a positive length in Angstrom.",
                    "float",
                    "4.5",
                    "angstrom",
                ),
                _config_field(
                    "spatial_orbits",
                    [],
                    (
                        "Generated symmetry-distinct hopping pathways and "
                        "their space-group mapping operations."
                    ),
                    "A generated list of BondOrbit dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "hopping_candidates",
                    [],
                    (
                        "Generated symmetry-allowed hopping suggestions; only "
                        "selected candidates enter the Hamiltonian."
                    ),
                    "A generated list of HoppingInvariant dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "hopping_terms",
                    [],
                    (
                        "Generated symmetry-covariant hopping matrices with "
                        "canonical meV coefficients."
                    ),
                    "A generated list of HoppingInvariant dictionaries.",
                    "list",
                    "[]",
                ),
                _config_field(
                    "periodic_axes",
                    [],
                    (
                        "Direct-lattice periodic axes. Empty uses all three "
                        "for a GUI-built crystal and infers active translated "
                        "axes for a Wannier90 import."
                    ),
                    "Empty, or one to three unique indices chosen from 0, 1, and 2.",
                    "list",
                    "[0, 1]",
                ),
                _config_field(
                    "electronic_energy_unit",
                    "eV",
                    (
                        "Input and plot unit for electronic-structure energies; "
                        "canonical model data remain in meV."
                    ),
                    "Either eV or meV.",
                    "str",
                    "eV",
                ),
                _config_field(
                    "electronic_backend",
                    "auto",
                    (
                        "Execution backend for bands, density of states, and "
                        "Fermi surfaces. Automatic mode uses the NumPy reference "
                        "path and bounded CPU threading when worthwhile; CuPy is "
                        "an explicit GPU opt-in."
                    ),
                    "One of auto, numpy, threaded, or cupy.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "electronic_workers",
                    0,
                    (
                        "Maximum CPU workers for electronic calculations. Zero "
                        "uses the nfit CPU-allocation policy and respects "
                        "NFIT_NUM_THREADS."
                    ),
                    "Zero for automatic allocation, or a positive integer.",
                    "int",
                    "8",
                ),
                _config_field(
                    "electronic_max_batch_mb",
                    256.0,
                    (
                        "Working-memory target for each electronic Hamiltonian "
                        "and eigensystem batch."
                    ),
                    "Positive finite memory size.",
                    "float",
                    "512",
                    "MiB",
                ),
                _config_field(
                    "chemical_potential_meV",
                    0.0,
                    (
                        "Canonical chemical potential used as the plotted energy "
                        "zero; the GUI converts it to the selected electronic unit."
                    ),
                    "Any finite energy stored in meV.",
                    "float",
                    "12.5",
                    "meV",
                ),
                _config_field(
                    "projection_groups",
                    {},
                    "Named orbital projections as zero-based basis-index lists.",
                    "JSON dictionary mapping labels to index lists.",
                    "dict",
                    '{"d": [0, 1, 2], "p": [3, 4]}',
                ),
                _config_field(
                    "band_path",
                    [
                        {"label": r"$\Gamma$", "k": [0.0, 0.0, 0.0]},
                        {"label": "X", "k": [0.5, 0.0, 0.0]},
                        {"label": "M", "k": [0.5, 0.5, 0.0]},
                        {"label": r"$\Gamma$", "k": [0.0, 0.0, 0.0]},
                    ],
                    (
                        "Ordered labeled nodes of the band path in the primitive "
                        "reciprocal basis."
                    ),
                    (
                        "JSON list of labeled three-coordinate nodes. A coordinate "
                        "of 0.5 reaches halfway to the neighboring reciprocal point "
                        "along that primitive basis vector."
                    ),
                    "list",
                    '[{"label": "G", "k": [0, 0, 0]}, '
                    '{"label": "X", "k": [0.5, 0, 0]}]',
                ),
                _config_field(
                    "band_path_convention",
                    "hinuma",
                    (
                        "Source convention for the configured high-symmetry "
                        "path."
                    ),
                    (
                        "One of manual, hinuma, or "
                        "setyawan_curtarolo."
                    ),
                    "str",
                    "hinuma",
                ),
                _config_field(
                    "band_path_metadata",
                    {},
                    (
                        "Provider, version, convention, and tolerance used to "
                        "generate an automatic high-symmetry path."
                    ),
                    "JSON dictionary; empty for a manual path.",
                    "dict",
                    (
                        '{"provider": "seekpath", "provider_version": "2.2.1", '
                        '"convention": "HPKOT", "symprec": 1e-5}'
                    ),
                ),
                _config_field(
                    "band_points_per_inv_angstrom",
                    80.0,
                    (
                        "Band-path interpolation intervals per inverse Angstrom "
                        "of physical reciprocal-space distance."
                    ),
                    "Positive finite number.",
                    "float",
                    "100",
                    "angstrom",
                ),
                _config_field(
                    "dos_method",
                    "gaussian",
                    (
                        "Integrate the density of states using Gaussian "
                        "broadening or ASE's linear tetrahedron method."
                    ),
                    "Either gaussian or tetrahedron.",
                    "str",
                    "tetrahedron",
                ),
                _config_field(
                    "dos_mesh",
                    [40, 40, 40],
                    "Uniform Brillouin-zone mesh for density of states.",
                    "One size per periodic dimension, or three lattice-axis sizes.",
                    "list",
                    "[80, 80, 1]",
                ),
                _config_field(
                    "dos_sampling_mode",
                    "automatic",
                    (
                        "Use a stored manually selected DOS mesh or certify a "
                        "concrete production mesh."
                    ),
                    "Either automatic or manual.",
                    "str",
                    "automatic",
                ),
                _config_field(
                    "dos_sampling_accuracy",
                    "standard",
                    (
                        "Observable-error profile for automatic DOS-mesh "
                        "certification."
                    ),
                    "One of preview, standard, high, or custom.",
                    "str",
                    "standard",
                ),
                _config_field(
                    "dos_sampling_custom_rtol",
                    0.01,
                    (
                        "Relative error target used only by the custom DOS "
                        "sampling profile."
                    ),
                    "Positive finite number.",
                    "float",
                    "0.005",
                ),
                _config_field(
                    "dos_sampling_max_refinements",
                    7,
                    "Maximum candidate meshes in one DOS certificate.",
                    "Integer of at least 3.",
                    "int",
                    "8",
                ),
                _config_field(
                    "dos_sampling_max_mesh_points",
                    2000000,
                    (
                        "Independent safety budget for full DOS-mesh points; "
                        "exceeding it reports an uncertified result."
                    ),
                    "Positive integer.",
                    "int",
                    "4000000",
                ),
                _config_field(
                    "dos_sampling_certificate",
                    {},
                    (
                        "Derived serialized DOS-mesh certificate. It records "
                        "tested meshes and errors and is not a physical input."
                    ),
                    "JSON dictionary.",
                    "dict",
                    "{}",
                ),
                _config_field(
                    "dos_symmetry",
                    "auto",
                    (
                        "Brillouin-zone symmetry policy for total density of "
                        "states. Auto reduces only when certified, full keeps "
                        "the original mesh, and reduced fails if unsafe."
                    ),
                    "One of auto, full, or reduced.",
                    "str",
                    "auto",
                ),
                _config_field(
                    "dos_auto_energy_range",
                    False,
                    (
                        "Derive density-of-states energy limits from sampled "
                        "band extrema, including Gaussian-tail padding."
                    ),
                    "Boolean.",
                    "bool",
                    "true",
                ),
                _config_field(
                    "dos_energy_min_meV",
                    -500.0,
                    "Canonical lower density-of-states energy.",
                    "Finite energy below dos_energy_max_meV.",
                    "float",
                    "-250",
                    "meV",
                ),
                _config_field(
                    "dos_energy_max_meV",
                    500.0,
                    "Canonical upper density-of-states energy.",
                    "Finite energy above dos_energy_min_meV.",
                    "float",
                    "250",
                    "meV",
                ),
                _config_field(
                    "dos_energy_points",
                    600,
                    "Number of points in the density-of-states energy grid.",
                    "Integer of at least 2.",
                    "int",
                    "1000",
                ),
                _config_field(
                    "dos_broadening_meV",
                    5.0,
                    (
                        "Canonical Gaussian standard deviation for density "
                        "of states; unused by tetrahedron integration."
                    ),
                    "Positive finite energy when dos_method is gaussian.",
                    "float",
                    "2.0",
                    "meV",
                ),
                _config_field(
                    "fermi_mesh_mode",
                    "spacing",
                    (
                        "Choose the Fermi-surface extraction mesh from one "
                        "physical reciprocal-space spacing or enter all grid sizes."
                    ),
                    "Either spacing or size.",
                    "str",
                    "spacing",
                ),
                _config_field(
                    "fermi_spacing_inv_angstrom",
                    0.025,
                    (
                        "Target physical spacing between neighboring "
                        "Fermi-surface grid points."
                    ),
                    "Positive finite reciprocal length.",
                    "float",
                    "0.02",
                    "angstrom^-1",
                ),
                _config_field(
                    "fermi_mesh",
                    [64, 64, 64],
                    (
                        "Resolved periodic grid used to extract the Fermi "
                        "surface, or the requested grid in explicit-size mode."
                    ),
                    "One size per periodic dimension, or three lattice-axis sizes.",
                    "list",
                    "[200, 200, 1]",
                ),
                _config_field(
                    "fermi_energy_meV",
                    0.0,
                    "Canonical target energy for Fermi-surface extraction.",
                    "Any finite energy.",
                    "float",
                    "0.0",
                    "meV",
                ),
            ),
            plots=(
                ModelPlotDefinition(
                    key="bands",
                    label="Band structure",
                    description=(
                        "Plot bands and configured orbital projections in the "
                        "selected electronic energy unit (eV by default)."
                    ),
                    calculate=_model_plot_calculator("tight_binding_band_structure"),
                    render=_model_plot_renderer("render_band_structure"),
                    script=_tight_binding_plot_script("bands"),
                ),
                ModelPlotDefinition(
                    key="dos",
                    label="Density of states",
                    description=(
                        "Plot total and orbital-projected DOS with both axes "
                        "converted to the selected electronic energy unit."
                    ),
                    calculate=_model_plot_calculator(
                        "tight_binding_density_of_states"
                    ),
                    render=_model_plot_renderer("render_density_of_states"),
                    script=_tight_binding_plot_script("dos"),
                ),
                ModelPlotDefinition(
                    key="fermi_surface",
                    label="Fermi surface",
                    description=(
                        "Plot 1D Fermi points, 2D contours, or a 3D surface and "
                        "label its target in the selected electronic energy unit."
                    ),
                    calculate=_model_plot_calculator("tight_binding_fermi_surface"),
                    render=_model_plot_renderer("render_fermi_surface"),
                    script=_tight_binding_plot_script("fermi_surface"),
                ),
            ),
            report_sections=_report_sections("tight_binding_report_sections"),
            documentation="tight_binding.md",
            citations=(
                "https://doi.org/10.1016/j.cpc.2007.11.016",
                "https://doi.org/10.1088/1361-648X/ab51ff",
            ),
            metadata={"component_plot_actions": True},
        )
    )


def register_lindhard_model(context: Mapping[str, Any]) -> None:
    """Register the built-in bare Lindhard response definition."""

    _install_registry_context(context)
    _register_lindhard_model()


def register_electronic_rpa_models(context: Mapping[str, Any]) -> None:
    """Register the built-in Stoner, matrix, and Hubbard-Hund RPA definitions."""

    _install_registry_context(context)
    _register_electronic_rpa_models()


def register_tight_binding_model(context: Mapping[str, Any]) -> None:
    """Register the built-in tight-binding electronic-structure definition."""

    _install_registry_context(context)
    _register_tight_binding_model()


def register_electronic_structure_models(context: Mapping[str, Any]) -> None:
    """Register every built-in electronic-structure model family."""

    _install_registry_context(context)
    _register_lindhard_model()
    _register_electronic_rpa_models()
    _register_tight_binding_model()
