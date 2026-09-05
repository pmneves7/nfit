import re
import shutil

import pytest

from nfit.model_registry import (
    MODEL_TYPE_REGISTRY,
    ModelDefinition,
    register_model_definition,
)
from nfit.pipeline import FitTimelineEntry
from nfit.report import (
    LatexCompileError,
    compile_latex_pdf,
    latex_escape,
    render_fit_report_latex,
)


def _check_balanced_environments(tex: str) -> None:
    """Every \\begin{X} has a matching \\end{X} (and vice versa)."""
    begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
    ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
    for env in set(begins) | set(ends):
        assert begins.count(env) == ends.count(env), f"unbalanced environment {env}"
    # Escaped \$ is literal text, not a math delimiter.
    assert tex.replace(r"\$", "").count("$") % 2 == 0, "unbalanced inline math"


def _background_entry(**overrides):
    payload = dict(
        name="fit 1",
        kind="result",
        snapshot={
            "datasets": [
                {"name": "scan", "parameters": {"temperature": 5.0},
                 "enabled": True, "fit_weight": 1.0, "scale_factor": 1.0}
            ],
            "models": [
                {"name": "bg", "type": "constant_background", "enabled": True,
                 "parameters": {"constant": 1.5},
                 "fit_parameters": {"constant": True},
                 "sharing": {}, "limits": {},
                 "constraints": [], "applies_to": None, "metadata": {}}
            ],
        },
        created_at="2026-07-13 10:00",
        optimizer="least_squares",
        goodness={
            "status": "converged", "chi2": 10.0, "reduced_chi2": 1.1,
            "n_points": 20, "n_variables": 1,
            "parameters": {"bg.constant": 1.5},
            "stderr": {"bg.constant": 0.02},
            "dataset_chi2": {"scan": 10.0},
            "dataset_reduced_chi2": {"scan": 1.1},
            "dataset_n_points": {"scan": 20},
            "skipped_datasets": [],
        },
        metadata={},
    )
    payload.update(overrides)
    return FitTimelineEntry(**payload)


def test_tight_binding_report_lists_named_parameter_state():
    identifier = "M1:onsite:abc123"
    model = {
        "name": "electrons",
        "type": "tight_binding",
        "enabled": True,
        "parameters": {identifier: 25.0},
        "fit_parameters": {identifier: True},
        "sharing": {
            identifier: {
                "mode": "grouped",
                "groups": {"scan1": "A", "scan2": "A"},
            }
        },
        "limits": {identifier: [-100.0, 100.0]},
        "constraints": [],
        "applies_to": None,
        "metadata": {},
        "config": {
            "model_digest": "a" * 64,
            "model_data": {"basis": [{"label": "orbital"}]},
            "onsite_terms": [
                {
                    "identifier": identifier,
                    "label": "M1 onsite energy",
                }
            ],
            "hopping_terms": [],
        },
    }
    entry = FitTimelineEntry(
        name="electronic fit",
        kind="result",
        snapshot={"datasets": [], "models": [model]},
        goodness={
            "parameters": {f"electrons.{identifier}": 24.5},
            "stderr": {f"electrons.{identifier}": 0.5},
        },
    )

    tex = render_fit_report_latex(entry, group_name="Electronic")

    assert "Tight-binding electronic structure" in tex
    assert "M1 onsite energy" in tex
    assert "grouped" in tex
    assert "canonical SHA-256 model digest" in tex
    _check_balanced_environments(tex)


def test_lindhard_report_records_linked_response_configuration():
    entry = _background_entry()
    entry.snapshot["models"] = [
        {
            "name": "response",
            "type": "lindhard",
            "enabled": True,
            "parameters": {"broadening": 2.0},
            "config": {
                "electronic_component": "bands",
                "response_mesh": [24, 24, 12],
                "response_mesh_shift": [0.5, 0.5, 0.0],
                "chemical_potential_mode": "filling",
                "response_backend": "numpy",
                "formula_units_per_cell": 2.0,
                "powder_orientations": 96,
                "response_sampling_certificate": {
                    "status": "certified",
                    "policy": {"relative_tolerance": 0.01},
                    "domain": {
                        "kind": "fit_datasets",
                        "dataset_count": 2,
                        "sampled_points": 48,
                    },
                },
            },
            "fit_parameters": {"broadening": True},
            "sharing": {},
            "limits": {},
            "constraints": [],
            "applies_to": None,
            "metadata": {},
        }
    ]
    entry.goodness["parameters"] = {"response.broadening": 1.8}
    entry.goodness["stderr"] = {"response.broadening": 0.2}

    tex = render_fit_report_latex(entry, group_name="Electronic")

    assert r"\section{Bare Lindhard spin susceptibility (response)}" in tex
    assert r"\texttt{bands}" in tex
    assert "1.80" in tex
    assert "96" in tex
    assert "Magnetic-center normalization" in tex
    assert "complete fitted pipeline" in tex
    assert "2 dataset(s), 48 point(s)" in tex
    assert "Shared magnetic form factor" not in tex
    _check_balanced_environments(tex)


def test_hubbard_hund_report_records_constraints_and_electronic_units():
    entry = _background_entry()
    entry.snapshot["models"] = [
        {
            "name": "interactions",
            "type": "hubbard_hund_rpa",
            "enabled": True,
            "parameters": {
                "U": 2.0,
                "U_prime": 1.4,
                "J_H": 0.3,
                "J_pair": 0.3,
            },
            "config": {
                "response_component": "bare",
                "singular_tolerance": 1.0e-10,
                "correlated_shells": ["M1_3d"],
                "rotationally_invariant": True,
            },
            "fit_parameters": {"U": True, "J_H": True},
            "sharing": {},
            "limits": {},
            "constraints": [],
            "applies_to": None,
            "metadata": {},
        }
    ]
    entry.goodness["parameters"] = {
        "interactions.U": 2.1,
        "interactions.J_H": 0.28,
    }
    entry.goodness["stderr"] = {
        "interactions.U": 0.1,
        "interactions.J_H": 0.02,
    }

    tex = render_fit_report_latex(entry, group_name="Electronic")

    assert "Hubbard-Hund RPA" in tex
    assert "U'=U-2J_H" in tex
    assert "M1\\_3d" in tex
    assert "eV" in tex
    _check_balanced_environments(tex)


def _full_rpa_entry():
    """A synthetic full-featured heisenberg_rpa fit entry."""
    sym = [[0.1, 0.2, 0.0], [0.2, -0.1, 0.0], [0.0, 0.0, 0.0]]
    dm = [[0.0, 0.5, 0.0], [-0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]
    config = {
        "crystal": {
            "lattice": {"a": 10.0, "b": 10.0, "c": 10.0,
                        "alpha": 90.0, "beta": 90.0, "gamma": 90.0},
            "spacegroup": "F d -3 m:2",
            "sites": [{"label": "M1", "position": [0.0, 0.0, 0.0], "ion": "Yb3"}],
        },
        "magnetic_sites": ["M1"],
        "site_positions": [[0, 0, 0]] * 16,
        "ion": "Yb3",
        "orbits": [
            {"label": "J1", "distance_angstrom": 3.5355,
             "bonds": [{"site_i": 0, "site_j": 1, "offset": [0, 0, 0]}] * 48}
        ],
        "anisotropy": {
            "J1": {"enabled": True, "basis": [
                {"name": "S1", "kind": "symmetric", "matrix": sym},
                {"name": "D1", "kind": "dm", "matrix": dm},
            ]}
        },
        "sia": {"M1": {"enabled": True, "sites": list(range(16)),
                        "basis": [{"name": "K1", "matrix": sym}]}},
        "dipole": {"enabled": True},
        "zeeman": {"enabled": True},
        "closure": {"mode": "onsager", "moment_mode": "fitted",
                    "moment_target": 1.0, "energy_cutoff_mev": 80.0,
                    "bz_grid": 8},
    }
    model = {
        "name": "M", "type": "heisenberg_rpa", "enabled": True,
        "config": config,
        "parameters": {"scale": 1.0, "chi0": 0.02, "gamma0": 3.0, "J1": 0.05,
                       "J1_S1": 0.01, "J1_D1": 0.002, "K1_M1": 0.03,
                       "D_dip": 0.21, "g_factor": 2.1, "chi_perp_ratio": 1.0,
                       "gamma_perp_ratio": 1.0, "m2_total": 0.9},
        "fit_parameters": {"J1": True, "J1_S1": True, "m2_total": True},
        "sharing": {"chi0": {"mode": "per_dataset"}},
        "limits": {"J1": [-1.0, 1.0]}, "constraints": [], "applies_to": None,
        "metadata": {"fitted_values": {
            "chi0": {"T5": 0.021, "T50": 0.018},
            "gamma0": {"T5": 2.9, "T50": 3.4},
        }},
    }
    goodness = {
        "status": "converged", "chi2": 240.0, "reduced_chi2": 1.2,
        "n_points": 210, "n_variables": 5,
        "parameters": {"M.J1": 0.051, "M.J1_S1": 0.011, "M.J1_D1": 0.002,
                       "M.K1_M1": 0.031, "M.D_dip": 0.214, "M.g_factor": 2.1,
                       "M.m2_total": 0.93, "M.scale": 1.0,
                       "M.chi0[T5]": 0.021, "M.chi0[T50]": 0.018},
        "stderr": {"M.J1": 0.001, "M.J1_S1": 0.0005, "M.m2_total": 0.04},
        "dataset_chi2": {"T5": 110.0, "T50": 130.0},
        "dataset_reduced_chi2": {"T5": 1.1, "T50": 1.3},
        "dataset_n_points": {"T5": 100, "T50": 110},
        "skipped_datasets": ["broken"],
    }
    return FitTimelineEntry(
        name="joint fit",
        kind="result",
        snapshot={
            "datasets": [
                {"name": "T5", "parameters": {
                    "temperature": 5.0,
                    "magnetic_field": {"magnitude_T": 4.0,
                                       "direction": [1, 1, 1], "frame": "uvw"},
                }, "enabled": True, "fit_weight": 1.0, "scale_factor": 1.0},
                {"name": "T50", "parameters": {"temperature": 50.0},
                 "enabled": True, "fit_weight": 0.5, "scale_factor": 1.0},
            ],
            "models": [model],
        },
        created_at="2026-07-13 11:00",
        optimizer="least_squares",
        goodness=goodness,
        metadata={"diagnostics": {
            "T5": {"temperature": 5.0, "mu_eff_sq": 0.93, "chi_static_q0": 0.4,
                    "chi_static_qpeak": 0.9, "chi0_gamma0": 0.06,
                    "stability_margin": 0.2, "stability_ratio": 0.8,
                    "lambda_shift": 0.12,
                    "chi0_eff": 0.021},
            "T50": {"temperature": 50.0, "mu_eff_sq": 0.93},
        }},
    )


def test_latex_escape_covers_special_characters():
    assert latex_escape("a&b_c%d#e") == r"a\&b\_c\%d\#e"
    assert latex_escape("x^y~z") == r"x\textasciicircum{}y\textasciitilde{}z"
    assert latex_escape("${}") == r"\$\{\}"
    assert latex_escape("a\\b") == r"a\textbackslash{}b"


def test_report_uses_registered_model_section_hook():
    key = "_report_hook_test"

    def sections(_entry, model, _goodness, _context):
        return (("model", f"\\section{{Hook for {model['name']}}}"),)

    register_model_definition(
        ModelDefinition(
            key=key,
            label="Report hook",
            description="Test report hook.",
            data_types=("*",),
            factory=lambda _component: lambda data, _params: data.signal * 0.0,
            report_sections=sections,
        )
    )
    try:
        entry = _background_entry()
        entry.snapshot["models"][0]["type"] = key
        tex = render_fit_report_latex(entry, group_name="G")
        assert r"\section{Hook for bg}" in tex
        assert r"\subsection{Additional model components}" not in tex
    finally:
        MODEL_TYPE_REGISTRY.pop(key, None)


def test_generalized_paramagnon_report_defines_complex_and_static_response():
    entry = _background_entry()
    entry.snapshot["models"] = [
        {
            "name": "pm",
            "type": "generalized_paramagnon",
            "enabled": True,
            "parameters": {
                "chi_peak": 2.0,
                "gamma0": 3.0,
                "relaxation_power": 1.0,
                "inverse_mode_energy_sq": 0.04,
                "xi_x": 2.0,
                "xi_y": 1.0,
                "xi_z": 1.0,
                "xi_yx": 0.0,
                "xi_zx": 0.0,
                "xi_zy": 0.0,
                "q0_h": 0.5,
                "q0_k": 0.0,
                "q0_l": 0.0,
            },
            "config": {
                "spatial_power": 2.0,
                "center_offsets": [[0.0, 0.0, 0.0]],
                "center_combination": "sum",
                "periodic": True,
            },
            "fit_parameters": {},
            "sharing": {},
            "limits": {},
            "constraints": [],
            "applies_to": None,
            "metadata": {},
        }
    ]
    entry.goodness["parameters"] = {
        f"pm.{name}": value
        for name, value in entry.snapshot["models"][0]["parameters"].items()
    }
    entry.goodness["stderr"] = {}
    tex = render_fit_report_latex(entry, group_name="G")
    assert "Generalized paramagnon model" in tex
    assert "\\chi(\\mathbf q,E)" in tex
    assert "elastic datasets use" in tex
    assert "Millis" in tex
    _check_balanced_environments(tex)


def test_report_omits_disabled_and_zero_weight_visualization_datasets():
    entry = _background_entry()
    entry.snapshot["datasets"].append(
        {
            "name": "disabled_volume",
            "parameters": {},
            "enabled": False,
            "fit_weight": 1.0,
            "scale_factor": 1.0,
        }
    )

    tex = render_fit_report_latex(entry, group_name="G")

    assert "scan" in tex
    assert "disabled\\_volume" not in tex


def test_adversarial_names_cannot_inject_latex():
    entry = _background_entry()
    evil = r"foo_$\input{x}%bar"
    entry.snapshot["datasets"][0]["name"] = evil
    entry.goodness["dataset_chi2"] = {evil: 10.0}
    entry.goodness["dataset_n_points"] = {evil: 20}
    tex = render_fit_report_latex(entry, group_name=evil, nfit_version="0.8.1")
    assert "\\input{x}" not in tex
    assert r"\$\textbackslash{}input\{x\}\%bar" in tex
    _check_balanced_environments(tex)


def test_minimal_background_report_is_complete_document():
    entry = _background_entry()
    tex = render_fit_report_latex(entry, group_name="G", nfit_version="0.8.1")
    assert tex.startswith("\\documentclass")
    assert tex.rstrip().endswith("\\end{document}")
    _check_balanced_environments(tex)
    assert "Fit summary" in tex
    assert "constant\\_background" in tex or "constant_background" not in tex
    assert "Fitted parameters" in tex
    # No RPA component: no Hamiltonian section, no bibliography.
    assert "Model Hamiltonian" not in tex
    assert "thebibliography" not in tex
    # Diagnostics absent: section omitted.
    assert "Physics diagnostics" not in tex


def test_report_states_covariance_and_dataset_weight_caveats():
    absolute = _background_entry()
    absolute.goodness["covariance_mode"] = "absolute"
    absolute_tex = render_fit_report_latex(absolute, group_name="G")
    assert "supplied absolute data uncertainties" in absolute_tex
    assert "user-selected weights" in absolute_tex
    assert "not rescaled by the fit residuals" in absolute_tex

    residual = _background_entry()
    residual.goodness["covariance_mode"] = "residual"
    residual_tex = render_fit_report_latex(residual, group_name="G")
    assert "scaled by the reduced chi-squared" in residual_tex
    assert "cannot distinguish underestimated statistical errors" in residual_tex


def test_old_entry_without_new_keys_degrades_gracefully():
    entry = _background_entry()
    del entry.goodness["dataset_n_points"]
    tex = render_fit_report_latex(entry, group_name="G")
    assert "--" in tex
    _check_balanced_environments(tex)


def test_posterior_columns_appear_when_present():
    entry = _background_entry()
    entry.goodness["posterior"] = {
        "parameters": {"bg.constant": {"median": 1.49, "p16": 1.45, "p84": 1.53}}
    }
    tex = render_fit_report_latex(entry, group_name="G")
    assert "Median" in tex and "84\\%" in tex
    without = render_fit_report_latex(_background_entry(), group_name="G")
    assert "Median" not in without


def test_posterior_quantiles_use_interval_supported_precision():
    entry = _background_entry()
    entry.goodness["posterior"] = {
        "parameters": {
            "bg.constant": {"median": 1.234567, "p16": 1.16789, "p84": 1.31001}
        }
    }

    tex = render_fit_report_latex(entry, group_name="G")

    assert "$1.235$" in tex
    assert "$1.168$" in tex
    assert "$1.310$" in tex


def test_zero_standard_error_remains_visible():
    entry = _background_entry()
    entry.goodness["stderr"]["bg.constant"] = 0.0

    tex = render_fit_report_latex(entry, group_name="G")

    assert "$0$" in tex


def test_report_uses_selected_posterior_best_sample_and_asymmetric_errors():
    entry = _background_entry()
    entry.goodness["posterior"] = {
        "parameters": {"bg.constant": {"median": 1.49, "p16": 1.4, "p84": 1.6}}
    }
    entry.metadata["posterior_display"] = {
        "use_posterior_uncertainties": True,
        "use_best_sample": True,
        "best_sample": {"parameters": {"bg.constant": 1.55}},
    }

    tex = render_fit_report_latex(entry, group_name="G")

    assert "1.55" in tex
    assert "- 0.15 / + 0.050" in tex


@pytest.mark.parametrize(
    ("value", "error", "value_text", "error_text"),
    [
        (1.234567, 0.067891, "1.235", "0.068"),
        (1234.5, 21.7, "1235", "22"),
        (0.0062831, 0.0001247, "0.00628", "0.00012"),
        (1.5, 0.02, "1.500", "0.020"),
    ],
)
def test_report_rounds_parameter_values_to_uncertainty_precision(
    value, error, value_text, error_text
):
    entry = _background_entry()
    entry.goodness["parameters"]["bg.constant"] = value
    entry.goodness["stderr"]["bg.constant"] = error

    tex = render_fit_report_latex(entry, group_name="G")

    assert f"${value_text}$" in tex
    assert f"${error_text}$" in tex


def test_report_keeps_full_precision_outside_generated_display():
    entry = _background_entry()
    value = 1.23456789
    error = 0.06789123
    entry.goodness["parameters"]["bg.constant"] = value
    entry.goodness["stderr"]["bg.constant"] = error

    render_fit_report_latex(entry, group_name="G")

    assert entry.goodness["parameters"]["bg.constant"] == value
    assert entry.goodness["stderr"]["bg.constant"] == error


def test_full_rpa_report_covers_every_term():
    entry = _full_rpa_entry()
    tex = render_fit_report_latex(entry, group_name="pyro", nfit_version="0.8.1")
    _check_balanced_environments(tex)
    # Crystal section with the awkward spacegroup symbol intact (escaped text).
    assert "F d -3 m:2" in tex
    assert "10" in tex  # lattice constant
    # Hamiltonian terms.
    assert "Heisenberg exchange" in tex
    assert "Anisotropic exchange" in tex
    assert "\\begin{bmatrix}" in tex
    assert "Dzyaloshinskii--Moriya" in tex
    assert "Single-ion anisotropy" in tex
    assert "-\\frac12\\sum_i \\mathbf{S}_i\\cdot\\mathsf{A}_i" in tex
    assert "$\\mathsf{A}_i=\\mathsf{J}_{ii}$" in tex
    assert "Dipole--dipole" in tex
    assert "Zeeman term" in tex
    assert "+\\,g\\mu_B\\,\\mathbf{B}\\cdot\\sum_i \\mathbf{S}_i" in tex
    assert "E_L = g\\mu_B B" in tex
    assert "\\chi_\\perp(\\Gamma_\\perp\\pm iE_L)" in tex
    assert "Dynamic response" in tex
    # chi'' is written out explicitly for the shared relaxational/inertial form.
    assert "\\chi''_{s}(\\mathbf{Q}, E) = \\sum_\\nu" in tex
    assert "\\chi_0 E/\\Gamma_0" in tex
    assert "(\\delta_\\nu-a_EE^2)^2" in tex
    assert "\\Gamma_\\nu=\\Gamma_0\\delta_\\nu" in tex
    assert "w_\\nu(\\mathbf{Q})" in tex
    # Per-dataset chi0/gamma0 table from fitted_values.
    assert "0.021" in tex and "0.018" in tex
    # Closure with the Onsager equation and fitted target.
    assert "Self-consistency closure" in tex
    assert "Onsager" in tex
    assert "m^2_{\\mathrm{tot}}" in tex
    # Tensor polarization convention branch.
    assert "\\hat Q_\\alpha \\hat Q_\\beta" in tex
    # Diagnostics table with a missing-value cell for the sparse T50 record.
    assert "Physics diagnostics" in tex
    # Only cited references appear, in the bibliography.
    for key in (
        "sunny",
        "ross2011",
        "enjalran2004",
        "berlin1952",
        "moriya1985",
        "squires",
        "welch2022",
    ):
        assert f"\\bibitem{{{key}}}" in tex
    assert "\\bibitem{takahashi1986}" not in tex  # TAC not used
    # Skipped dataset note.
    assert "broken" in tex


def test_scalar_model_uses_isotropic_polarization_branch():
    entry = _full_rpa_entry()
    model = entry.snapshot["models"][0]
    for key in ("anisotropy", "sia", "dipole", "zeeman", "closure"):
        model["config"].pop(key, None)
    tex = render_fit_report_latex(entry, group_name="pyro")
    assert "\\mathcal P[\\chi''_s]=2\\chi''_s" in tex
    assert "\\frac{g}{2}" in tex
    assert "\\pi\\,[1 - e^{-E/k_BT}]" in tex
    assert "\\mu_0(g\\mu_B)^2\\widetilde\\chi''_s" in tex
    assert "m$^3$ per ion" in tex
    assert "Anisotropic exchange" not in tex
    assert "Self-consistency closure" not in tex
    _check_balanced_environments(tex)


def test_elastic_rpa_report_uses_quasistatic_cross_section():
    entry = _full_rpa_entry()
    for dataset in entry.snapshot["datasets"]:
        dataset["data_type"] = "single_crystal_elastic"

    tex = render_fit_report_latex(entry, group_name="elastic")

    assert "quasistatic approximation" in tex
    assert "\\frac{d\\sigma}{d\\Omega}" in tex
    assert "\\frac{d^2\\sigma}{d\\Omega\\,dE}" not in tex
    assert "magnetic Bragg intensity" in tex


def test_two_rpa_components_get_separate_hamiltonians():
    entry = _full_rpa_entry()
    import copy

    second = copy.deepcopy(entry.snapshot["models"][0])
    second["name"] = "M2"
    entry.snapshot["models"].append(second)
    tex = render_fit_report_latex(entry, group_name="pyro")
    assert "Model Hamiltonian (M)" in tex
    assert "Model Hamiltonian (M2)" in tex
    _check_balanced_environments(tex)


_HAS_TEX = any(shutil.which(name) for name in ("pdflatex", "tectonic", "xelatex", "lualatex"))


@pytest.mark.skipif(not _HAS_TEX, reason="no TeX engine installed")
def test_full_report_compiles_to_pdf(tmp_path):
    """The real LaTeX-validity referee: the full document must compile."""
    entry = _full_rpa_entry()
    entry.goodness["posterior"] = {
        "parameters": {"M.J1": {"median": 0.05, "p16": 0.049, "p84": 0.052}}
    }
    tex = render_fit_report_latex(entry, group_name="pyro", nfit_version="0.8.1")
    output = tmp_path / "report.pdf"
    compile_latex_pdf(tex, output)
    assert output.exists() and output.stat().st_size > 1000


@pytest.mark.skipif(not _HAS_TEX, reason="no TeX engine installed")
def test_compile_error_carries_log_tail(tmp_path):
    with pytest.raises(LatexCompileError) as excinfo:
        compile_latex_pdf(
            "\\documentclass{article}\\begin{document}\\undefinedmacro\\end{document}",
            tmp_path / "bad.pdf",
        )
    assert excinfo.value.engine is not None
    assert excinfo.value.log_tail.strip()


def test_compile_without_engine_names_installs(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(LatexCompileError) as excinfo:
        compile_latex_pdf("\\documentclass{article}", tmp_path / "x.pdf")
    assert excinfo.value.engine is None
    message = str(excinfo.value)
    assert "basictex" in message and "tectonic" in message
