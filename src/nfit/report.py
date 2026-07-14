"""Publication-grade LaTeX report of a stored fit result.

:func:`render_fit_report_latex` turns a persisted fit-history entry (its
snapshot, goodness-of-fit, and diagnostics metadata) into a complete,
standalone LaTeX article describing the fit statistics per dataset and the
model Hamiltonian term by term — exchange orbits, anisotropic tensors,
single-ion anisotropy, dipole coupling, Zeeman term, dynamic response,
self-consistency closure, and the cross-section convention — with only the
references actually used cited. :func:`compile_latex_pdf` compiles the source
with a locally installed TeX engine.

The renderer is pure (no Qt imports) and never raises on missing or old
data: entries recorded before a given metric existed render with ``--`` cells
or omit the section.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "LatexCompileError",
    "compile_latex_pdf",
    "latex_escape",
    "render_fit_report_latex",
]


# --- text helpers -----------------------------------------------------------

import re as _re

_ESCAPES = {
    "\\": r"\textbackslash{}",
    "{": r"\{",
    "}": r"\}",
    "$": r"\$",
    "&": r"\&",
    "#": r"\#",
    "%": r"\%",
    "_": r"\_",
    "^": r"\textasciicircum{}",
    "~": r"\textasciitilde{}",
}
_ESCAPE_PATTERN = _re.compile("|".join(_re.escape(char) for char in _ESCAPES))


def latex_escape(text: Any) -> str:
    """Escape a user-supplied string for LaTeX text mode (single pass)."""

    return _ESCAPE_PATTERN.sub(lambda match: _ESCAPES[match.group()], str(text))


def _fmt(value: Any) -> str:
    """Number formatting matching the GUI's 6-significant-digit convention."""

    if value is None:
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return latex_escape(value)
    if not np.isfinite(number):
        return "--"
    return f"{number:.6g}"


def _pm(value: Any, stderr: Any = None) -> str:
    """Inner math-mode ``value \\pm stderr`` (``\\text{--}`` when unknown)."""

    if value is None:
        return "\\text{--}"
    if stderr is None:
        return _fmt(value)
    if isinstance(stderr, Mapping):
        minus = stderr.get("minus")
        plus = stderr.get("plus")
        if minus is not None and plus is not None:
            return f"{_fmt(value)}^{{+{_fmt(plus)}}}_{{-{_fmt(minus)}}}"
    return f"{_fmt(value)} \\pm {_fmt(stderr)}"


def _fmt_pm(value: Any, stderr: Any = None) -> str:
    """``value \\pm stderr`` wrapped in math mode; ``--`` when unknown."""

    if value is None:
        return "--"
    return f"${_pm(value, stderr)}$"


def _fmt_uncertainty(value: Any, stderr: Any = None) -> str:
    """Format either a symmetric standard error or asymmetric posterior errors."""

    if stderr is None:
        return "--"
    if isinstance(stderr, Mapping):
        minus = stderr.get("minus")
        plus = stderr.get("plus")
        if minus is not None and plus is not None:
            return f"$- {_fmt(minus)} / + {_fmt(plus)}$"
    return _fmt(stderr)


def _bmatrix(matrix: Any) -> str:
    """A 3x3 (or any) matrix as an amsmath bmatrix."""

    rows = [
        " & ".join(_fmt(value) for value in row) for row in np.asarray(matrix, dtype=float)
    ]
    body = " \\\\\n".join(rows)
    return "\\begin{bmatrix}\n" + body + "\n\\end{bmatrix}"


def _offset_text(offset: Iterable[Any]) -> str:
    return "(" + ", ".join(_fmt(x) for x in offset) + ")"


# --- bibliography -----------------------------------------------------------

_REFERENCES: dict[str, str] = {
    "moriya1985": (
        "T. Moriya, \\emph{Spin Fluctuations in Itinerant Electron Magnetism}, "
        "Springer Series in Solid-State Sciences 56 (Springer, Berlin, 1985)."
    ),
    "sunny": (
        "D. Dahlbom \\emph{et al.}, Sunny.jl -- symmetry-distinct bond "
        "convention, \\texttt{https://github.com/SunnySuite/Sunny.jl}."
    ),
    "ross2011": (
        "K. A. Ross, L. Savary, B. D. Gaulin, and L. Balents, "
        "Phys.\\ Rev.\\ X \\textbf{1}, 021002 (2011); "
        "Phys.\\ Rev.\\ B \\textbf{84}, 064430 (2011)."
    ),
    "moriya1960": (
        "T. Moriya, ``Anisotropic superexchange interaction and weak "
        "ferromagnetism'', Phys.\\ Rev.\\ \\textbf{120}, 91 (1960)."
    ),
    "enjalran2004": (
        "M. Enjalran and M. J. P. Gingras, "
        "Phys.\\ Rev.\\ B \\textbf{70}, 174426 (2004)."
    ),
    "berlin1952": (
        "T. H. Berlin and M. Kac, Phys.\\ Rev.\\ \\textbf{86}, 821 (1952)."
    ),
    "brout1967": (
        "R. Brout and H. Thomas, Physics Physique Fizika \\textbf{3}, 317 (1967)."
    ),
    "takahashi1986": (
        "Y. Takahashi, J.\\ Phys.\\ Soc.\\ Jpn.\\ \\textbf{55}, 3553 (1986)."
    ),
    "brown": (
        "P. J. Brown, ``Magnetic form factors'', \\emph{International Tables "
        "for Crystallography}, Vol.\\ C, Section 4.4.5."
    ),
}


class _Citations:
    """Collects citation keys in first-use order."""

    def __init__(self) -> None:
        self.keys: list[str] = []

    def cite(self, *keys: str) -> str:
        for key in keys:
            if key not in self.keys:
                self.keys.append(key)
        return "\\cite{" + ",".join(keys) + "}"

    def bibliography(self) -> str:
        if not self.keys:
            return ""
        items = "\n".join(
            f"\\bibitem{{{key}}} {_REFERENCES[key]}" for key in self.keys
        )
        return (
            "\\begin{thebibliography}{9}\n" + items + "\n\\end{thebibliography}\n"
        )


# --- snapshot / goodness access ---------------------------------------------


def _snapshot_models(fit_entry: Any) -> list[dict[str, Any]]:
    snapshot = fit_entry.snapshot if isinstance(fit_entry.snapshot, dict) else {}
    models = snapshot.get("models")
    return [model for model in models if isinstance(model, dict)] if models else []


def _snapshot_datasets(fit_entry: Any) -> list[dict[str, Any]]:
    snapshot = fit_entry.snapshot if isinstance(fit_entry.snapshot, dict) else {}
    datasets = snapshot.get("datasets")
    return [d for d in datasets if isinstance(d, dict)] if datasets else []


def _goodness(fit_entry: Any) -> dict[str, Any]:
    raw = fit_entry.goodness if isinstance(fit_entry.goodness, dict) else {}
    metadata = _entry_metadata(fit_entry)
    display = metadata.get("posterior_display")
    if not isinstance(display, dict):
        return raw
    goodness = dict(raw)
    parameters = raw.get("parameters") if isinstance(raw.get("parameters"), dict) else {}
    displayed = dict(parameters)
    best = display.get("best_sample") if display.get("use_best_sample") else None
    best_params = best.get("parameters") if isinstance(best, dict) else None
    if isinstance(best_params, dict):
        displayed.update(best_params)
    goodness["parameters"] = displayed
    if display.get("use_posterior_uncertainties"):
        posterior = raw.get("posterior") if isinstance(raw.get("posterior"), dict) else {}
        posterior_params = posterior.get("parameters") if isinstance(posterior.get("parameters"), dict) else {}
        errors: dict[str, dict[str, float]] = {}
        for name, value in displayed.items():
            row = posterior_params.get(name) if isinstance(posterior_params.get(name), dict) else {}
            try:
                lower = float(row["p16"])
                upper = float(row["p84"])
                center = float(value)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(lower) and np.isfinite(upper) and np.isfinite(center):
                errors[str(name)] = {
                    "minus": max(0.0, center - lower),
                    "plus": max(0.0, upper - center),
                }
        goodness["stderr"] = errors
    return goodness


def _entry_metadata(fit_entry: Any) -> dict[str, Any]:
    return fit_entry.metadata if isinstance(fit_entry.metadata, dict) else {}


def _param_value(
    goodness: Mapping[str, Any], model: Mapping[str, Any], name: str
) -> tuple[Any, Any]:
    """Best-fit value and stderr of a component parameter.

    Prefers the fitted value under the qualified ``component.param`` name in
    the goodness record; falls back to the snapshot component's stored value
    (fixed parameters). ``stderr`` is ``None`` when unknown or fixed.
    """

    qualified = f"{model.get('name')}.{name}"
    parameters = goodness.get("parameters")
    if isinstance(parameters, dict) and qualified in parameters:
        stderr = goodness.get("stderr")
        err = stderr.get(qualified) if isinstance(stderr, dict) else None
        return parameters[qualified], err
    values = model.get("parameters")
    if isinstance(values, dict) and name in values:
        return values[name], None
    return None, None


def _per_dataset_values(model: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Per-dataset fitted values for a per-dataset-shared parameter."""

    metadata = model.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    fitted = metadata.get("fitted_values")
    if not isinstance(fitted, dict):
        return {}
    scoped = fitted.get(name)
    return dict(scoped) if isinstance(scoped, dict) else {}


def _config(model: Mapping[str, Any]) -> dict[str, Any]:
    config = model.get("config")
    return config if isinstance(config, dict) else {}


def _has_section(config: Mapping[str, Any], key: str) -> bool:
    block = config.get(key)
    if not isinstance(block, dict):
        return False
    if key in ("dipole", "zeeman"):
        return bool(block.get("enabled"))
    # anisotropy / sia: any enabled entry with a basis.
    return any(
        isinstance(spec, dict) and spec.get("enabled") and spec.get("basis")
        for spec in block.values()
    )


def _is_tensor_mode(config: Mapping[str, Any]) -> bool:
    return any(_has_section(config, key) for key in ("anisotropy", "sia", "dipole", "zeeman"))


# --- section builders --------------------------------------------------------


def _section_summary(fit_entry: Any, group_name: str) -> str:
    goodness = _goodness(fit_entry)
    datasets = _snapshot_datasets(fit_entry)
    diagnostics = _entry_metadata(fit_entry).get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    dataset_chi2 = goodness.get("dataset_chi2") or {}
    dataset_red = goodness.get("dataset_reduced_chi2") or {}
    n_points = goodness.get("dataset_n_points") or {}

    lines = ["\\section{Fit summary}", "\\begin{itemize}"]
    lines.append(f"\\item Data group: {latex_escape(group_name)}")
    lines.append(f"\\item Fit result: {latex_escape(fit_entry.name)}")
    if fit_entry.created_at:
        lines.append(f"\\item Recorded: {latex_escape(fit_entry.created_at)}")
    lines.append(f"\\item Optimizer: {latex_escape(fit_entry.optimizer)}")
    status = goodness.get("status")
    if status is not None:
        lines.append(f"\\item Status: {latex_escape(status)}")
    if goodness.get("chi2") is not None:
        lines.append(
            f"\\item Total $\\chi^2 = {_fmt(goodness.get('chi2'))}$, reduced "
            f"$\\chi^2_\\nu = {_fmt(goodness.get('reduced_chi2'))}$ with "
            f"{_fmt(goodness.get('n_points'))} points and "
            f"{_fmt(goodness.get('n_variables'))} free parameters"
        )
    lines.append("\\end{itemize}")

    fitted_names = list(dataset_chi2) or [d.get("name") for d in datasets]
    if fitted_names:
        lines.append("\\begin{longtable}{l l r r r r r}")
        lines.append("\\toprule")
        lines.append(
            "Dataset & $T$ (K) & $|B|$ (T) & $N$ & weight & $\\chi^2$ & "
            "$\\chi^2_\\nu$ \\\\"
        )
        lines.append("\\midrule")
        by_name = {d.get("name"): d for d in datasets}
        for name in fitted_names:
            dataset = by_name.get(name, {})
            parameters = dataset.get("parameters") or {}
            temperature = parameters.get("temperature")
            if temperature in (None, "") and isinstance(diagnostics.get(name), dict):
                temperature = diagnostics[name].get("temperature")
            field = parameters.get("magnetic_field")
            magnitude = (
                field.get("magnitude_T") if isinstance(field, dict) else None
            )
            lines.append(
                " & ".join(
                    [
                        latex_escape(name),
                        _fmt(temperature),
                        _fmt(magnitude),
                        _fmt(n_points.get(name)),
                        _fmt(dataset.get("fit_weight", 1.0)),
                        _fmt(dataset_chi2.get(name)),
                        _fmt(dataset_red.get(name)),
                    ]
                )
                + " \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{longtable}")
    skipped = goodness.get("skipped_datasets")
    if skipped:
        names = ", ".join(latex_escape(name) for name in skipped)
        lines.append(
            f"Datasets skipped (no applicable model component): {names}.\n"
        )
    return "\n".join(lines) + "\n"


def _section_crystal(model: Mapping[str, Any]) -> str:
    config = _config(model)
    crystal = config.get("crystal")
    if not isinstance(crystal, dict):
        return ""
    lines = [f"\\section{{Crystal structure ({latex_escape(model.get('name'))})}}"]
    spacegroup = crystal.get("spacegroup")
    if spacegroup:
        lines.append(f"Space group: {latex_escape(spacegroup)}.")
    lattice = crystal.get("lattice")
    if isinstance(lattice, dict):
        lines.append(
            "Lattice parameters: "
            f"$a = {_fmt(lattice.get('a'))}$\\,\\AA{{}}, "
            f"$b = {_fmt(lattice.get('b'))}$\\,\\AA{{}}, "
            f"$c = {_fmt(lattice.get('c'))}$\\,\\AA{{}}, "
            f"$\\alpha = {_fmt(lattice.get('alpha', 90.0))}^\\circ$, "
            f"$\\beta = {_fmt(lattice.get('beta', 90.0))}^\\circ$, "
            f"$\\gamma = {_fmt(lattice.get('gamma', 90.0))}^\\circ$."
        )
    sites = crystal.get("sites")
    magnetic = {str(label) for label in config.get("magnetic_sites", [])}
    if isinstance(sites, list) and sites:
        lines.append("\\begin{longtable}{l l l l}")
        lines.append("\\toprule")
        lines.append("Site & Position (fractional) & Ion & Magnetic \\\\")
        lines.append("\\midrule")
        for site in sites:
            if not isinstance(site, dict):
                continue
            label = str(site.get("label", ""))
            position = site.get("position", [])
            lines.append(
                " & ".join(
                    [
                        latex_escape(label),
                        _offset_text(position),
                        latex_escape(site.get("ion", "")),
                        "yes" if (not magnetic or label in magnetic) else "no",
                    ]
                )
                + " \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{longtable}")
    n_expanded = len(config.get("site_positions") or [])
    if n_expanded:
        lines.append(
            f"The space group expands these to {n_expanded} magnetic sites in "
            "the unit cell."
        )
    return "\n".join(lines) + "\n"


def _hamiltonian_display(config: Mapping[str, Any]) -> str:
    terms = ["-\\sum_{\\langle ij\\rangle} J_{ij}\\,\\mathbf{S}_i\\cdot\\mathbf{S}_j"]
    if _has_section(config, "anisotropy"):
        terms.append(
            "-\\sum_{\\langle ij\\rangle} \\mathbf{S}_i\\cdot"
            "\\mathsf{T}_{ij}\\,\\mathbf{S}_j"
        )
    if _has_section(config, "sia"):
        terms.append("-\\sum_i \\mathbf{S}_i\\cdot\\mathsf{A}_i\\,\\mathbf{S}_i")
    if _has_section(config, "dipole"):
        terms.append("+\\,\\mathcal{H}_{\\mathrm{dip}}")
    if _has_section(config, "zeeman"):
        terms.append(
            "-\\,g\\mu_B\\,\\mathbf{B}\\cdot\\sum_i \\mathbf{S}_i"
        )
    return (
        "\\begin{equation}\n\\mathcal{H} = "
        + " ".join(terms)
        + "\n\\end{equation}\n"
    )


def _section_exchange(
    model: Mapping[str, Any], goodness: Mapping[str, Any], cite: _Citations
) -> str:
    config = _config(model)
    orbits = config.get("orbits") or []
    lines = ["\\subsection{Heisenberg exchange}"]
    lines.append(
        "Each symmetry-distinct bond orbit carries one isotropic exchange "
        "constant (bonds counted once; the Hermitian conjugate is implied). "
        "The evaluator uses the extended-zone lattice Fourier transform "
        + cite.cite("sunny")
        + ":"
    )
    lines.append(
        "\\begin{equation}\nJ(\\mathbf{Q})_{ab} = \\sum_{\\mathrm{bonds}\\,"
        "(a,b,\\mathbf{n})} J_{\\mathrm{bond}}\\, e^{2\\pi i\\,\\mathbf{Q}\\cdot"
        "(\\mathbf{r}_b + \\mathbf{n} - \\mathbf{r}_a)} .\n\\end{equation}"
    )
    if orbits:
        lines.append("\\begin{longtable}{l r r l l}")
        lines.append("\\toprule")
        lines.append(
            "Orbit & $d$ (\\AA) & Multiplicity & Example bond & $J$ (meV) \\\\"
        )
        lines.append("\\midrule")
        for orbit in orbits:
            if not isinstance(orbit, dict):
                continue
            label = str(orbit.get("label", ""))
            bonds = orbit.get("bonds") or []
            example = ""
            if bonds and isinstance(bonds[0], dict):
                bond = bonds[0]
                example = (
                    f"{_fmt(bond.get('site_i'))} $\\to$ {_fmt(bond.get('site_j'))} "
                    f"+ {_offset_text(bond.get('offset', (0, 0, 0)))}"
                )
            value, stderr = _param_value(goodness, model, label)
            lines.append(
                " & ".join(
                    [
                        latex_escape(label),
                        _fmt(orbit.get("distance_angstrom")),
                        str(len(bonds)),
                        example,
                        _fmt_pm(value, stderr),
                    ]
                )
                + " \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def _section_anisotropy(
    model: Mapping[str, Any], goodness: Mapping[str, Any], cite: _Citations
) -> str:
    config = _config(model)
    if not _has_section(config, "anisotropy"):
        return ""
    lines = ["\\subsection{Anisotropic exchange}"]
    lines.append(
        "The symmetry-allowed anisotropic exchange of each orbit is expanded "
        "in an orthonormal basis of $3\\times3$ tensors on the representative "
        "bond "
        + cite.cite("ross2011", "moriya1960")
        + "; symmetry-equivalent bonds carry $R\\,\\mathsf{T}\\,R^{\\mathsf{T}}$ "
        "with $R$ the Cartesian rotation of the generating operation "
        "(transposed when the operation reverses the bond). Fitted "
        "coefficients multiply the unit-Frobenius basis matrices below "
        "(Cartesian crystal frame, meV):"
    )
    for orbit_label, spec in (config.get("anisotropy") or {}).items():
        if not isinstance(spec, dict) or not spec.get("enabled"):
            continue
        for element in spec.get("basis", []):
            if not isinstance(element, dict):
                continue
            name = f"{orbit_label}_{element.get('name')}"
            kind = str(element.get("kind", ""))
            kind_text = (
                "antisymmetric (Dzyaloshinskii--Moriya)"
                if kind == "dm"
                else "symmetric traceless"
            )
            value, stderr = _param_value(goodness, model, name)
            lines.append(
                f"\\paragraph{{{latex_escape(name)}}} {kind_text}, coefficient "
                f"{_fmt_pm(value, stderr)}\\,meV:"
            )
            lines.append(
                "\\begin{equation*}\n"
                + _bmatrix(element.get("matrix", np.zeros((3, 3))))
                + "\n\\end{equation*}"
            )
    return "\n".join(lines) + "\n"


def _section_sia(model: Mapping[str, Any], goodness: Mapping[str, Any]) -> str:
    config = _config(model)
    if not _has_section(config, "sia"):
        return ""
    lines = ["\\subsection{Single-ion anisotropy}"]
    lines.append(
        "Each magnetic site class carries the symmetry-allowed rank-2 "
        "traceless on-site tensor of its point group, rotated to each "
        "symmetry-equivalent site by the site's generating operation. "
        "Basis matrices are given on the representative site (Cartesian "
        "crystal frame, meV):"
    )
    for class_label, spec in (config.get("sia") or {}).items():
        if not isinstance(spec, dict) or not spec.get("enabled"):
            continue
        for element in spec.get("basis", []):
            if not isinstance(element, dict):
                continue
            name = f"{element.get('name')}_{class_label}"
            value, stderr = _param_value(goodness, model, name)
            lines.append(
                f"\\paragraph{{{latex_escape(name)}}} coefficient "
                f"{_fmt_pm(value, stderr)}\\,meV:"
            )
            lines.append(
                "\\begin{equation*}\n"
                + _bmatrix(element.get("matrix", np.zeros((3, 3))))
                + "\n\\end{equation*}"
            )
    return "\n".join(lines) + "\n"


def _section_dipole(
    model: Mapping[str, Any], goodness: Mapping[str, Any], cite: _Citations
) -> str:
    config = _config(model)
    if not _has_section(config, "dipole"):
        return ""
    from .dipole import dipole_coupling_constant

    value, stderr = _param_value(goodness, model, "D_dip")
    g_value, _ = _param_value(goodness, model, "g_factor")
    g_factor = float(g_value) if g_value is not None else 2.0
    physical = dipole_coupling_constant(g_factor)
    lines = ["\\subsection{Dipole--dipole coupling}"]
    lines.append(
        "Long-range magnetic dipole--dipole coupling is included through the "
        "Ewald-summed dipole tensor (tinfoil boundary conditions) "
        + cite.cite("enjalran2004")
        + ", multiplied by a single strength "
        f"$D_{{\\mathrm{{dip}}}} = {_pm(value, stderr)}"
        + "$\\,meV\\,\\AA$^3$. The physical point-dipole value is "
        f"$(\\mu_0/4\\pi)(g\\mu_B)^2 = {_fmt(physical)}$\\,meV\\,\\AA$^3$ "
        f"for $g = {_fmt(g_factor)}$."
    )
    return "\n".join(lines) + "\n"


def _section_zeeman(
    fit_entry: Any, model: Mapping[str, Any], goodness: Mapping[str, Any]
) -> str:
    config = _config(model)
    if not _has_section(config, "zeeman"):
        return ""
    lines = ["\\subsection{Zeeman term}"]
    g_value, g_err = _param_value(goodness, model, "g_factor")
    chi_ratio, chi_err = _param_value(goodness, model, "chi_perp_ratio")
    gamma_ratio, gamma_err = _param_value(goodness, model, "gamma_perp_ratio")
    lines.append(
        "In the applied field the local propagator becomes a tensor in the "
        "field frame ($\\hat z = \\hat B$): longitudinal "
        "$\\chi_\\parallel/(1 - i\\omega/\\Gamma_\\parallel)$ and transverse "
        "circular $\\chi_\\perp/\\bigl(1 - i(\\omega \\mp \\omega_L)/"
        "\\Gamma_\\perp\\bigr)$ with Larmor energy $\\omega_L = g\\mu_B B$. "
        f"Fitted: $g = {_pm(g_value, g_err)}$, "
        f"$\\chi_\\perp/\\chi_0 = {_pm(chi_ratio, chi_err)}$, "
        f"$\\Gamma_\\perp/\\Gamma_0 = {_pm(gamma_ratio, gamma_err)}$."
    )
    rows = []
    for dataset in _snapshot_datasets(fit_entry):
        parameters = dataset.get("parameters") or {}
        field = parameters.get("magnetic_field")
        if not isinstance(field, dict):
            continue
        rows.append(
            " & ".join(
                [
                    latex_escape(dataset.get("name")),
                    _fmt(field.get("magnitude_T")),
                    _offset_text(field.get("direction", (0, 0, 1))),
                    latex_escape(field.get("frame", "uvw")),
                ]
            )
            + " \\\\"
        )
    if rows:
        lines.append("\\begin{longtable}{l r l l}")
        lines.append("\\toprule")
        lines.append("Dataset & $|B|$ (T) & Direction & Frame \\\\")
        lines.append("\\midrule")
        lines.extend(rows)
        lines.append("\\bottomrule")
        lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def _section_dynamic_response(
    fit_entry: Any, model: Mapping[str, Any], goodness: Mapping[str, Any], cite: _Citations
) -> str:
    lines = ["\\subsection{Dynamic response}"]
    lines.append(
        "The local spins relax with a single-site susceptibility "
        "$\\chi_0(\\omega) = \\chi_0/(1 - i\\omega/\\Gamma_0)$, coupled in the "
        "random phase approximation "
        + cite.cite("moriya1985")
        + ":"
    )
    lines.append(
        "\\begin{equation}\n\\chi(\\mathbf{Q},\\omega) = \\bigl[\\,1 - "
        "\\chi_0(\\omega)\\,J(\\mathbf{Q})\\,\\bigr]^{-1}\\chi_0(\\omega) .\n"
        "\\end{equation}"
    )
    lines.append(
        "Diagonalizing the (Hermitian) exchange matrix at each $\\mathbf{Q}$, "
        "$J(\\mathbf{Q})\\,U_\\nu(\\mathbf{Q}) = \\lambda_\\nu(\\mathbf{Q})\\,"
        "U_\\nu(\\mathbf{Q})$, decouples the response into $N$ relaxational "
        "modes, and the measured dissipative susceptibility is their sum with "
        "neutron structure-factor weights:"
    )
    lines.append(
        "\\begin{equation}\n\\chi''(\\mathbf{Q}, E) = \\sum_\\nu "
        "w_\\nu(\\mathbf{Q})\\; \\chi_{\\mathbf{Q}\\nu}\\, \\frac{\\Gamma_\\nu\\, "
        "E}{E^2 + \\Gamma_\\nu^2},\n\\end{equation}"
    )
    lines.append(
        "where the mode susceptibility and relaxation rate are\n"
        "\\begin{equation}\n\\chi_{\\mathbf{Q}\\nu} = \\frac{\\chi_0}"
        "{1 - \\lambda_\\nu(\\mathbf{Q})\\,\\chi_0}, \\qquad "
        "\\Gamma_\\nu = \\Gamma_0\\,\\bigl[\\,1 - \\lambda_\\nu(\\mathbf{Q})\\,"
        "\\chi_0\\,\\bigr],\n\\end{equation}"
    )
    lines.append(
        "and the weight $w_\\nu(\\mathbf{Q}) = |\\sum_a U_{a\\nu}(\\mathbf{Q})|^2/N$ "
        "is the uniform sublattice sum (the extended-zone $J(\\mathbf{Q})$ "
        "already carries the pair phases). The relaxation rate softens as the "
        "Stoner-like criterion $\\max_{\\mathbf{Q},\\nu}\\lambda_\\nu(\\mathbf{Q})"
        "\\,\\chi_0 \\to 1$ is approached, at which the RPA denominator "
        "$1 - \\lambda_\\nu\\chi_0$ vanishes and the system orders."
    )
    if _is_tensor_mode(_config(model)):
        lines.append(
            "With the anisotropic terms active the modes carry Cartesian spin "
            "indices ($3N$ modes with vector amplitudes $w_{\\alpha\\nu}$), so "
            "$\\chi''$ above becomes the $3\\times3$ tensor "
            "$\\chi''_{\\alpha\\beta}(\\mathbf{Q}, E)$ contracted with the "
            "polarization weight in the cross section below."
        )
    chi0_scopes = _per_dataset_values(model, "chi0")
    gamma0_scopes = _per_dataset_values(model, "gamma0")
    if chi0_scopes or gamma0_scopes:
        names = sorted(set(chi0_scopes) | set(gamma0_scopes))
        lines.append(
            "The local parameters were fitted per dataset:"
        )
        lines.append("\\begin{longtable}{l r r}")
        lines.append("\\toprule")
        lines.append(
            "Dataset & $\\chi_0$ (meV$^{-1}$) & $\\Gamma_0$ (meV) \\\\"
        )
        lines.append("\\midrule")
        for name in names:
            lines.append(
                " & ".join(
                    [
                        latex_escape(name),
                        _fmt(chi0_scopes.get(name)),
                        _fmt(gamma0_scopes.get(name)),
                    ]
                )
                + " \\\\"
            )
        lines.append("\\bottomrule")
        lines.append("\\end{longtable}")
    else:
        chi0, chi0_err = _param_value(goodness, model, "chi0")
        gamma0, gamma0_err = _param_value(goodness, model, "gamma0")
        lines.append(
            f"Fitted values: $\\chi_0 = {_pm(chi0, chi0_err)}$"
            "\\,meV$^{-1}$, "
            f"$\\Gamma_0 = {_pm(gamma0, gamma0_err)}$\\,meV "
            "(shared across the fitted datasets)."
        )
    return "\n".join(lines) + "\n"


_CLOSURE_EQUATIONS = {
    "onsager": (
        "Onsager reaction field (spherical-model closure) "
        "\\cite{berlin1952,brout1967}: the exchange is shifted, "
        "$J(\\mathbf{Q}) \\to J(\\mathbf{Q}) - \\lambda(T)$, with "
        "$\\lambda(T)$ solved at each temperature so the per-site "
        "fluctuation amplitude satisfies the moment sum rule\n"
        "\\begin{equation}\n\\langle m^2\\rangle = \\frac{1}{\\pi V_{BZ}}"
        "\\int_{BZ} d\\mathbf{Q} \\int_0^{\\Lambda} d\\omega\\, "
        "\\coth\\!\\Bigl(\\frac{\\omega}{2k_BT}\\Bigr)\\, "
        "\\mathrm{Tr}\\,\\chi''(\\mathbf{Q},\\omega) = m^2_{\\mathrm{tot}} .\n"
        "\\end{equation}"
    ),
    "scr": (
        "Moriya self-consistent renormalization (SCR) \\cite{moriya1985}: "
        "the effective local susceptibility obeys\n"
        "\\begin{equation}\n\\chi_{0,\\mathrm{eff}}^{-1}(T) = \\chi_0^{-1} + "
        "u\\,\\langle m^2\\rangle(T),\n\\end{equation}\n"
        "solved self-consistently; here $\\chi_0$ denotes the bare ($T=0$) "
        "value and $u$ the mode-coupling constant."
    ),
    "tac": (
        "Takahashi total-amplitude conservation (TAC) \\cite{takahashi1986}: "
        "$\\chi_{0,\\mathrm{eff}}(T)$ is solved so the zero-point plus "
        "thermal fluctuation amplitude is conserved,\n"
        "\\begin{equation}\n\\langle m^2\\rangle_{\\mathrm{ZP}} + \\langle "
        "m^2\\rangle_{T} = A_{\\mathrm{tot}} .\n\\end{equation}"
    ),
}

_CLOSURE_CITATIONS = {
    "onsager": ("berlin1952", "brout1967"),
    "scr": ("moriya1985",),
    "tac": ("takahashi1986",),
}


def _section_closure(
    model: Mapping[str, Any], goodness: Mapping[str, Any], cite: _Citations
) -> str:
    config = _config(model)
    from .closures import ClosureSpec

    try:
        spec = ClosureSpec.from_config(config)
    except ValueError:
        spec = None
    if spec is None:
        return ""
    lines = ["\\subsection{Self-consistency closure}"]
    cite.cite(*_CLOSURE_CITATIONS[spec.mode])
    lines.append(_CLOSURE_EQUATIONS[spec.mode])
    details = [
        f"energy cutoff $\\Lambda = {_fmt(spec.energy_cutoff_mev)}$\\,meV",
        f"Brillouin-zone grid ${spec.bz_grid}^3$",
    ]
    if spec.mode == "onsager":
        if spec.moment_mode == "fitted":
            value, stderr = _param_value(goodness, model, "m2_total")
            details.append(
                f"fitted moment target $m^2_{{\\mathrm{{tot}}}} = "
                f"{_pm(value, stderr)}$"
            )
        else:
            details.append(
                f"fixed moment target $m^2_{{\\mathrm{{tot}}}} = "
                f"{_fmt(spec.moment_target)}$"
            )
    elif spec.mode == "scr":
        value, stderr = _param_value(goodness, model, "mode_coupling_u")
        details.append(f"mode coupling $u = {_pm(value, stderr)}$")
    elif spec.mode == "tac":
        if spec.moment_mode == "fitted":
            value, stderr = _param_value(goodness, model, "total_amplitude")
            details.append(
                f"fitted amplitude $A_{{\\mathrm{{tot}}}} = "
                f"{_pm(value, stderr)}$"
            )
        else:
            details.append(
                f"fixed amplitude $A_{{\\mathrm{{tot}}}} = {_fmt(spec.moment_target)}$"
            )
    lines.append(
        "Numerical settings: " + "; ".join(details) + ". Amplitude targets are "
        "in model units (the convention in which $J\\chi_0$ is dimensionless)."
    )
    return "\n".join(lines) + "\n"


def _section_cross_section(
    model: Mapping[str, Any], goodness: Mapping[str, Any], cite: _Citations
) -> str:
    config = _config(model)
    tensor = _is_tensor_mode(config)
    lines = ["\\subsection{Cross section}"]
    scale, scale_err = _param_value(goodness, model, "scale")
    ion = config.get("ion")
    if tensor:
        polarization = (
            "the unpolarized channel is computed from the dissipative tensor, "
            "$I \\propto \\sum_{\\alpha\\beta} \\tfrac{1}{3}"
            "(\\delta_{\\alpha\\beta} - \\hat Q_\\alpha \\hat Q_\\beta)\\,"
            "\\chi''_{\\alpha\\beta}$ (this reduces to the isotropic factor "
            "$2/3$ in the Heisenberg limit)"
        )
    else:
        polarization = (
            "the isotropic (Heisenberg) polarization factor $2/3$ multiplies "
            "$\\chi''$"
        )
    lines.append(
        "Measured intensity follows"
        "\n\\begin{equation}\nI(\\mathbf{Q}, E) = s\\, P(\\hat{\\mathbf{Q}})\\, "
        "|f(Q)|^2\\, \\frac{\\chi''(\\mathbf{Q}, E)}{1 - e^{-E/k_BT}},\n"
        "\\end{equation}\n"
        f"where {polarization}, "
        f"$s = {_pm(scale, scale_err)}$ is the overall scale"
        + (
            f", and $|f(Q)|^2$ is the {latex_escape(ion)} magnetic form factor "
            "in the $\\langle j_0\\rangle$ analytic approximation "
            + cite.cite("brown")
            if ion
            else ""
        )
        + "."
    )
    return "\n".join(lines) + "\n"


def _section_other_components(
    models: list[dict[str, Any]], goodness: Mapping[str, Any]
) -> str:
    others = [
        model
        for model in models
        if model.get("type") != "heisenberg_rpa" and model.get("enabled", True)
    ]
    if not others:
        return ""
    lines = ["\\subsection{Additional model components}"]
    for model in others:
        parameters = model.get("parameters") or {}
        parts = []
        for name in parameters:
            value, stderr = _param_value(goodness, model, name)
            parts.append(f"{latex_escape(name)} = {_fmt_pm(value, stderr)}")
        lines.append(
            f"\\paragraph{{{latex_escape(model.get('name'))}}} "
            f"type {latex_escape(model.get('type'))}; "
            + ("; ".join(parts) if parts else "no parameters")
            + "."
        )
    return "\n".join(lines) + "\n"


def _section_parameters(fit_entry: Any) -> str:
    goodness = _goodness(fit_entry)
    parameters = goodness.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        return ""
    stderr = goodness.get("stderr") if isinstance(goodness.get("stderr"), dict) else {}
    posterior = goodness.get("posterior") if isinstance(goodness.get("posterior"), dict) else {}
    posterior_params = (
        posterior.get("parameters") if isinstance(posterior.get("parameters"), dict) else {}
    )
    has_posterior = bool(posterior_params)
    display = _entry_metadata(fit_entry).get("posterior_display")
    use_posterior_uncertainties = bool(
        isinstance(display, dict) and display.get("use_posterior_uncertainties")
    )

    models = {model.get("name"): model for model in _snapshot_models(fit_entry)}

    def _row_meta(qualified: str) -> tuple[str, str]:
        component, _, param = qualified.partition(".")
        base = param.partition("[")[0]
        model = models.get(component)
        if not isinstance(model, dict):
            return "--", "--"
        varied = (model.get("fit_parameters") or {}).get(base)
        sharing = (model.get("sharing") or {}).get(base)
        scope = latex_escape(
            str(sharing.get("mode")) if isinstance(sharing, dict) else "global"
        )
        limits = (model.get("limits") or {}).get(base)
        limits_text = (
            f"[{_fmt(limits[0])}, {_fmt(limits[1])}]"
            if isinstance(limits, (list, tuple)) and len(limits) == 2
            else "--"
        )
        return ("varied" if varied else "fixed") + f" ({scope})", limits_text

    lines = ["\\section{Fitted parameters}"]
    uncertainty_heading = "68\\% error" if use_posterior_uncertainties else "Std.\\ err."
    if has_posterior:
        lines.append("\\begin{longtable}{l r r l l r r r}")
        header = (
            f"Parameter & Value & {uncertainty_heading} & Status & Limits & Median & "
            "16\\% & 84\\% \\\\"
        )
    else:
        lines.append("\\begin{longtable}{l r r l l}")
        header = f"Parameter & Value & {uncertainty_heading} & Status & Limits \\\\"
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")
    for name, value in parameters.items():
        status, limits_text = _row_meta(str(name))
        cells = [
            latex_escape(name),
            _fmt(value),
            _fmt_uncertainty(value, stderr.get(name)) if name in stderr else "--",
            status,
            limits_text,
        ]
        if has_posterior:
            row = posterior_params.get(name)
            row = row if isinstance(row, dict) else {}
            cells.extend(
                [_fmt(row.get("median")), _fmt(row.get("p16")), _fmt(row.get("p84"))]
            )
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


_DIAGNOSTIC_REPORT_COLUMNS = (
    ("temperature", "$T$ (K)"),
    ("mu_eff_sq", "$\\mu_{\\mathrm{eff}}^2$"),
    ("chi_static_q0", "$\\chi(0)$"),
    ("chi_static_qpeak", "$\\chi_{\\mathrm{pk}}$"),
    ("chi0_gamma0", "$\\chi_0\\Gamma_0$"),
    ("distance_to_instability", "$1 - \\lambda\\chi_0$"),
    ("lambda_shift", "$\\lambda$ (meV)"),
    ("chi0_eff", "$\\chi_{0,\\mathrm{eff}}$"),
)


def _section_diagnostics(fit_entry: Any) -> str:
    diagnostics = _entry_metadata(fit_entry).get("diagnostics")
    if not isinstance(diagnostics, dict) or not diagnostics:
        return ""
    lines = ["\\section{Physics diagnostics}"]
    lines.append(
        "Model-derived quantities per dataset: the effective fluctuating "
        "moment $\\mu_{\\mathrm{eff}}^2$ (Brillouin-zone and energy integral "
        "of $\\chi''$ up to the cutoff), the Kramers--Kronig static "
        "susceptibility at $\\mathbf{Q}=0$ and at its zone peak, and the "
        "distance to the RPA instability."
    )
    columns = _DIAGNOSTIC_REPORT_COLUMNS
    lines.append("\\begin{longtable}{l" + " r" * len(columns) + "}")
    lines.append("\\toprule")
    lines.append(
        "Dataset & " + " & ".join(header for _key, header in columns) + " \\\\"
    )
    lines.append("\\midrule")
    for name in sorted(diagnostics):
        record = diagnostics[name] if isinstance(diagnostics[name], dict) else {}
        lines.append(
            latex_escape(name)
            + " & "
            + " & ".join(_fmt(record.get(key)) for key, _header in columns)
            + " \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    return "\n".join(lines) + "\n"


def _section_methods(fit_entry: Any, nfit_version: str) -> str:
    goodness = _goodness(fit_entry)
    display = _entry_metadata(fit_entry).get("posterior_display")
    use_posterior_uncertainties = bool(
        isinstance(display, dict) and display.get("use_posterior_uncertainties")
    )
    use_best_sample = bool(isinstance(display, dict) and display.get("use_best_sample"))
    sampler_note = (
        " Posterior uncertainties were estimated by affine-invariant MCMC "
        "sampling (emcee) started from the least-squares optimum."
        if isinstance(goodness.get("posterior"), dict)
        else ""
    )
    if use_posterior_uncertainties:
        sampler_note += " The displayed uncertainties use the emcee 16--84\\% interval."
    if use_best_sample:
        sampler_note += " Displayed best-fit values use the highest-log-probability stored emcee sample."
    return (
        "\\section{Methods}\n"
        "Model parameters were optimized by weighted least squares, "
        "minimizing $\\chi^2 = \\sum_k \\bigl[(I_k^{\\mathrm{obs}} - "
        "I_k^{\\mathrm{model}})/\\sigma_k\\bigr]^2$ over the unmasked data "
        "points of every fitted dataset; the reduced $\\chi^2_\\nu$ divides "
        "by the number of points minus free parameters."
        + sampler_note
        + f" All computations used the \\texttt{{nfit}} package, version "
        f"{latex_escape(nfit_version)}.\n"
    )


# --- top-level renderer -------------------------------------------------------


def render_fit_report_latex(
    fit_entry: Any,
    *,
    group_name: str,
    nfit_version: str | None = None,
) -> str:
    """Render a stored fit result as a standalone LaTeX article.

    ``fit_entry`` is a :class:`~nfit.pipeline.FitTimelineEntry` (or anything
    with the same ``name``/``snapshot``/``goodness``/``metadata``/
    ``optimizer``/``created_at`` attributes). The output compiles with any
    standard LaTeX engine and needs only the amsmath, amssymb, booktabs,
    longtable, and geometry packages.
    """

    if nfit_version is None:
        try:
            from importlib.metadata import version

            nfit_version = version("nfit")
        except Exception:
            nfit_version = "unknown"

    goodness = _goodness(fit_entry)
    models = _snapshot_models(fit_entry)
    rpa_models = [
        model
        for model in models
        if model.get("type") == "heisenberg_rpa" and model.get("enabled", True)
    ]
    cite = _Citations()

    body: list[str] = []
    body.append(_section_summary(fit_entry, group_name))
    for model in rpa_models:
        body.append(_section_crystal(model))
    for model in rpa_models:
        config = _config(model)
        heading = "Model Hamiltonian"
        if len(rpa_models) > 1:
            heading += f" ({latex_escape(model.get('name'))})"
        body.append(f"\\section{{{heading}}}")
        body.append(
            "The active terms of the spin Hamiltonian are (each bond counted "
            "once; interaction matrices in the Cartesian crystal frame):"
        )
        body.append(_hamiltonian_display(config))
        body.append(_section_exchange(model, goodness, cite))
        body.append(_section_anisotropy(model, goodness, cite))
        body.append(_section_sia(model, goodness))
        body.append(_section_dipole(model, goodness, cite))
        body.append(_section_zeeman(fit_entry, model, goodness))
        body.append(_section_dynamic_response(fit_entry, model, goodness, cite))
        body.append(_section_closure(model, goodness, cite))
        body.append(_section_cross_section(model, goodness, cite))
    body.append(_section_other_components(models, goodness))
    body.append(_section_parameters(fit_entry))
    body.append(_section_diagnostics(fit_entry))
    body.append(_section_methods(fit_entry, nfit_version))
    body.append(cite.bibliography())

    title = latex_escape(f"Fit report: {group_name}")
    subtitle = latex_escape(fit_entry.name)
    date = latex_escape(fit_entry.created_at) if fit_entry.created_at else "\\today"
    preamble = "\n".join(
        [
            "\\documentclass[11pt]{article}",
            "\\usepackage[margin=2.5cm]{geometry}",
            "\\usepackage{amsmath}",
            "\\usepackage{amssymb}",
            "\\usepackage{booktabs}",
            "\\usepackage{longtable}",
            f"\\title{{{title}\\\\\\large {subtitle}}}",
            f"\\author{{nfit {latex_escape(nfit_version)}}}",
            f"\\date{{{date}}}",
            "\\begin{document}",
            "\\maketitle",
            "",
        ]
    )
    content = "\n".join(part for part in body if part)
    return preamble + content + "\n\\end{document}\n"


# --- PDF compilation ----------------------------------------------------------

_ENGINE_ORDER = ("pdflatex", "tectonic", "xelatex", "lualatex")
_INSTALL_HINT = (
    "Install a TeX engine to produce PDFs, e.g. on macOS "
    "'brew install --cask basictex' or in a conda environment "
    "'conda install -c conda-forge tectonic'."
)


class LatexCompileError(RuntimeError):
    """PDF compilation failed; ``engine`` is None when no engine was found."""

    def __init__(self, message: str, *, engine: str | None, log_tail: str = "") -> None:
        super().__init__(message)
        self.engine = engine
        self.log_tail = log_tail


def _find_engine() -> str | None:
    for name in _ENGINE_ORDER:
        if shutil.which(name):
            return name
    return None


def _log_tail(build_dir: Path, stdout: str, n_lines: int = 40) -> str:
    log_files = list(build_dir.glob("*.log"))
    text = log_files[0].read_text(errors="replace") if log_files else stdout
    lines = text.splitlines()
    return "\n".join(lines[-n_lines:])


def compile_latex_pdf(tex_source: str, output_path: str | Path) -> None:
    """Compile LaTeX source to a PDF at ``output_path`` using a local engine.

    Tries ``pdflatex``, ``tectonic``, ``xelatex``, ``lualatex`` in that order.
    The ``*latex`` engines run twice (longtable and cross-references need a
    second pass); tectonic resolves references internally. Raises
    :class:`LatexCompileError` with the tail of the build log on failure, or
    with ``engine=None`` and install instructions when no engine exists.
    """

    engine = _find_engine()
    if engine is None:
        raise LatexCompileError(
            "No LaTeX engine found on PATH (looked for "
            + ", ".join(_ENGINE_ORDER)
            + "). "
            + _INSTALL_HINT,
            engine=None,
        )
    output_path = Path(output_path)
    with tempfile.TemporaryDirectory(prefix="nfit_report_") as tmp:
        build_dir = Path(tmp)
        tex_path = build_dir / "report.tex"
        tex_path.write_text(tex_source, encoding="utf-8")
        if engine == "tectonic":
            commands = [[engine, "--outdir", str(build_dir), str(tex_path)]]
        else:
            command = [
                engine,
                "-interaction=nonstopmode",
                "-halt-on-error",
                "-output-directory",
                str(build_dir),
                str(tex_path),
            ]
            commands = [command, command]
        for command in commands:
            result = subprocess.run(
                command,
                cwd=build_dir,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                raise LatexCompileError(
                    f"{engine} failed with exit code {result.returncode}",
                    engine=engine,
                    log_tail=_log_tail(build_dir, result.stdout + result.stderr),
                )
        pdf_path = build_dir / "report.pdf"
        if not pdf_path.exists():
            raise LatexCompileError(
                f"{engine} reported success but produced no PDF",
                engine=engine,
                log_tail=_log_tail(build_dir, ""),
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf_path, output_path)
