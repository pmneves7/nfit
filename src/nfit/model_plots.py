"""Scriptable calculations and renderers owned by registered response models."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .brillouin_zone import band_path_reciprocal_lattice
from .electronic_structure import (
    BandResult,
    DensityOfStatesResult,
    FermiSurfaceResult,
    band_path,
    calculate_bands,
    density_of_states,
    electronic_energy_from_meV,
    electronic_energy_to_meV,
    electronic_model_from_component,
    fermi_surface,
    k_mesh,
    normalize_electronic_energy_unit,
    reciprocal_mesh_shape_for_spacing,
)
from .spin_fluctuations import generalized_paramagnon_susceptibility

_TIGHT_BINDING_PLOT_FIELDS = {
    "bands": {
        "band_path",
        "band_path_convention",
        "band_points_per_inv_angstrom",
    },
    "dos": {
        "dos_method",
        "dos_mesh",
        "dos_sampling_mode",
        "dos_sampling_accuracy",
        "dos_sampling_custom_rtol",
        "dos_symmetry",
        "dos_energy_min_meV",
        "dos_energy_max_meV",
        "dos_auto_energy_range",
        "dos_energy_points",
        "dos_broadening_meV",
    },
    "fermi_surface": {
        "fermi_mesh_mode",
        "fermi_spacing_inv_angstrom",
        "fermi_mesh",
        "fermi_energy_meV",
    },
}


@dataclass(frozen=True)
class ElectronicPlotStyle:
    """Scriptable presentation settings shared by 1D electronic plots."""

    line_color: str = "#1f77b4"
    line_width: float = 1.5
    marker: str = ""
    marker_size: float = 5.0
    marker_face_color: str = "none"
    font_size: float = 12.0
    border_width: float = 1.5
    show_legend: bool = True
    show_orbital_projections: bool = True
    fermi_line_color: str = "#000000"
    fermi_line_width: float = 1.0
    fermi_line_style: str = "--"
    symmetry_line_color: str = "#d9d9d9"
    symmetry_line_width: float = 1.0
    symmetry_line_style: str = "-"

    def __post_init__(self) -> None:
        for name in (
            "line_color",
            "marker_face_color",
            "fermi_line_color",
            "symmetry_line_color",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be a nonempty color specification")
        if str(self.marker) not in {"", "o", "s", "^", "D", "+", "x"}:
            raise ValueError("marker is not a supported electronic-plot marker")
        for name in (
            "line_width",
            "marker_size",
            "font_size",
            "border_width",
            "fermi_line_width",
            "symmetry_line_width",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if float(self.font_size) <= 0.0:
            raise ValueError("font_size must be positive")
        for name in ("fermi_line_style", "symmetry_line_style"):
            if str(getattr(self, name)) not in {"-", "--", ":", "-."}:
                raise ValueError(f"{name} is not a supported line style")


def apply_electronic_plot_style(
    figure: Any,
    style: ElectronicPlotStyle | None = None,
) -> ElectronicPlotStyle:
    """Apply reusable electronic-plot styling to a rendered figure."""

    settings = ElectronicPlotStyle() if style is None else style
    for axis in figure.axes:
        for line in axis.lines:
            role = line.get_gid()
            if role == "nfit-electronic-data":
                line.set_color(settings.line_color)
                line.set_linewidth(settings.line_width)
                line.set_marker(settings.marker)
                line.set_markersize(settings.marker_size)
                line.set_markeredgecolor(settings.line_color)
                line.set_markerfacecolor(settings.marker_face_color)
            elif role == "nfit-electronic-projection":
                line.set_color(settings.line_color)
                line.set_linewidth(settings.line_width)
                line.set_marker(settings.marker)
                line.set_markersize(settings.marker_size)
                line.set_markeredgecolor(settings.line_color)
                line.set_markerfacecolor(settings.marker_face_color)
                line.set_visible(settings.show_orbital_projections)
            elif role == "nfit-electronic-projection-key":
                line.set_visible(settings.show_orbital_projections)
            elif role == "nfit-fermi-line":
                line.set_color(settings.fermi_line_color)
                line.set_linewidth(settings.fermi_line_width)
                line.set_linestyle(settings.fermi_line_style)
            elif role == "nfit-symmetry-line":
                line.set_color(settings.symmetry_line_color)
                line.set_linewidth(settings.symmetry_line_width)
                line.set_linestyle(settings.symmetry_line_style)
        for collection in axis.collections:
            if collection.get_gid() == "nfit-electronic-projection":
                collection.set_visible(settings.show_orbital_projections)
        axis.xaxis.label.set_fontsize(settings.font_size)
        axis.yaxis.label.set_fontsize(settings.font_size)
        if hasattr(axis, "zaxis"):
            axis.zaxis.label.set_fontsize(settings.font_size)
        axis.title.set_fontsize(settings.font_size)
        axis.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            width=settings.border_width,
            labelsize=settings.font_size,
        )
        for spine in axis.spines.values():
            spine.set_linewidth(settings.border_width)
        legend_handles = [
            line
            for line in axis.lines
            if line.get_gid()
            in {
                "nfit-electronic-data",
                "nfit-electronic-projection",
                "nfit-electronic-projection-key",
            }
            and not str(line.get_label()).startswith("_")
            and (
                line.get_gid() == "nfit-electronic-data"
                or settings.show_orbital_projections
            )
        ]
        if legend_handles:
            legend = axis.legend(
                handles=legend_handles,
                labels=[line.get_label() for line in legend_handles],
                fancybox=False,
            )
        else:
            legend = axis.get_legend()
        if legend is not None:
            legend.set_visible(settings.show_legend and bool(legend_handles))
            for text in legend.get_texts():
                text.set_fontsize(settings.font_size)
            frame = legend.get_frame()
            frame.set_edgecolor("black")
            frame.set_linewidth(settings.border_width)
            frame.set_alpha(1.0)
            frame.set_boxstyle("square", pad=0.25)
    return settings


def configure_tight_binding_plot(
    component: Any,
    plot_key: str,
    **settings: Any,
) -> Any:
    """Atomically store one tight-binding plot's scriptable configuration."""

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("plot configuration requires a tight_binding component")
    key = str(plot_key)
    try:
        allowed = _TIGHT_BINDING_PLOT_FIELDS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_TIGHT_BINDING_PLOT_FIELDS))
        raise ValueError(
            f"unknown tight-binding plot {plot_key!r}; expected {choices}"
        ) from exc
    unexpected = set(settings) - allowed
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(
            f"unsupported {key} plot setting(s): {names}"
        )

    previous = copy.deepcopy(component.config)
    try:
        if key == "bands":
            convention = str(
                settings.get(
                    "band_path_convention",
                    component.config.get("band_path_convention", "manual"),
                )
            )
            if convention != "manual":
                from .brillouin_zone import set_tight_binding_standard_path

                set_tight_binding_standard_path(component, convention)
                settings = {
                    name: value
                    for name, value in settings.items()
                    if name not in {"band_path", "band_path_convention"}
                }
            else:
                settings["band_path_convention"] = "manual"
                component.config["band_path_metadata"] = {}
        component.config.update(settings)
        from .model_registry import validate_model_component

        validate_model_component(component)
        if key == "bands":
            path = component.config.get("band_path", ())
            if not isinstance(path, list) or len(path) < 2:
                raise ValueError(
                    "band_path must contain at least two labelled nodes"
                )
        elif key == "dos":
            mesh = component.config.get("dos_mesh", ())
            if (
                not isinstance(mesh, (list, tuple))
                or not mesh
                or any(
                    int(size) != float(size) or int(size) < 1
                    for size in mesh
                )
            ):
                raise ValueError(
                    "dos_mesh must contain positive integer sizes"
                )
            if int(component.config.get("dos_energy_points", 0)) < 2:
                raise ValueError("dos_energy_points must be at least two")
        elif key == "fermi_surface":
            mode = str(component.config.get("fermi_mesh_mode", "spacing"))
            if mode not in {"spacing", "size"}:
                raise ValueError(
                    "fermi_mesh_mode must be spacing or size"
                )
            spacing = float(
                component.config.get(
                    "fermi_spacing_inv_angstrom",
                    0.025,
                )
            )
            if not np.isfinite(spacing) or spacing <= 0.0:
                raise ValueError(
                    "fermi_spacing_inv_angstrom must be positive and finite"
                )
            mesh = component.config.get("fermi_mesh", ())
            if (
                not isinstance(mesh, (list, tuple))
                or not mesh
                or any(
                    int(size) != float(size) or int(size) < 2
                    for size in mesh
                )
            ):
                raise ValueError(
                    "fermi_mesh must contain integer sizes of at least two"
                )
    except Exception:
        component.config = previous
        raise
    return component


_LINDHARD_PLOT_FIELDS = {
    "complex_energy_scan": {
        "plot_q_reduced",
        "plot_energy_min_meV",
        "plot_energy_max_meV",
        "plot_energy_points",
        "plot_temperature_K",
    },
    "convergence": {
        "plot_q_reduced",
        "plot_energy_min_meV",
        "plot_energy_max_meV",
        "plot_temperature_K",
        "convergence_mesh_scales",
        "convergence_broadening_scales",
        "convergence_energy_points",
        "convergence_relative_floor",
    },
}


def configure_lindhard_plot(
    component: Any,
    plot_key: str,
    **settings: Any,
) -> Any:
    """Atomically store one Lindhard viewer's scriptable calculation settings."""

    if getattr(component, "type", None) != "lindhard":
        raise TypeError("Lindhard plot configuration requires a lindhard component")
    key = str(plot_key)
    try:
        allowed = _LINDHARD_PLOT_FIELDS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(_LINDHARD_PLOT_FIELDS))
        raise ValueError(
            f"unknown Lindhard plot {plot_key!r}; expected {choices}"
        ) from exc
    unexpected = set(settings) - allowed
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"unsupported {key} plot setting(s): {names}")
    previous = copy.deepcopy(component.config)
    try:
        component.config.update(settings)
        from .model_registry import validate_model_component

        validate_model_component(component)
    except Exception:
        component.config = previous
        raise
    return component


def _component_sampling_policy(
    config: Mapping[str, Any],
    prefix: str,
) -> Any:
    from .electronic_sampling import sampling_policy

    accuracy = str(config.get(f"{prefix}_sampling_accuracy", "standard"))
    return sampling_policy(
        accuracy,
        relative_tolerance=(
            float(config.get(f"{prefix}_sampling_custom_rtol", 0.01))
            if accuracy == "custom"
            else None
        ),
    )


def certify_tight_binding_dos_sampling(
    component: Any,
    *,
    progress_callback: Any | None = None,
) -> Any:
    """Certify and store one concrete DOS production mesh."""

    from .electronic_sampling import certify_dos_sampling

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("DOS sampling certification requires tight_binding")
    previous = copy.deepcopy(component.config)
    try:
        model = electronic_model_from_component(component)
        config = component.config
        points = int(config.get("dos_energy_points", 600))
        if bool(config.get("dos_auto_energy_range", False)):
            provisional = tight_binding_density_of_states(component)
            energy = np.asarray(provisional.energy_meV, dtype=float)
        else:
            energy = np.linspace(
                float(config.get("dos_energy_min_meV", -500.0)),
                float(config.get("dos_energy_max_meV", 500.0)),
                points,
            )
        certificate = certify_dos_sampling(
            model,
            energy,
            # automatic_mesh_ladder begins at half the supplied longest-axis
            # size. Keep certification independent of a previously accepted
            # production mesh and begin with a sparse 16-point longest axis.
            seed_mesh=[32] * model.dimension,
            policy=_component_sampling_policy(config, "dos"),
            max_refinements=int(
                config.get("dos_sampling_max_refinements", 7)
            ),
            max_mesh_points=int(
                config.get("dos_sampling_max_mesh_points", 2_000_000)
            ),
            symmetry=str(config.get("dos_symmetry", "auto")),
            broadening_meV=float(config.get("dos_broadening_meV", 5.0)),
            method=str(config.get("dos_method", "gaussian")),
            chemical_potential_meV=float(
                config.get("chemical_potential_meV", 0.0)
            ),
            projections=_projection_groups(component),
            progress_callback=progress_callback,
            **_electronic_execution_kwargs(component),
        )
        component.config["dos_sampling_certificate"] = certificate.to_dict()
        if certificate.certified:
            component.config["dos_mesh"] = list(certificate.chosen_mesh or ())
        from .model_registry import validate_model_component

        validate_model_component(component)
    except Exception:
        component.config = previous
        raise
    return certificate


def certify_lindhard_component_sampling(
    component: Any,
    components: Mapping[str, Any],
    *,
    q_reduced: ArrayLike | None = None,
    energy_meV: ArrayLike | None = None,
    temperature_K: float | None = None,
    progress_callback: Any | None = None,
) -> Any:
    """Certify and store a Lindhard mesh on an explicit or inspection domain."""

    from .electronic_sampling import certify_lindhard_sampling

    if getattr(component, "type", None) != "lindhard":
        raise TypeError("response sampling certification requires lindhard")
    previous = copy.deepcopy(component.config)
    try:
        source = _lindhard_source(component, components)
        model = electronic_model_from_component(source)
        config = component.config
        if energy_meV is None:
            energy = np.linspace(
                float(config.get("plot_energy_min_meV", -100.0)),
                float(config.get("plot_energy_max_meV", 100.0)),
                int(config.get("convergence_energy_points", 9)),
            )
        else:
            energy = np.asarray(energy_meV, dtype=float)
        if q_reduced is None:
            q = np.broadcast_to(
                np.asarray(
                    config.get("plot_q_reduced", [0.5, 0.5, 0.5]),
                    dtype=float,
                ),
                (energy.size, 3),
            )
            domain_kind = "model convergence viewer"
        else:
            q = np.asarray(q_reduced, dtype=float)
            domain_kind = "explicit caller-supplied points"
        temperature = (
            float(config.get("plot_temperature_K", 10.0))
            if temperature_K is None
            else float(temperature_K)
        )
        filling = (
            float(config.get("filling_per_cell", 1.0))
            if str(config.get("chemical_potential_mode", "source"))
            == "filling"
            else None
        )
        certificate = certify_lindhard_sampling(
            model,
            q,
            energy,
            seed_mesh=config.get("response_mesh", [16, 16, 16]),
            temperature_K=temperature,
            broadening_meV=float(
                component.parameters.get("broadening", 5.0)
            ),
            chemical_potential_meV=float(
                source.config.get("chemical_potential_meV", 0.0)
            ),
            filling_per_cell=filling,
            policy=_component_sampling_policy(config, "response"),
            max_refinements=int(
                config.get("response_sampling_max_refinements", 7)
            ),
            max_mesh_points=int(
                config.get("response_sampling_max_mesh_points", 500_000)
            ),
            mesh_shift=config.get(
                "response_mesh_shift", [0.0, 0.0, 0.0]
            ),
            symmetry=str(config.get("response_symmetry", "auto")),
            backend=str(config.get("response_backend", "auto")),
            workers=int(config.get("response_workers", 0)),
            max_batch_bytes=int(
                float(config.get("response_max_batch_mb", 256.0))
                * 1024**2
            ),
            transition_max_batch_bytes=int(
                float(
                    config.get(
                        "response_transition_max_batch_mb",
                        256.0,
                    )
                )
                * 1024**2
            ),
            transition_backend=str(
                config.get("response_transition_backend", "auto")
            ),
            progress_callback=progress_callback,
        )
        payload = certificate.to_dict()
        payload["provenance"]["domain_kind"] = domain_kind
        payload["provenance"]["source_component"] = source.name
        component.config["response_sampling_certificate"] = payload
        if certificate.certified:
            component.config["response_mesh"] = list(
                certificate.chosen_mesh or ()
            )
        from .model_registry import validate_model_component

        validate_model_component(component)
    except Exception:
        component.config = previous
        raise
    return certificate


def generalized_paramagnon_energy_scan(
    energy: ArrayLike,
    *,
    spatial_kernels: ArrayLike = (1.0,),
    chi_peak: float = 1.0,
    gamma0: float = 2.0,
    relaxation_power: float = 1.0,
    inverse_mode_energy_sq: float = 0.0,
) -> dict[str, np.ndarray]:
    """Calculate complex response curves at specified spatial kernels.

    This GUI-independent result is suitable for notebooks, scripts, tests, or
    the registered model-plot renderer.
    """

    energy_array = np.asarray(energy, dtype=float)
    if energy_array.ndim != 1:
        raise ValueError("energy must be one-dimensional")
    kernels = np.atleast_1d(np.asarray(spatial_kernels, dtype=float))
    response = generalized_paramagnon_susceptibility(
        kernels[:, None],
        energy_array[None, :],
        chi_peak=chi_peak,
        gamma0=gamma0,
        relaxation_power=relaxation_power,
        inverse_mode_energy_sq=inverse_mode_energy_sq,
    )
    return {
        "energy": energy_array,
        "spatial_kernels": kernels,
        "susceptibility": response,
    }


def render_generalized_paramagnon_energy_scan(
    result: dict[str, np.ndarray],
    *,
    axes: Sequence[Any] | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """Render real and imaginary susceptibility panels with Matplotlib."""

    import matplotlib.pyplot as plt

    if axes is None:
        figure, created_axes = plt.subplots(2, 1, sharex=True)
        real_axis, imaginary_axis = created_axes
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain real and imaginary axes")
        real_axis, imaginary_axis = axes
        figure = real_axis.figure
    energy = np.asarray(result["energy"], dtype=float)
    kernels = np.asarray(result["spatial_kernels"], dtype=float)
    response = np.asarray(result["susceptibility"], dtype=np.complex128)
    for index, kernel in enumerate(kernels):
        label = f"A = {kernel:g}"
        real_axis.plot(energy, response[index].real, label=label)
        imaginary_axis.plot(energy, response[index].imag, label=label)
    real_axis.set_ylabel(r"$\chi'$ (meV$^{-1}$)")
    imaginary_axis.set_ylabel(r"$\chi''$ (meV$^{-1}$)")
    imaginary_axis.set_xlabel("Energy transfer (meV)")
    real_axis.legend()
    imaginary_axis.legend()
    figure.tight_layout()
    return figure, (real_axis, imaginary_axis)


def generalized_paramagnon_energy_scan_script(
    *,
    energy: Sequence[float],
    spatial_kernels: Sequence[float] = (1.0,),
    chi_peak: float = 1.0,
    gamma0: float = 2.0,
    relaxation_power: float = 1.0,
    inverse_mode_energy_sq: float = 0.0,
) -> str:
    """Return an editable, GUI-free script reproducing an energy-scan plot."""

    arguments = {
        "energy": [float(value) for value in energy],
        "spatial_kernels": [float(value) for value in spatial_kernels],
        "chi_peak": float(chi_peak),
        "gamma0": float(gamma0),
        "relaxation_power": float(relaxation_power),
        "inverse_mode_energy_sq": float(inverse_mode_energy_sq),
    }
    return "\n".join(
        [
            "from nfit.model_plots import (",
            "    generalized_paramagnon_energy_scan,",
            "    render_generalized_paramagnon_energy_scan,",
            ")",
            "",
            f"settings = {arguments!r}",
            "result = generalized_paramagnon_energy_scan(**settings)",
            "figure, axes = render_generalized_paramagnon_energy_scan(result)",
            "figure.show()",
            "",
        ]
    )


def lindhard_energy_scan_unbound(component: Any) -> Any:
    """Explain why a linked electronic component is required."""

    raise ValueError(
        f"{component.name!r} must be calculated with its referenced "
        "tight-binding component"
    )


def _lindhard_source(
    component: Any,
    components: Mapping[str, Any],
) -> Any:
    config = component.config if isinstance(component.config, dict) else {}
    source_name = str(config.get("electronic_component", "")).strip()
    source = components.get(source_name) if source_name else None
    if source is None and not source_name:
        candidates = [
            item
            for item in components.values()
            if getattr(item, "type", None) == "tight_binding"
            and bool(getattr(item, "enabled", True))
        ]
        source = candidates[0] if len(candidates) == 1 else None
    if source is None or getattr(source, "type", None) != "tight_binding":
        raise ValueError(
            f"{component.name!r} must reference an enabled tight-binding "
            "component; an empty reference is automatic only when exactly "
            "one compatible component is enabled"
        )
    return source


def _validate_response_backend_for_plot(
    model: Any,
    mesh: Any,
    config: Mapping[str, Any],
    *,
    backend: str,
    workers: int,
    max_batch_bytes: int,
) -> None:
    if backend == "numpy" or not bool(config.get("response_validate_backend", True)):
        return
    from .electronic_backends import validate_electronic_backend

    count = min(
        int(config.get("response_backend_probe_points", 8)),
        mesh.reduced_coordinates.shape[0],
    )
    indices = np.linspace(
        0,
        mesh.reduced_coordinates.shape[0] - 1,
        count,
        dtype=int,
    )
    validation = validate_electronic_backend(
        model,
        mesh.reduced_coordinates[indices],
        backend,
        workers=workers,
        max_batch_bytes=max_batch_bytes,
        relative_tolerance=float(config.get("response_backend_rtol", 1.0e-10)),
        absolute_tolerance_meV=float(
            config.get("response_backend_atol_meV", 1.0e-8)
        ),
        require_requested_backend=True,
    )
    if not validation.passed:
        raise RuntimeError(
            f"electronic response backend validation failed: {validation.reason}"
        )


def lindhard_energy_scan(
    component: Any,
    components: Mapping[str, Any],
) -> Any:
    """Calculate the configured complex bare spin-response energy scan."""

    from .electronic_response import (
        ElectronicResponseCache,
        bare_spin_susceptibility,
        chemical_potential_for_filling,
        response_k_mesh,
    )

    source = _lindhard_source(component, components)
    model = electronic_model_from_component(source)
    config = component.config
    energy = np.linspace(
        float(config.get("plot_energy_min_meV", -100.0)),
        float(config.get("plot_energy_max_meV", 100.0)),
        int(config.get("plot_energy_points", 401)),
    )
    Q = np.broadcast_to(
        np.asarray(config.get("plot_q_reduced", [0.5, 0.5, 0.5]), dtype=float),
        (energy.size, 3),
    )
    filling_mesh = k_mesh(
        model,
        config.get("response_mesh", [16, 16, 16]),
        shift=config.get("response_mesh_shift", [0.0, 0.0, 0.0]),
        symmetry="full",
    )
    mesh = response_k_mesh(
        model,
        config.get("response_mesh", [16, 16, 16]),
        Q,
        shift=config.get("response_mesh_shift", [0.0, 0.0, 0.0]),
        symmetry=str(config.get("response_symmetry", "auto")),
    )
    temperature = float(config.get("plot_temperature_K", 10.0))
    execution = {
        "backend": str(config.get("response_backend", "auto")),
        "workers": int(config.get("response_workers", 0)),
        "max_batch_bytes": int(
            float(config.get("response_max_batch_mb", 256.0)) * 1024**2
        ),
        "transition_max_batch_bytes": int(
            float(config.get("response_transition_max_batch_mb", 256.0))
            * 1024**2
        ),
        "transition_backend": str(
            config.get("response_transition_backend", "auto")
        ),
        "q_evaluation": str(config.get("response_q_evaluation", "auto")),
        "q_interpolation_rtol": float(
            config.get("response_q_interpolation_rtol", 0.0)
        ),
        "q_interpolation_atol": float(
            config.get("response_q_interpolation_atol", 0.0)
        ),
        "q_interpolation_mesh": (
            None
            if not config.get("response_q_interpolation_mesh", [])
            else config.get("response_q_interpolation_mesh")
        ),
        "q_validation_points": int(
            config.get("response_q_validation_points", 8)
        ),
        "cache": ElectronicResponseCache(
            max_bytes=int(
                float(config.get("response_cache_mb", 512.0)) * 1024**2
            ),
            max_entries=int(config.get("response_cache_entries", 64)),
        ),
    }
    _validate_response_backend_for_plot(
        model,
        filling_mesh,
        config,
        backend=execution["backend"],
        workers=execution["workers"],
        max_batch_bytes=execution["max_batch_bytes"],
    )
    if str(config.get("chemical_potential_mode", "source")) == "filling":
        mu = chemical_potential_for_filling(
            model,
            filling_mesh,
            float(config.get("filling_per_cell", 1.0)),
            temperature_K=temperature,
            backend=execution["backend"],
            workers=execution["workers"],
            max_batch_bytes=execution["max_batch_bytes"],
            cache=execution["cache"],
        )
    else:
        mu = float(source.config.get("chemical_potential_meV", 0.0))
    return bare_spin_susceptibility(
        model,
        Q,
        energy,
        mesh,
        temperature_K=temperature,
        chemical_potential_meV=mu,
        broadening_meV=float(component.parameters.get("broadening", 5.0)),
        **execution,
    )


def render_lindhard_energy_scan(
    result: Any,
    *,
    axes: Sequence[Any] | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """Render real and imaginary isotropic bare spin susceptibility."""

    import matplotlib.pyplot as plt

    from .electronic_response import isotropic_spin_component

    if axes is None:
        figure, created_axes = plt.subplots(2, 1, sharex=True)
        real_axis, imaginary_axis = created_axes
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain real and imaginary axes")
        real_axis, imaginary_axis = axes
        figure = real_axis.figure
    response = isotropic_spin_component(result)
    real_axis.plot(result.energy_meV, response.real)
    imaginary_axis.plot(result.energy_meV, response.imag)
    real_axis.set_ylabel(r"$\chi'_{\mathrm{iso}}$ (meV$^{-1}$ cell$^{-1}$)")
    imaginary_axis.set_ylabel(
        r"$\chi''_{\mathrm{iso}}$ (meV$^{-1}$ cell$^{-1}$)"
    )
    imaginary_axis.set_xlabel("Energy transfer (meV)")
    Q = np.asarray(result.Q_reduced[0], dtype=float)
    response_label = (
        "Bare spin response"
        if result.response_kind == "bare"
        else "Interaction-dressed spin response"
    )
    dressing = (
        result.provenance.get("dressing", {})
        if isinstance(result.provenance, Mapping)
        else {}
    )
    static_stability = (
        dressing.get("static_stability", {})
        if isinstance(dressing, Mapping)
        else {}
    )
    warning = ""
    if isinstance(static_stability, Mapping) and static_stability.get(
        "near_instability"
    ):
        warning = " — near sampled RPA instability"
    elif isinstance(dressing, Mapping) and dressing.get("near_pole"):
        warning = " — near sampled RPA pole"
    real_axis.set_title(
        rf"{response_label} at $Q=({Q[0]:g},{Q[1]:g},{Q[2]:g})$ r.l.u."
        + warning
    )
    figure.tight_layout()
    return figure, (real_axis, imaginary_axis)


def lindhard_energy_scan_script_unbound(component: Any) -> str:
    """Explain why a linked electronic component is required."""

    lindhard_energy_scan_unbound(component)
    raise AssertionError("unreachable")


def lindhard_energy_scan_script(
    component: Any,
    components: Mapping[str, Any],
) -> str:
    """Return an editable GUI-free script for a bare response energy scan."""

    source = _lindhard_source(component, components)
    config = component.config
    lines = _component_model_script(source)
    lines.extend(
        [
            "import numpy as np",
            "from nfit import (",
            "    bare_spin_susceptibility,",
            "    chemical_potential_for_filling,",
            "    ElectronicResponseCache,",
            "    k_mesh,",
            "    response_k_mesh,",
            ")",
            "from nfit.model_plots import render_lindhard_energy_scan",
            "",
            f"sampling_certificate = {config.get('response_sampling_certificate', {})!r}",
            f"mesh_shape = {config.get('response_mesh', [16, 16, 16])!r}",
            f"mesh_shift = {config.get('response_mesh_shift', [0.0, 0.0, 0.0])!r}",
            "filling_mesh = k_mesh(model, mesh_shape, shift=mesh_shift, symmetry='full')",
            f"temperature_K = {float(config.get('plot_temperature_K', 10.0))!r}",
            f"energy_meV = np.linspace({float(config.get('plot_energy_min_meV', -100.0))!r}, {float(config.get('plot_energy_max_meV', 100.0))!r}, {int(config.get('plot_energy_points', 401))!r})",
            f"Q_reduced = np.broadcast_to(np.asarray({config.get('plot_q_reduced', [0.5, 0.5, 0.5])!r}, dtype=float), (energy_meV.size, 3))",
            "mesh = response_k_mesh(",
            "    model, mesh_shape, Q_reduced, shift=mesh_shift,",
            f"    symmetry={str(config.get('response_symmetry', 'auto'))!r},",
            ")",
            f"backend = {str(config.get('response_backend', 'auto'))!r}",
            f"workers = {int(config.get('response_workers', 0))!r}",
            f"max_batch_bytes = {int(float(config.get('response_max_batch_mb', 256.0)) * 1024**2)!r}",
            f"transition_max_batch_bytes = {int(float(config.get('response_transition_max_batch_mb', 256.0)) * 1024**2)!r}",
            f"transition_backend = {str(config.get('response_transition_backend', 'auto'))!r}",
            f"q_evaluation = {str(config.get('response_q_evaluation', 'auto'))!r}",
            f"q_interpolation_rtol = {float(config.get('response_q_interpolation_rtol', 0.0))!r}",
            f"q_interpolation_atol = {float(config.get('response_q_interpolation_atol', 0.0))!r}",
            f"q_interpolation_mesh = {config.get('response_q_interpolation_mesh', [])!r} or None",
            f"q_validation_points = {int(config.get('response_q_validation_points', 8))!r}",
            "response_cache = ElectronicResponseCache(",
            f"    max_bytes={int(float(config.get('response_cache_mb', 512.0)) * 1024**2)!r},",
            f"    max_entries={int(config.get('response_cache_entries', 64))!r},",
            ")",
        ]
    )
    if str(config.get("chemical_potential_mode", "source")) == "filling":
        lines.extend(
            [
                "chemical_potential_meV = chemical_potential_for_filling(",
                f"    model, filling_mesh, {float(config.get('filling_per_cell', 1.0))!r},",
                "    temperature_K=temperature_K, backend=backend,",
                "    workers=workers, max_batch_bytes=max_batch_bytes,",
                "    cache=response_cache,",
                ")",
            ]
        )
    else:
        lines.append(
            "chemical_potential_meV = "
            f"{float(source.config.get('chemical_potential_meV', 0.0))!r}"
        )
    lines.extend(
        [
            "result = bare_spin_susceptibility(",
            "    model, Q_reduced, energy_meV, mesh,",
            "    temperature_K=temperature_K,",
            "    chemical_potential_meV=chemical_potential_meV,",
            f"    broadening_meV={float(component.parameters.get('broadening', 5.0))!r},",
            "    backend=backend, workers=workers,",
            "    max_batch_bytes=max_batch_bytes,",
            "    transition_max_batch_bytes=transition_max_batch_bytes,",
            "    transition_backend=transition_backend,",
            "    q_evaluation=q_evaluation,",
            "    q_interpolation_rtol=q_interpolation_rtol,",
            "    q_interpolation_atol=q_interpolation_atol,",
            "    q_interpolation_mesh=q_interpolation_mesh,",
            "    q_validation_points=q_validation_points,",
            "    cache=response_cache,",
            ")",
            "figure, axes = render_lindhard_energy_scan(result)",
            "figure.show()",
            "",
        ]
    )
    return "\n".join(lines)


def lindhard_convergence_scan_unbound(component: Any) -> Any:
    """Explain why convergence needs a linked electronic component."""

    lindhard_energy_scan_unbound(component)
    raise AssertionError("unreachable")


def lindhard_convergence_scan(
    component: Any,
    components: Mapping[str, Any],
) -> Any:
    """Calculate separate response-mesh and broadening convergence metrics."""

    from .electronic_response import (
        ElectronicResponseCache,
        chemical_potential_for_filling,
        response_convergence_scan,
    )

    source = _lindhard_source(component, components)
    model = electronic_model_from_component(source)
    config = component.config
    base_shape = tuple(int(value) for value in config.get("response_mesh", [16] * 3))
    mesh_shapes = []
    for scale in config.get("convergence_mesh_scales", [0.5, 0.75, 1.0]):
        shape = tuple(max(1, int(round(value * float(scale)))) for value in base_shape)
        if shape not in mesh_shapes:
            mesh_shapes.append(shape)
    broadening = float(component.parameters.get("broadening", 5.0))
    broadenings = [
        broadening * float(scale)
        for scale in config.get("convergence_broadening_scales", [2.0, 1.0, 0.5])
    ]
    energy = np.linspace(
        float(config.get("plot_energy_min_meV", -100.0)),
        float(config.get("plot_energy_max_meV", 100.0)),
        int(config.get("convergence_energy_points", 9)),
    )
    Q = np.broadcast_to(
        np.asarray(config.get("plot_q_reduced", [0.5, 0.5, 0.5]), dtype=float),
        (energy.size, 3),
    )
    temperature = float(config.get("plot_temperature_K", 10.0))
    backend = str(config.get("response_backend", "auto"))
    workers = int(config.get("response_workers", 0))
    batch_bytes = int(float(config.get("response_max_batch_mb", 256.0)) * 1024**2)
    transition_bytes = int(
        float(config.get("response_transition_max_batch_mb", 256.0)) * 1024**2
    )
    cache = ElectronicResponseCache(
        max_bytes=int(float(config.get("response_cache_mb", 512.0)) * 1024**2),
        max_entries=int(config.get("response_cache_entries", 64)),
    )
    validation_mesh = k_mesh(
        model,
        mesh_shapes[-1],
        shift=config.get("response_mesh_shift", [0.0, 0.0, 0.0]),
        symmetry="full",
    )
    _validate_response_backend_for_plot(
        model,
        validation_mesh,
        config,
        backend=backend,
        workers=workers,
        max_batch_bytes=batch_bytes,
    )
    if str(config.get("chemical_potential_mode", "source")) == "filling":
        mu = chemical_potential_for_filling(
            model,
            validation_mesh,
            float(config.get("filling_per_cell", 1.0)),
            temperature_K=temperature,
            backend=backend,
            workers=workers,
            max_batch_bytes=batch_bytes,
            cache=cache,
        )
    else:
        mu = float(source.config.get("chemical_potential_meV", 0.0))
    return response_convergence_scan(
        model,
        Q,
        energy,
        mesh_shapes=mesh_shapes,
        broadenings_meV=broadenings,
        temperature_K=temperature,
        chemical_potential_meV=mu,
        mesh_shift=config.get("response_mesh_shift", [0.0, 0.0, 0.0]),
        relative_floor=float(config.get("convergence_relative_floor", 1.0e-12)),
        backend=backend,
        workers=workers,
        max_batch_bytes=batch_bytes,
        transition_max_batch_bytes=transition_bytes,
        transition_backend=str(
            config.get("response_transition_backend", "auto")
        ),
        cache=cache,
    )


def render_lindhard_convergence_scan(
    result: Any,
    *,
    axes: Sequence[Any] | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """Render mesh errors and broadening changes on separate panels."""

    import matplotlib.pyplot as plt

    if axes is None:
        figure, created_axes = plt.subplots(1, 2)
        mesh_axis, broadening_axis = created_axes
    else:
        if len(axes) != 2:
            raise ValueError("axes must contain mesh and broadening axes")
        mesh_axis, broadening_axis = axes
        figure = mesh_axis.figure
    mesh_sizes = np.asarray([np.prod(shape) for shape in result.mesh_shapes])
    floor = np.finfo(float).tiny
    for broadening_index, broadening in enumerate(result.broadenings_meV):
        mesh_axis.semilogy(
            mesh_sizes,
            np.maximum(
                result.mesh_max_relative_error[:, broadening_index],
                floor,
            ),
            marker="o",
            label=rf"$\eta={broadening:g}$ meV",
        )
    for mesh_index, shape in enumerate(result.mesh_shapes):
        broadening_axis.semilogy(
            result.broadenings_meV,
            np.maximum(
                result.broadening_max_relative_change[mesh_index],
                floor,
            ),
            marker="o",
            label=str(tuple(shape)),
        )
    mesh_axis.set_xlabel("Full-mesh point count")
    mesh_axis.set_ylabel("Maximum relative mesh error")
    broadening_axis.set_xlabel(r"Broadening $\eta$ (meV)")
    broadening_axis.set_ylabel("Maximum relative broadening change")
    mesh_axis.legend()
    broadening_axis.legend(title="mesh")
    figure.suptitle("Bare-response numerical convergence")
    figure.tight_layout()
    return figure, (mesh_axis, broadening_axis)


def lindhard_convergence_scan_script_unbound(component: Any) -> str:
    """Explain why a linked component context is required."""

    lindhard_convergence_scan_unbound(component)
    raise AssertionError("unreachable")


def lindhard_convergence_scan_script(
    component: Any,
    components: Mapping[str, Any],
) -> str:
    """Return an editable GUI-free convergence calculation and plot script."""

    from .model_registry import serialize_model_component

    source = _lindhard_source(component, components)
    payloads = [
        serialize_model_component(item, purpose="workflow")
        for item in (source, component)
    ]
    return "\n".join(
        [
            "from nfit import ModelComponentSpec",
            "from nfit.model_plots import (",
            "    lindhard_convergence_scan,",
            "    render_lindhard_convergence_scan,",
            ")",
            "",
            f"payloads = {payloads!r}",
            "models = [ModelComponentSpec(**payload) for payload in payloads]",
            "components = {item.name: item for item in models}",
            f"result = lindhard_convergence_scan(components[{component.name!r}], components)",
            "figure, axes = render_lindhard_convergence_scan(result)",
            "figure.show()",
            "",
        ]
    )


def electronic_rpa_energy_scan_unbound(component: Any) -> Any:
    """Explain why a linked bare-response component is required."""

    raise ValueError(
        f"{component.name!r} must be calculated with its referenced "
        "Lindhard and tight-binding components"
    )


def electronic_rpa_energy_scan(
    component: Any,
    components: Mapping[str, Any],
) -> Any:
    """Calculate a configured Stoner, matrix, or Hubbard-Hund RPA scan."""

    from .electronic_interactions import (
        correlated_basis_indices,
        hubbard_hund_spin_vertex,
        matrix_interaction_vertex,
        project_implicit_spin_response,
        rpa_dress_susceptibility,
        scalar_stoner_vertex,
    )
    from .electronic_response import (
        ElectronicResponseCache,
        bare_lindhard_susceptibility,
        chemical_potential_for_filling,
        orbital_pair_operator_basis,
    )

    response_name = str(component.config.get("response_component", "")).strip()
    response_component = components.get(response_name)
    if response_component is None or getattr(response_component, "type", None) != "lindhard":
        raise ValueError(
            f"{component.name!r} must reference an enabled Lindhard component"
        )
    tolerance = float(component.config.get("singular_tolerance", 1.0e-12))
    if component.type != "hubbard_hund_rpa":
        bare = lindhard_energy_scan(response_component, components)
        if component.type == "stoner_rpa":
            vertex = scalar_stoner_vertex(
                bare.operator_labels,
                float(component.parameters.get("I", 0.1)),
                energy_unit="eV",
            )
        elif component.type == "matrix_rpa":
            vertex = matrix_interaction_vertex(
                bare.operator_labels,
                float(component.parameters.get("scale", 0.1))
                * np.asarray(component.config.get("vertex_matrix", np.eye(3))),
                energy_unit="eV",
                channel=str(component.config.get("channel", "spin")),
            )
        else:
            raise ValueError(f"unsupported electronic RPA model {component.type!r}")
        return rpa_dress_susceptibility(
            bare,
            vertex,
            singular_tolerance=tolerance,
            near_pole_tolerance=float(
                component.config.get("near_pole_tolerance", 1.0e-3)
            ),
            static_warning_margin=float(
                component.config.get(
                    "static_stability_warning_margin", 0.05
                )
            ),
            reject_sampled_static_instability=bool(
                component.config.get(
                    "reject_sampled_static_instability", False
                )
            ),
        )

    source = _lindhard_source(response_component, components)
    model = electronic_model_from_component(source)
    config = response_component.config
    energy = np.linspace(
        float(config.get("plot_energy_min_meV", -100.0)),
        float(config.get("plot_energy_max_meV", 100.0)),
        int(config.get("plot_energy_points", 401)),
    )
    Q = np.broadcast_to(
        np.asarray(config.get("plot_q_reduced", [0.5, 0.5, 0.5]), dtype=float),
        (energy.size, 3),
    )
    mesh = k_mesh(
        model,
        config.get("response_mesh", [16, 16, 16]),
        shift=config.get("response_mesh_shift", [0.0, 0.0, 0.0]),
        symmetry="full",
    )
    temperature = float(config.get("plot_temperature_K", 10.0))
    execution = {
        "backend": str(config.get("response_backend", "auto")),
        "workers": int(config.get("response_workers", 0)),
        "max_batch_bytes": int(
            float(config.get("response_max_batch_mb", 256.0)) * 1024**2
        ),
    }
    cache = ElectronicResponseCache(
        max_bytes=int(float(config.get("response_cache_mb", 512.0)) * 1024**2),
        max_entries=int(config.get("response_cache_entries", 64)),
    )
    _validate_response_backend_for_plot(
        model,
        mesh,
        config,
        backend=execution["backend"],
        workers=execution["workers"],
        max_batch_bytes=execution["max_batch_bytes"],
    )
    if str(config.get("chemical_potential_mode", "source")) == "filling":
        mu = chemical_potential_for_filling(
            model,
            mesh,
            float(config.get("filling_per_cell", 1.0)),
            temperature_K=temperature,
            **execution,
            cache=cache,
        )
    else:
        mu = float(source.config.get("chemical_potential_meV", 0.0))
    shells = component.config.get("correlated_shells", [])
    if isinstance(shells, str):
        shells = [item.strip() for item in shells.split(",") if item.strip()]
    indices = correlated_basis_indices(model, shells)
    operators = orbital_pair_operator_basis(model, list(indices))
    bare_pairs = bare_lindhard_susceptibility(
        model,
        Q,
        energy,
        mesh,
        operators,
        temperature_K=temperature,
        chemical_potential_meV=mu,
        broadening_meV=float(
            response_component.parameters.get("broadening", 5.0)
        ),
        **execution,
        transition_max_batch_bytes=int(
            float(config.get("response_transition_max_batch_mb", 256.0))
            * 1024**2
        ),
        transition_backend=str(
            config.get("response_transition_backend", "auto")
        ),
        cache=cache,
        q_evaluation=str(config.get("response_q_evaluation", "auto")),
        q_interpolation_rtol=float(
            config.get("response_q_interpolation_rtol", 0.0)
        ),
        q_interpolation_atol=float(
            config.get("response_q_interpolation_atol", 0.0)
        ),
        q_interpolation_mesh=(
            None
            if not config.get("response_q_interpolation_mesh", [])
            else config.get("response_q_interpolation_mesh")
        ),
        q_validation_points=int(
            config.get("response_q_validation_points", 8)
        ),
    )
    vertex = hubbard_hund_spin_vertex(
        model,
        indices,
        U=float(component.parameters.get("U", 1.0)),
        U_prime=float(component.parameters.get("U_prime", 0.6)),
        J_H=float(component.parameters.get("J_H", 0.2)),
        J_pair=float(component.parameters.get("J_pair", 0.2)),
        rotationally_invariant=bool(
            component.config.get("rotationally_invariant", True)
        ),
        energy_unit="eV",
    )
    return project_implicit_spin_response(
        rpa_dress_susceptibility(
            bare_pairs,
            vertex,
            singular_tolerance=tolerance,
            near_pole_tolerance=float(
                component.config.get("near_pole_tolerance", 1.0e-3)
            ),
            static_warning_margin=float(
                component.config.get(
                    "static_stability_warning_margin", 0.05
                )
            ),
            reject_sampled_static_instability=bool(
                component.config.get(
                    "reject_sampled_static_instability", False
                )
            ),
        ),
        model,
        indices,
    )


def electronic_rpa_energy_scan_script_unbound(component: Any) -> str:
    """Explain why a linked component context is required."""

    electronic_rpa_energy_scan_unbound(component)
    raise AssertionError("unreachable")


def electronic_rpa_energy_scan_script(
    component: Any,
    components: Mapping[str, Any],
) -> str:
    """Return an editable GUI-free script for an RPA energy scan."""

    from .model_registry import serialize_model_component

    response_name = str(component.config.get("response_component", "")).strip()
    response = components.get(response_name)
    if response is None:
        electronic_rpa_energy_scan_unbound(component)
    source = _lindhard_source(response, components)
    payloads = [
        serialize_model_component(item, purpose="workflow")
        for item in (source, response, component)
    ]
    return "\n".join(
        [
            "from nfit import ModelComponentSpec",
            "from nfit.model_plots import (",
            "    electronic_rpa_energy_scan,",
            "    render_lindhard_energy_scan,",
            ")",
            "",
            f"payloads = {payloads!r}",
            "models = [ModelComponentSpec(**payload) for payload in payloads]",
            "components = {item.name: item for item in models}",
            f"result = electronic_rpa_energy_scan(components[{component.name!r}], components)",
            "figure, axes = render_lindhard_energy_scan(result)",
            "figure.show()",
            "",
        ]
    )


def _projection_groups(component: Any) -> dict[str, list[int]]:
    config = component.config if isinstance(component.config, dict) else {}
    raw = config.get("projection_groups", {})
    if not isinstance(raw, dict):
        raise ValueError("projection_groups must be a dictionary of basis-index lists")
    return {
        str(label): [int(index) for index in indices]
        for label, indices in raw.items()
    }


def _electronic_display_unit(component: Any) -> str:
    config = component.config if isinstance(component.config, dict) else {}
    return normalize_electronic_energy_unit(
        config.get("electronic_energy_unit", "eV")
    )


def _with_electronic_display_unit(result: Any, component: Any) -> Any:
    return replace(
        result,
        provenance={
            **dict(result.provenance),
            "display_energy_unit": _electronic_display_unit(component),
        },
    )


def _electronic_execution_kwargs(component: Any) -> dict[str, Any]:
    config = component.config if isinstance(component.config, dict) else {}
    workers = int(config.get("electronic_workers", 0))
    return {
        "backend": str(config.get("electronic_backend", "auto")),
        "workers": None if workers == 0 else workers,
        "max_batch_bytes": int(
            float(config.get("electronic_max_batch_mb", 256.0)) * 1024**2
        ),
    }


def tight_binding_band_structure(component: Any) -> BandResult:
    """Calculate a configured tight-binding band path for a model component."""

    model = electronic_model_from_component(component)
    config = component.config
    raw_nodes = config.get(
        "band_path",
        [
            {"label": r"$\Gamma$", "k": [0.0, 0.0, 0.0]},
            {"label": "X", "k": [0.5, 0.0, 0.0]},
            {"label": "M", "k": [0.5, 0.5, 0.0]},
            {"label": r"$\Gamma$", "k": [0.0, 0.0, 0.0]},
        ],
    )
    if not isinstance(raw_nodes, list) or len(raw_nodes) < 2:
        raise ValueError("band_path must contain at least two labeled nodes")
    nodes = [item["k"] if isinstance(item, dict) else item for item in raw_nodes]
    labels = [
        str(item.get("label", f"K{index}")) if isinstance(item, dict) else f"K{index}"
        for index, item in enumerate(raw_nodes)
    ]
    breaks = [
        bool(item.get("break_before", False))
        if isinstance(item, dict)
        else False
        for item in raw_nodes
    ]
    sampling = band_path(
        model,
        nodes,
        labels=labels,
        break_before=breaks,
        points_per_inv_angstrom=float(
            config.get("band_points_per_inv_angstrom", 80.0)
        ),
        coordinate_reciprocal_lattice=band_path_reciprocal_lattice(component),
    )
    return _with_electronic_display_unit(
        calculate_bands(
            model,
            sampling,
            chemical_potential_meV=float(
                config.get("chemical_potential_meV", 0.0)
            ),
            projections=_projection_groups(component),
            include_eigenvectors=False,
            **_electronic_execution_kwargs(component),
        ),
        component,
    )


def tight_binding_density_of_states(component: Any) -> DensityOfStatesResult:
    """Calculate configured total and orbital-projected density of states."""

    model = electronic_model_from_component(component)
    config = component.config
    projections = _projection_groups(component)
    symmetry = str(config.get("dos_symmetry", "auto"))
    method = str(config.get("dos_method", "gaussian"))
    mesh = k_mesh(
        model,
        config.get("dos_mesh", [40, 40, 40]),
        symmetry="full" if method == "tetrahedron" or projections else symmetry,
    )
    energy_points = int(config.get("dos_energy_points", 600))
    if bool(config.get("dos_auto_energy_range", False)):
        energy = None
    else:
        energy_min = float(config.get("dos_energy_min_meV", -500.0))
        energy_max = float(config.get("dos_energy_max_meV", 500.0))
        energy = np.linspace(energy_min, energy_max, energy_points)
    return _with_electronic_display_unit(
        density_of_states(
            model,
            mesh,
            energy,
            broadening_meV=float(config.get("dos_broadening_meV", 5.0)),
            method=method,
            symmetry=symmetry,
            energy_points=energy_points,
            chemical_potential_meV=float(
                config.get("chemical_potential_meV", 0.0)
            ),
            projections=projections,
            **_electronic_execution_kwargs(component),
        ),
        component,
    )


def tight_binding_fermi_surface(component: Any) -> FermiSurfaceResult:
    """Calculate configured Fermi points, contours, or surfaces."""

    model = electronic_model_from_component(component)
    config = component.config
    mesh = config.get("fermi_mesh", [64, 64, 64])
    if str(config.get("fermi_mesh_mode", "spacing")) == "spacing":
        mesh = reciprocal_mesh_shape_for_spacing(
            model,
            float(config.get("fermi_spacing_inv_angstrom", 0.025)),
        )
    return _with_electronic_display_unit(
        fermi_surface(
            model,
            mesh,
            target_energy_meV=float(
                config.get(
                    "fermi_energy_meV",
                    config.get("chemical_potential_meV", 0.0),
                )
            ),
            projections=_projection_groups(component),
            **_electronic_execution_kwargs(component),
        ),
        component,
    )


def render_band_structure(
    result: BandResult,
    *,
    energy_unit: str | None = None,
    axes: Sequence[Any] | None = None,
    style: ElectronicPlotStyle | None = None,
) -> tuple[Any, Any]:
    """Render bands in eV by default, or an explicitly requested energy unit."""

    import matplotlib.pyplot as plt

    if result.sampling.kind != "path":
        raise ValueError("band-structure rendering requires a path result")
    if axes is None:
        figure, axis = plt.subplots()
    else:
        if len(axes) != 1:
            raise ValueError("axes must contain one band axis")
        axis = axes[0]
        figure = axis.figure
    distance = np.asarray(result.sampling.path_distance_inv_angstrom)
    unit = normalize_electronic_energy_unit(
        energy_unit or result.provenance.get("display_energy_unit", "eV")
    )
    relative = electronic_energy_from_meV(
        result.energies_meV - result.chemical_potential_meV, unit
    )
    band_lines = axis.plot(distance, relative, zorder=1)
    for line in band_lines:
        line.set_gid("nfit-electronic-data")
    for group_index, (label, weights) in enumerate(result.projected_weights.items()):
        color = f"C{group_index % 10}"
        for band_index in range(relative.shape[1]):
            projection = axis.scatter(
                distance,
                relative[:, band_index],
                s=4.0 + 24.0 * weights[:, band_index],
                color=color,
                alpha=0.55,
                linewidths=0,
                zorder=2,
            )
            projection.set_gid("nfit-electronic-projection")
        projection_key = axis.plot(
            [],
            [],
            marker="o",
            linestyle="",
            color=color,
            label=label,
        )[0]
        projection_key.set_gid("nfit-electronic-projection-key")
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    for index, label in result.sampling.labels:
        position = float(distance[int(index)])
        if tick_positions and np.isclose(
            position,
            tick_positions[-1],
            rtol=0.0,
            atol=1.0e-12,
        ):
            tick_labels[-1] = f"{tick_labels[-1]}|{label}"
        else:
            tick_positions.append(position)
            tick_labels.append(str(label))
    axis.set_xticks(tick_positions, tick_labels)
    for position in tick_positions:
        symmetry_line = axis.axvline(position, zorder=0)
        symmetry_line.set_gid("nfit-symmetry-line")
    fermi_line = axis.axhline(0.0)
    fermi_line.set_gid("nfit-fermi-line")
    axis.set_xlim(distance[0], distance[-1])
    axis.set_ylabel(rf"$E-\mu$ ({unit})")
    axis.set_xlabel("Wavevector path")
    if result.projected_weights:
        axis.legend(fancybox=False)
    apply_electronic_plot_style(figure, style)
    figure.tight_layout()
    return figure, axis


def render_density_of_states(
    result: DensityOfStatesResult,
    *,
    energy_unit: str | None = None,
    axes: Sequence[Any] | None = None,
    style: ElectronicPlotStyle | None = None,
) -> tuple[Any, Any]:
    """Render DOS in eV by default, preserving its integrated state count."""

    import matplotlib.pyplot as plt

    if axes is None:
        figure, axis = plt.subplots()
    else:
        if len(axes) != 1:
            raise ValueError("axes must contain one density-of-states axis")
        axis = axes[0]
        figure = axis.figure
    unit = normalize_electronic_energy_unit(
        energy_unit or result.provenance.get("display_energy_unit", "eV")
    )
    relative = electronic_energy_from_meV(
        result.energy_meV - result.chemical_potential_meV, unit
    )
    density_scale = float(electronic_energy_to_meV(1.0, unit))
    total_line = axis.plot(
        relative,
        result.total_per_meV_cell * density_scale,
        label="total",
    )[0]
    total_line.set_gid("nfit-electronic-data")
    for label, values in result.projected_per_meV_cell.items():
        line = axis.plot(relative, values * density_scale, label=label)[0]
        line.set_gid("nfit-electronic-projection")
    fermi_line = axis.axvline(0.0)
    fermi_line.set_gid("nfit-fermi-line")
    axis.set_xlabel(rf"$E-\mu$ ({unit})")
    axis.set_ylabel(f"DOS (states / {unit} / cell)")
    axis.legend(fancybox=False)
    apply_electronic_plot_style(figure, style)
    figure.tight_layout()
    return figure, axis


def render_fermi_surface(
    result: FermiSurfaceResult,
    *,
    energy_unit: str | None = None,
    axes: Sequence[Any] | None = None,
) -> tuple[Any, Any]:
    """Render a 1D, 2D, or 3D Fermi-surface result in reduced coordinates."""

    import matplotlib.pyplot as plt

    if axes is None:
        if result.dimension == 3:
            figure = plt.figure()
            axis = figure.add_subplot(111, projection="3d")
        else:
            figure, axis = plt.subplots()
    else:
        if len(axes) != 1:
            raise ValueError("axes must contain one Fermi-surface axis")
        axis = axes[0]
        figure = axis.figure
    for sheet in result.sheets:
        color = f"C{sheet.band_index % 10}"
        if result.dimension == 1:
            axis.scatter(
                sheet.vertices_reduced[:, result.periodic_axes[0]],
                np.zeros(sheet.vertices_reduced.shape[0]),
                color=color,
                label=f"band {sheet.band_index}",
            )
        elif result.dimension == 2:
            axis_x, axis_y = result.periodic_axes
            for start, stop in sheet.connectivity:
                segment = sheet.vertices_reduced[[start, stop]]
                axis.plot(segment[:, axis_x], segment[:, axis_y], color=color)
            axis.plot([], [], color=color, label=f"band {sheet.band_index}")
        else:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection

            triangles = sheet.vertices_reduced[sheet.connectivity]
            collection = Poly3DCollection(
                triangles, alpha=0.65, facecolor=color, edgecolor="none"
            )
            axis.add_collection3d(collection)
            axis.plot([], [], [], color=color, label=f"band {sheet.band_index}")
    axis.set_xlabel(r"$k_1$ (r.l.u.)")
    if result.dimension >= 2:
        axis.set_ylabel(r"$k_2$ (r.l.u.)")
    if result.dimension == 3:
        axis.set_zlabel(r"$k_3$ (r.l.u.)")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_zlim(0.0, 1.0)
    elif result.dimension == 2:
        axis.set_aspect("equal")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
    if result.sheets:
        axis.legend()
    figure.tight_layout()
    return figure, axis


def _component_model_script(component: Any) -> list[str]:
    config = component.config
    source_path = str(config.get("source_path", "")).strip()
    if source_path:
        raw_axes = config.get("periodic_axes", [])
        lines = [
            "from nfit import import_wannier90",
            f"model = import_wannier90({source_path!r}, periodic_axes={None if not raw_axes else tuple(raw_axes)!r})",
        ]
        expected = str(config.get("model_digest", "")).strip()
        if expected:
            lines.append(f"assert model.content_digest == {expected!r}")
        return lines
    return [
        "from nfit import ElectronicModel",
        f"model = ElectronicModel.from_dict({config.get('model_data', {})!r})",
    ]


def tight_binding_plot_script(component: Any, plot_key: str) -> str:
    """Return an editable script for one configured electronic-structure plot."""

    config = component.config
    lines = _component_model_script(component)
    projections = _projection_groups(component)
    energy_unit = _electronic_display_unit(component)
    chemical_potential = float(
        electronic_energy_from_meV(
            config.get("chemical_potential_meV", 0.0), energy_unit
        )
    )
    execution_backend = str(config.get("electronic_backend", "auto"))
    execution_workers = int(config.get("electronic_workers", 0))
    max_batch_bytes = int(
        float(config.get("electronic_max_batch_mb", 256.0)) * 1024**2
    )
    dos_symmetry = str(config.get("dos_symmetry", "auto"))
    dos_method = str(config.get("dos_method", "gaussian"))
    mesh_symmetry = (
        "full" if dos_method == "tetrahedron" or projections else dos_symmetry
    )
    lines.extend(
        [
            "from nfit import electronic_energy_to_meV",
            f"energy_unit = {energy_unit!r}",
            f"chemical_potential = {chemical_potential!r}",
            "chemical_potential_meV = electronic_energy_to_meV(chemical_potential, energy_unit)",
            f"electronic_backend = {execution_backend!r}",
            f"electronic_workers = {None if execution_workers == 0 else execution_workers!r}",
            f"max_batch_bytes = {max_batch_bytes!r}",
        ]
    )
    if plot_key == "bands":
        raw_nodes = config.get("band_path", [])
        nodes = [item["k"] if isinstance(item, dict) else item for item in raw_nodes]
        labels = [
            str(item.get("label", f"K{index}"))
            if isinstance(item, dict)
            else f"K{index}"
            for index, item in enumerate(raw_nodes)
        ]
        breaks = [
            bool(item.get("break_before", False))
            if isinstance(item, dict)
            else False
            for item in raw_nodes
        ]
        lines.extend(
            [
                "from nfit import band_path, calculate_bands",
                "from nfit.model_plots import ElectronicPlotStyle, render_band_structure",
                "plot_style = ElectronicPlotStyle()",
                "path_reciprocal_lattice = "
                f"{band_path_reciprocal_lattice(component).tolist()!r}",
                f"sampling = band_path(model, {nodes!r}, labels={labels!r}, break_before={breaks!r}, points_per_inv_angstrom={float(config.get('band_points_per_inv_angstrom', 80.0))!r}, coordinate_reciprocal_lattice=path_reciprocal_lattice)",
                f"result = calculate_bands(model, sampling, chemical_potential_meV=chemical_potential_meV, projections={projections!r}, include_eigenvectors=False, backend=electronic_backend, workers=electronic_workers, max_batch_bytes=max_batch_bytes)",
                "figure, axis = render_band_structure(result, energy_unit=energy_unit, style=plot_style)",
            ]
        )
    elif plot_key == "dos":
        energy_min = float(
            electronic_energy_from_meV(
                config.get("dos_energy_min_meV", -500.0), energy_unit
            )
        )
        energy_max = float(
            electronic_energy_from_meV(
                config.get("dos_energy_max_meV", 500.0), energy_unit
            )
        )
        broadening = float(
            electronic_energy_from_meV(
                config.get("dos_broadening_meV", 5.0), energy_unit
            )
        )
        lines.extend(
            [
                "import numpy as np",
                "from nfit import density_of_states, k_mesh",
                "from nfit.model_plots import ElectronicPlotStyle, render_density_of_states",
                "plot_style = ElectronicPlotStyle()",
                f"sampling_certificate = {config.get('dos_sampling_certificate', {})!r}",
                f"mesh = k_mesh(model, {config.get('dos_mesh', [40, 40, 40])!r}, symmetry={mesh_symmetry!r})",
                f"automatic_energy_range = {bool(config.get('dos_auto_energy_range', False))!r}",
                f"energy_points = {int(config.get('dos_energy_points', 600))!r}",
                f"energy = None if automatic_energy_range else np.linspace({energy_min!r}, {energy_max!r}, energy_points)",
                "energy_meV = None if energy is None else electronic_energy_to_meV(energy, energy_unit)",
                f"broadening = {broadening!r}",
                "broadening_meV = electronic_energy_to_meV(broadening, energy_unit)",
                f"result = density_of_states(model, mesh, energy_meV, broadening_meV=broadening_meV, method={dos_method!r}, symmetry={dos_symmetry!r}, energy_points=energy_points, chemical_potential_meV=chemical_potential_meV, projections={projections!r}, backend=electronic_backend, workers=electronic_workers, max_batch_bytes=max_batch_bytes)",
                "figure, axis = render_density_of_states(result, energy_unit=energy_unit, style=plot_style)",
            ]
        )
    elif plot_key == "fermi_surface":
        target = float(
            electronic_energy_from_meV(
                config.get(
                    "fermi_energy_meV",
                    config.get("chemical_potential_meV", 0.0),
                ),
                energy_unit,
            )
        )
        lines.extend(
            [
                "from nfit import fermi_surface, reciprocal_mesh_shape_for_spacing",
                "from nfit.model_plots import render_fermi_surface",
                f"target_energy = {target!r}",
                "target_energy_meV = electronic_energy_to_meV(target_energy, energy_unit)",
                f"fermi_mesh_mode = {config.get('fermi_mesh_mode', 'spacing')!r}",
                f"fermi_spacing_inv_angstrom = {float(config.get('fermi_spacing_inv_angstrom', 0.025))!r}",
                f"fermi_mesh = {config.get('fermi_mesh', [64, 64, 64])!r}",
                "if fermi_mesh_mode == 'spacing':",
                "    fermi_mesh = reciprocal_mesh_shape_for_spacing(model, fermi_spacing_inv_angstrom)",
                f"result = fermi_surface(model, fermi_mesh, target_energy_meV=target_energy_meV, projections={projections!r}, backend=electronic_backend, workers=electronic_workers, max_batch_bytes=max_batch_bytes)",
            ]
        )
    else:
        raise KeyError(f"unknown tight-binding plot {plot_key!r}")
    viewer_keys = {
        "bands": "band_structure",
        "dos": "density_of_states",
        "fermi_surface": "fermi_surface",
    }
    if plot_key == "fermi_surface":
        lines.extend(
            [
                "",
                "if globals().get('__name__') == '__main__':",
                "    from PySide6 import QtWidgets",
                "    app = QtWidgets.QApplication.instance()",
                "    owns_app = app is None",
                "    if owns_app:",
                "        app = QtWidgets.QApplication([])",
                "    if result.dimension == 3:",
                "        from nfit.qt_fermi_surface_viewer import FermiSurfaceViewOptions, show_fermi_surface_result",
                "        view_options = FermiSurfaceViewOptions(",
                "            band_opacity=0.65,",
                "            text_size=14,",
                "            grid_line_width=1.0,",
                "            show_legend=True,",
                "            shading='smooth',",
                "        )",
                "        window = show_fermi_surface_result(result, view_options=view_options)",
                "    else:",
                "        from nfit.qt_electronic_viewer import show_electronic_figure",
                "        figure, axis = render_fermi_surface(result, energy_unit=energy_unit)",
                "        window = show_electronic_figure(",
                "            figure, viewer_key='fermi_surface'",
                "        )",
                "    if owns_app:",
                "        app.exec()",
                "else:",
                "    figure, axis = render_fermi_surface(result, energy_unit=energy_unit)",
                "    figure.show()",
                "",
            ]
        )
        return "\n".join(lines)
    lines.extend(
        [
            "",
            "# Use nfit's standard plot-plus-settings window when this file is run.",
            "if globals().get('__name__') == '__main__':",
            "    from PySide6 import QtWidgets",
            "    from nfit.qt_electronic_viewer import show_electronic_figure",
            "    app = QtWidgets.QApplication.instance()",
            "    owns_app = app is None",
            "    if owns_app:",
            "        app = QtWidgets.QApplication([])",
            "    window = show_electronic_figure(",
            f"        figure, viewer_key={viewer_keys[plot_key]!r},",
            "        style=plot_style,",
            "    )",
            "    if owns_app:",
            "        app.exec()",
            "else:",
            "    figure.show()",
            "",
        ]
    )
    return "\n".join(lines)
