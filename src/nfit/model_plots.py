"""Scriptable calculations and renderers owned by registered response models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
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
)
from .spin_fluctuations import generalized_paramagnon_susceptibility


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
    source = components.get(source_name)
    if source is None or getattr(source, "type", None) != "tight_binding":
        raise ValueError(
            f"{component.name!r} must reference an enabled tight-binding component"
        )
    return source


def lindhard_energy_scan(
    component: Any,
    components: Mapping[str, Any],
) -> Any:
    """Calculate the configured complex bare spin-response energy scan."""

    from .electronic_response import (
        bare_spin_susceptibility,
        chemical_potential_for_filling,
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
        np.asarray(config.get("plot_q_reduced", [0.0, 0.0, 0.0]), dtype=float),
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
        "backend": str(config.get("response_backend", "numpy")),
        "workers": int(config.get("response_workers", 1)),
        "max_batch_bytes": int(
            float(config.get("response_max_batch_mb", 256.0)) * 1024**2
        ),
    }
    if str(config.get("chemical_potential_mode", "source")) == "filling":
        mu = chemical_potential_for_filling(
            model,
            mesh,
            float(config.get("filling_per_cell", 1.0)),
            temperature_K=temperature,
            **execution,
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
    real_axis.set_title(
        rf"{response_label} at $Q=({Q[0]:g},{Q[1]:g},{Q[2]:g})$ r.l.u."
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
            "    k_mesh,",
            ")",
            "from nfit.model_plots import render_lindhard_energy_scan",
            "",
            f"mesh_shape = {config.get('response_mesh', [16, 16, 16])!r}",
            f"mesh_shift = {config.get('response_mesh_shift', [0.0, 0.0, 0.0])!r}",
            "mesh = k_mesh(model, mesh_shape, shift=mesh_shift, symmetry='full')",
            f"temperature_K = {float(config.get('plot_temperature_K', 10.0))!r}",
            f"energy_meV = np.linspace({float(config.get('plot_energy_min_meV', -100.0))!r}, {float(config.get('plot_energy_max_meV', 100.0))!r}, {int(config.get('plot_energy_points', 401))!r})",
            f"Q_reduced = np.broadcast_to(np.asarray({config.get('plot_q_reduced', [0.0, 0.0, 0.0])!r}, dtype=float), (energy_meV.size, 3))",
            f"backend = {str(config.get('response_backend', 'numpy'))!r}",
            f"workers = {int(config.get('response_workers', 1))!r}",
            f"max_batch_bytes = {int(float(config.get('response_max_batch_mb', 256.0)) * 1024**2)!r}",
        ]
    )
    if str(config.get("chemical_potential_mode", "source")) == "filling":
        lines.extend(
            [
                "chemical_potential_meV = chemical_potential_for_filling(",
                f"    model, mesh, {float(config.get('filling_per_cell', 1.0))!r},",
                "    temperature_K=temperature_K, backend=backend,",
                "    workers=workers, max_batch_bytes=max_batch_bytes,",
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
            ")",
            "figure, axes = render_lindhard_energy_scan(result)",
            "figure.show()",
            "",
        ]
    )
    return "\n".join(lines)


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
        np.asarray(config.get("plot_q_reduced", [0.0, 0.0, 0.0]), dtype=float),
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
        "backend": str(config.get("response_backend", "numpy")),
        "workers": int(config.get("response_workers", 1)),
        "max_batch_bytes": int(
            float(config.get("response_max_batch_mb", 256.0)) * 1024**2
        ),
    }
    if str(config.get("chemical_potential_mode", "source")) == "filling":
        mu = chemical_potential_for_filling(
            model,
            mesh,
            float(config.get("filling_per_cell", 1.0)),
            temperature_K=temperature,
            **execution,
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
        points_per_segment=int(config.get("band_points_per_segment", 60)),
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
    symmetry = str(config.get("dos_symmetry", "full"))
    if projections and symmetry == "reduced":
        raise ValueError(
            "projected DOS cannot require symmetry reduction because an "
            "arbitrary orbital projector need not be symmetry invariant"
        )
    mesh = k_mesh(
        model,
        config.get("dos_mesh", [40, 40, 40]),
        symmetry="full" if projections else symmetry,
    )
    energy_min = float(config.get("dos_energy_min_meV", -500.0))
    energy_max = float(config.get("dos_energy_max_meV", 500.0))
    energy = np.linspace(
        energy_min,
        energy_max,
        int(config.get("dos_energy_points", 600)),
    )
    return _with_electronic_display_unit(
        density_of_states(
            model,
            mesh,
            energy,
            broadening_meV=float(config.get("dos_broadening_meV", 5.0)),
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
    return _with_electronic_display_unit(
        fermi_surface(
            model,
            config.get("fermi_mesh", [64, 64, 64]),
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
    axis.plot(distance, relative, color="0.55", linewidth=1.0, zorder=1)
    for group_index, (label, weights) in enumerate(result.projected_weights.items()):
        color = f"C{group_index % 10}"
        for band_index in range(relative.shape[1]):
            axis.scatter(
                distance,
                relative[:, band_index],
                s=4.0 + 24.0 * weights[:, band_index],
                color=color,
                alpha=0.55,
                linewidths=0,
                zorder=2,
            )
        axis.plot([], [], marker="o", linestyle="", color=color, label=label)
    tick_indices, tick_labels = zip(*result.sampling.labels, strict=True)
    tick_positions = distance[np.asarray(tick_indices, dtype=int)]
    axis.set_xticks(tick_positions, tick_labels)
    for position in tick_positions:
        axis.axvline(position, color="0.85", linewidth=0.8, zorder=0)
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xlim(distance[0], distance[-1])
    axis.set_ylabel(rf"$\epsilon-\mu$ ({unit})")
    axis.set_xlabel("Wavevector path")
    if result.projected_weights:
        axis.legend()
    figure.tight_layout()
    return figure, axis


def render_density_of_states(
    result: DensityOfStatesResult,
    *,
    energy_unit: str | None = None,
    axes: Sequence[Any] | None = None,
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
    axis.plot(
        relative,
        result.total_per_meV_cell * density_scale,
        color="black",
        label="total",
    )
    for label, values in result.projected_per_meV_cell.items():
        axis.plot(relative, values * density_scale, label=label)
    axis.axvline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    axis.set_xlabel(rf"$\epsilon-\mu$ ({unit})")
    axis.set_ylabel(f"DOS (states / {unit} / cell)")
    axis.legend()
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
    unit = normalize_electronic_energy_unit(
        energy_unit or result.provenance.get("display_energy_unit", "eV")
    )
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
    target = float(electronic_energy_from_meV(result.target_energy_meV, unit))
    axis.set_title(f"Constant-energy surface at E = {target:g} {unit}")
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
    dos_symmetry = str(config.get("dos_symmetry", "full"))
    if projections and dos_symmetry == "reduced":
        raise ValueError(
            "projected DOS cannot require symmetry reduction because an "
            "arbitrary orbital projector need not be symmetry invariant"
        )
    if projections:
        dos_symmetry = "full"
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
                "from nfit.model_plots import render_band_structure",
                "path_reciprocal_lattice = "
                f"{band_path_reciprocal_lattice(component).tolist()!r}",
                f"sampling = band_path(model, {nodes!r}, labels={labels!r}, break_before={breaks!r}, points_per_segment={int(config.get('band_points_per_segment', 60))!r}, coordinate_reciprocal_lattice=path_reciprocal_lattice)",
                f"result = calculate_bands(model, sampling, chemical_potential_meV=chemical_potential_meV, projections={projections!r}, include_eigenvectors=False, backend=electronic_backend, workers=electronic_workers, max_batch_bytes=max_batch_bytes)",
                "figure, axis = render_band_structure(result, energy_unit=energy_unit)",
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
                "from nfit.model_plots import render_density_of_states",
                f"mesh = k_mesh(model, {config.get('dos_mesh', [40, 40, 40])!r}, symmetry={dos_symmetry!r})",
                f"energy = np.linspace({energy_min!r}, {energy_max!r}, {int(config.get('dos_energy_points', 600))!r})",
                "energy_meV = electronic_energy_to_meV(energy, energy_unit)",
                f"broadening = {broadening!r}",
                "broadening_meV = electronic_energy_to_meV(broadening, energy_unit)",
                f"result = density_of_states(model, mesh, energy_meV, broadening_meV=broadening_meV, chemical_potential_meV=chemical_potential_meV, projections={projections!r}, backend=electronic_backend, workers=electronic_workers, max_batch_bytes=max_batch_bytes)",
                "figure, axis = render_density_of_states(result, energy_unit=energy_unit)",
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
                "from nfit import fermi_surface",
                "from nfit.model_plots import render_fermi_surface",
                f"target_energy = {target!r}",
                "target_energy_meV = electronic_energy_to_meV(target_energy, energy_unit)",
                f"result = fermi_surface(model, {config.get('fermi_mesh', [64, 64, 64])!r}, target_energy_meV=target_energy_meV, projections={projections!r}, backend=electronic_backend, workers=electronic_workers, max_batch_bytes=max_batch_bytes)",
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
                "        from nfit.qt_fermi_surface_viewer import show_fermi_surface_result",
                "        window = show_fermi_surface_result(result)",
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
            f"        figure, viewer_key={viewer_keys[plot_key]!r}",
            "    )",
            "    if owns_app:",
            "        app.exec()",
            "else:",
            "    figure.show()",
            "",
        ]
    )
    return "\n".join(lines)
