"""Scriptable calculations and renderers owned by registered response models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .electronic_structure import (
    BandResult,
    DensityOfStatesResult,
    ElectronicModel,
    FermiSurfaceResult,
    band_path,
    calculate_bands,
    density_of_states,
    fermi_surface,
    import_wannier90,
    k_mesh,
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


def _electronic_model_from_component(component: Any) -> ElectronicModel:
    config = component.config if isinstance(component.config, dict) else {}
    source_path = str(config.get("source_path", "")).strip()
    if source_path:
        raw_axes = config.get("periodic_axes", [])
        model = import_wannier90(
            source_path,
            periodic_axes=(
                None
                if not raw_axes
                else tuple(int(value) for value in raw_axes)
            ),
        )
    else:
        payload = config.get("model_data")
        if not isinstance(payload, dict) or not payload:
            raise ValueError(
                "tight-binding model has no source; import Wannier90 data or "
                "supply a canonical model_data dictionary"
            )
        model = ElectronicModel.from_dict(payload)
    expected = str(config.get("model_digest", "")).strip()
    if expected and expected != model.content_digest:
        raise ValueError("tight-binding source no longer matches its stored digest")
    return model


def _projection_groups(component: Any) -> dict[str, list[int]]:
    config = component.config if isinstance(component.config, dict) else {}
    raw = config.get("projection_groups", {})
    if not isinstance(raw, dict):
        raise ValueError("projection_groups must be a dictionary of basis-index lists")
    return {
        str(label): [int(index) for index in indices]
        for label, indices in raw.items()
    }


def tight_binding_band_structure(component: Any) -> BandResult:
    """Calculate a configured tight-binding band path for a model component."""

    model = _electronic_model_from_component(component)
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
    sampling = band_path(
        model,
        nodes,
        labels=labels,
        points_per_segment=int(config.get("band_points_per_segment", 60)),
    )
    return calculate_bands(
        model,
        sampling,
        chemical_potential_meV=float(config.get("chemical_potential_meV", 0.0)),
        projections=_projection_groups(component),
        include_eigenvectors=False,
    )


def tight_binding_density_of_states(component: Any) -> DensityOfStatesResult:
    """Calculate configured total and orbital-projected density of states."""

    model = _electronic_model_from_component(component)
    config = component.config
    mesh = k_mesh(model, config.get("dos_mesh", [40, 40, 40]))
    energy_min = float(config.get("dos_energy_min_meV", -500.0))
    energy_max = float(config.get("dos_energy_max_meV", 500.0))
    energy = np.linspace(
        energy_min,
        energy_max,
        int(config.get("dos_energy_points", 600)),
    )
    return density_of_states(
        model,
        mesh,
        energy,
        broadening_meV=float(config.get("dos_broadening_meV", 5.0)),
        chemical_potential_meV=float(config.get("chemical_potential_meV", 0.0)),
        projections=_projection_groups(component),
    )


def tight_binding_fermi_surface(component: Any) -> FermiSurfaceResult:
    """Calculate configured Fermi points, contours, or surfaces."""

    model = _electronic_model_from_component(component)
    config = component.config
    return fermi_surface(
        model,
        config.get("fermi_mesh", [100, 100, 40]),
        target_energy_meV=float(
            config.get(
                "fermi_energy_meV",
                config.get("chemical_potential_meV", 0.0),
            )
        ),
        projections=_projection_groups(component),
    )


def render_band_structure(
    result: BandResult, *, axes: Sequence[Any] | None = None
) -> tuple[Any, Any]:
    """Render total and orbital-projected bands along a path."""

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
    relative = result.energies_meV - result.chemical_potential_meV
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
    axis.set_ylabel(r"$\epsilon-\mu$ (meV)")
    axis.set_xlabel("Wavevector path")
    if result.projected_weights:
        axis.legend()
    figure.tight_layout()
    return figure, axis


def render_density_of_states(
    result: DensityOfStatesResult, *, axes: Sequence[Any] | None = None
) -> tuple[Any, Any]:
    """Render total and projected electronic density of states."""

    import matplotlib.pyplot as plt

    if axes is None:
        figure, axis = plt.subplots()
    else:
        if len(axes) != 1:
            raise ValueError("axes must contain one density-of-states axis")
        axis = axes[0]
        figure = axis.figure
    relative = result.energy_meV - result.chemical_potential_meV
    axis.plot(relative, result.total_per_meV_cell, color="black", label="total")
    for label, values in result.projected_per_meV_cell.items():
        axis.plot(relative, values, label=label)
    axis.axvline(0.0, color="0.4", linewidth=0.8, linestyle="--")
    axis.set_xlabel(r"$\epsilon-\mu$ (meV)")
    axis.set_ylabel("DOS (states / meV / cell)")
    axis.legend()
    figure.tight_layout()
    return figure, axis


def render_fermi_surface(
    result: FermiSurfaceResult, *, axes: Sequence[Any] | None = None
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
    chemical_potential = float(config.get("chemical_potential_meV", 0.0))
    if plot_key == "bands":
        raw_nodes = config.get("band_path", [])
        nodes = [item["k"] if isinstance(item, dict) else item for item in raw_nodes]
        labels = [
            str(item.get("label", f"K{index}"))
            if isinstance(item, dict)
            else f"K{index}"
            for index, item in enumerate(raw_nodes)
        ]
        lines.extend(
            [
                "from nfit import band_path, calculate_bands",
                "from nfit.model_plots import render_band_structure",
                f"sampling = band_path(model, {nodes!r}, labels={labels!r}, points_per_segment={int(config.get('band_points_per_segment', 60))!r})",
                f"result = calculate_bands(model, sampling, chemical_potential_meV={chemical_potential!r}, projections={projections!r}, include_eigenvectors=False)",
                "figure, axis = render_band_structure(result)",
            ]
        )
    elif plot_key == "dos":
        lines.extend(
            [
                "import numpy as np",
                "from nfit import density_of_states, k_mesh",
                "from nfit.model_plots import render_density_of_states",
                f"mesh = k_mesh(model, {config.get('dos_mesh', [40, 40, 40])!r})",
                f"energy = np.linspace({float(config.get('dos_energy_min_meV', -500.0))!r}, {float(config.get('dos_energy_max_meV', 500.0))!r}, {int(config.get('dos_energy_points', 600))!r})",
                f"result = density_of_states(model, mesh, energy, broadening_meV={float(config.get('dos_broadening_meV', 5.0))!r}, chemical_potential_meV={chemical_potential!r}, projections={projections!r})",
                "figure, axis = render_density_of_states(result)",
            ]
        )
    elif plot_key == "fermi_surface":
        lines.extend(
            [
                "from nfit import fermi_surface",
                "from nfit.model_plots import render_fermi_surface",
                f"result = fermi_surface(model, {config.get('fermi_mesh', [100, 100, 40])!r}, target_energy_meV={float(config.get('fermi_energy_meV', chemical_potential))!r}, projections={projections!r})",
                "figure, axis = render_fermi_surface(result)",
            ]
        )
    else:
        raise KeyError(f"unknown tight-binding plot {plot_key!r}")
    lines.extend(["figure.show()", ""])
    return "\n".join(lines)
