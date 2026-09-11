"""Non-destructive momentum-path extraction and visualization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .analysis.coordinates import physical_axis_vectors, rlu_to_q_matrix
from .mdhisto import MDHistoAxis, MDHistoData


@dataclass(frozen=True)
class KPathNode:
    """A labelled absolute conventional-cell HKL point on a momentum path."""

    label: str
    hkl: tuple[float, float, float]
    break_before: bool = False


@dataclass(frozen=True)
class KPathSampling:
    """Interpolated path coordinates and label locations."""

    nodes: tuple[KPathNode, ...]
    points_hkl: NDArray[np.float64]
    distance_inv_angstrom: NDArray[np.float64]
    node_indices: tuple[int, ...]


def coerce_kpath_nodes(nodes: Sequence[KPathNode | Mapping[str, Any]]) -> tuple[KPathNode, ...]:
    """Validate the public node representation without reducing higher-zone HKLs."""

    result = []
    for index, item in enumerate(nodes):
        if isinstance(item, KPathNode):
            node = item
        else:
            coordinate = item.get("hkl", item.get("k", ()))
            values = np.asarray(coordinate, dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError("every k-path node requires three finite HKL values")
            node = KPathNode(
                str(item.get("label", f"K{index}")),
                tuple(float(value) for value in values),
                bool(item.get("break_before", False)),
            )
        if index == 0 and node.break_before:
            raise ValueError("the first k-path node cannot begin a disconnected section")
        result.append(node)
    if len(result) < 2:
        raise ValueError("a k path requires at least two nodes")
    return tuple(result)


def sample_kpath(
    nodes: Sequence[KPathNode | Mapping[str, Any]],
    q_matrix: ArrayLike,
    *,
    step_inv_angstrom: float = 0.03,
) -> KPathSampling:
    """Interpolate connected HKL nodes at approximately uniform physical spacing."""

    path_nodes = coerce_kpath_nodes(nodes)
    matrix = np.asarray(q_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("q_matrix must be a finite 3x3 matrix")
    step = float(step_inv_angstrom)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_inv_angstrom must be finite and positive")
    node_hkl = np.asarray([node.hkl for node in path_nodes], dtype=float)
    node_q = node_hkl @ matrix.T
    points = [node_hkl[0]]
    distances = [0.0]
    node_indices = [0]
    typical_step = step
    for index in range(len(path_nodes) - 1):
        if path_nodes[index + 1].break_before:
            points.append(node_hkl[index + 1])
            distances.append(distances[-1] + typical_step)
            node_indices.append(len(points) - 1)
            continue
        length = float(np.linalg.norm(node_q[index + 1] - node_q[index]))
        intervals = max(1, int(np.ceil(length / step)))
        interval = length / intervals
        typical_step = interval if interval > 0.0 else typical_step
        for part in range(1, intervals + 1):
            fraction = part / intervals
            points.append(
                (1.0 - fraction) * node_hkl[index]
                + fraction * node_hkl[index + 1]
            )
            distances.append(distances[-1] + interval)
        node_indices.append(len(points) - 1)
    return KPathSampling(
        path_nodes,
        np.asarray(points, dtype=float),
        np.asarray(distances, dtype=float),
        tuple(node_indices),
    )


def standard_kpath(
    lattice_parameters: Mapping[str, float],
    spacegroup: str,
    *,
    symprec: float = 1.0e-5,
) -> tuple[KPathNode, ...]:
    """Return ASE's standard Bravais-lattice path in conventional HKL coordinates."""

    from .brillouin_zone import standard_band_path
    from .crystal import lattice_vectors, primitive_lattice_vectors

    lattice = {key: float(value) for key, value in lattice_parameters.items()}
    conventional = lattice_vectors(lattice)
    primitive = primitive_lattice_vectors(lattice, str(spacegroup))
    primitive_reciprocal = np.linalg.inv(primitive).T
    conventional_reciprocal = np.linalg.inv(conventional).T
    primitive_path = standard_band_path(
        {"lattice": lattice, "spacegroup": str(spacegroup), "sites": []},
        convention="setyawan_curtarolo",
        symprec=float(symprec),
    )
    result = []
    for node in primitive_path:
        conventional_hkl = np.linalg.solve(
            conventional_reciprocal,
            primitive_reciprocal @ np.asarray(node["k"], dtype=float),
        )
        result.append(
            KPathNode(
                str(node["label"]),
                tuple(float(value) for value in np.round(conventional_hkl, 12)),
                break_before=bool(node.get("break_before", False)),
            )
        )
    if len(result) < 2:
        raise ValueError("could not construct a standard path for this lattice")
    return tuple(result)


def _edges_from_centers(centers: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(centers) == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5])
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def prepare_mdhisto_kpath(
    data: MDHistoData,
    nodes: Sequence[KPathNode | Mapping[str, Any]],
    *,
    step_inv_angstrom: float = 0.03,
    transverse_width_inv_angstrom: float = 0.08,
    minimum_voxels: int = 1,
    lattice_parameters: Mapping[str, float] | None = None,
) -> MDHistoData:
    """Average a 4D HKL-energy histogram inside a tube around a path.

    Source momentum voxels are assigned to their nearest sampled path point in
    physical reciprocal space. Accepted values are combined with inverse-
    variance weights and uncertainties are propagated as ``1/sqrt(sum(1/sigma²))``.
    """

    if not isinstance(data, MDHistoData):
        raise TypeError("k-path extraction requires an MDHistoData object")
    momentum_dims = [index for index, axis in enumerate(data.axes) if axis.kind == "momentum"]
    energy_dims = [
        index for index, axis in enumerate(data.axes)
        if axis.kind == "energy" or axis.role == "energy_transfer"
    ]
    if len(momentum_dims) != 3 or len(energy_dims) != 1 or len(data.axes) != 4:
        raise ValueError("k-path extraction currently requires three HKL axes and one energy axis")
    vectors = physical_axis_vectors(data)[momentum_dims, :3]
    if np.linalg.matrix_rank(vectors) < 3:
        raise ValueError("the three momentum axes must span HKL space")
    metadata = dict(data.metadata)
    if lattice_parameters is not None:
        metadata["lattice_parameters"] = dict(lattice_parameters)
    q_matrix = rlu_to_q_matrix(metadata)
    sampling = sample_kpath(nodes, q_matrix, step_inv_angstrom=step_inv_angstrom)
    width = float(transverse_width_inv_angstrom)
    required = int(minimum_voxels)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("transverse_width_inv_angstrom must be finite and positive")
    if required < 1:
        raise ValueError("minimum_voxels must be positive")

    axis_coordinates = np.meshgrid(
        *(data.axes[dim].centers for dim in momentum_dims), indexing="ij"
    )
    hkl = np.zeros((*axis_coordinates[0].shape, 3), dtype=float)
    for values, vector in zip(axis_coordinates, vectors, strict=True):
        hkl += values[..., None] * vector
    source_q = hkl.reshape(-1, 3) @ q_matrix.T
    path_q = sampling.points_hkl @ q_matrix.T
    node_q = np.asarray([node.hkl for node in sampling.nodes]) @ q_matrix.T
    node_distance = sampling.distance_inv_angstrom[
        np.asarray(sampling.node_indices, dtype=int)
    ]
    distances = np.full(len(source_q), np.inf, dtype=float)
    projected_distance = np.zeros(len(source_q), dtype=float)
    for index in range(len(sampling.nodes) - 1):
        if sampling.nodes[index + 1].break_before:
            continue
        direction = node_q[index + 1] - node_q[index]
        length_sq = float(direction @ direction)
        if length_sq <= np.finfo(float).eps:
            continue
        fraction = np.clip(
            ((source_q - node_q[index]) @ direction) / length_sq,
            0.0,
            1.0,
        )
        separation = source_q - (
            node_q[index] + fraction[:, None] * direction
        )
        candidate = np.linalg.norm(separation, axis=1)
        better = candidate < distances
        distances[better] = candidate[better]
        projected_distance[better] = (
            node_distance[index]
            + fraction[better]
            * (node_distance[index + 1] - node_distance[index])
        )
    right = np.searchsorted(
        sampling.distance_inv_angstrom, projected_distance, side="left"
    )
    right = np.clip(right, 0, len(path_q) - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(projected_distance - sampling.distance_inv_angstrom[left]) <= np.abs(
        sampling.distance_inv_angstrom[right] - projected_distance
    )
    assignments = np.where(use_left, left, right)
    accepted_q = distances <= width
    order = (*momentum_dims, energy_dims[0])
    signal = np.transpose(np.asarray(data.signal, dtype=float), order).reshape(len(source_q), -1)
    errors = np.transpose(np.asarray(data.errors, dtype=float), order).reshape(len(source_q), -1)
    mask = np.transpose(np.asarray(data.mask, dtype=bool), order).reshape(len(source_q), -1)
    output_signal = np.full((len(path_q), signal.shape[1]), np.nan, dtype=float)
    output_errors = np.full_like(output_signal, np.nan)
    output_events = np.zeros_like(output_signal)
    output_mask = np.ones_like(output_signal, dtype=bool)
    for energy_index in range(signal.shape[1]):
        valid = (
            accepted_q & ~mask[:, energy_index] & np.isfinite(signal[:, energy_index])
            & np.isfinite(errors[:, energy_index]) & (errors[:, energy_index] > 0.0)
        )
        bins = assignments[valid]
        weights = 1.0 / np.square(errors[valid, energy_index])
        weight_sum = np.bincount(bins, weights=weights, minlength=len(path_q))
        weighted_signal = np.bincount(
            bins, weights=weights * signal[valid, energy_index], minlength=len(path_q)
        )
        counts = np.bincount(bins, minlength=len(path_q))
        usable = (weight_sum > 0.0) & (counts >= required)
        output_signal[usable, energy_index] = weighted_signal[usable] / weight_sum[usable]
        output_errors[usable, energy_index] = 1.0 / np.sqrt(weight_sum[usable])
        output_events[:, energy_index] = counts
        output_mask[usable, energy_index] = False

    energy_axis = data.axes[energy_dims[0]]
    path_metadata = {
        "nodes": [
            {"label": node.label, "hkl": list(node.hkl), "break_before": node.break_before}
            for node in sampling.nodes
        ],
        "node_indices": list(sampling.node_indices),
        "node_distances_inv_angstrom": [
            float(sampling.distance_inv_angstrom[index]) for index in sampling.node_indices
        ],
        "points_hkl": sampling.points_hkl.tolist(),
        "step_inv_angstrom": float(step_inv_angstrom),
        "transverse_width_inv_angstrom": width,
        "minimum_voxels": required,
        "q_matrix": q_matrix.tolist(),
    }
    return MDHistoData(
        axes=(
            MDHistoAxis(
                "k-path distance",
                _edges_from_centers(sampling.distance_inv_angstrom),
                "angstrom^-1",
                "momentum",
                metadata={"kpath": path_metadata},
            ),
            energy_axis,
        ),
        signal=output_signal,
        errors=output_errors,
        mask=output_mask,
        num_events=output_events,
        metadata={**metadata, "analysis_kind": "kpath_visualization", "kpath": path_metadata},
    )


def format_hkl(hkl: Sequence[float]) -> str:
    """Return a compact crystallographic coordinate label."""

    return "(" + " ".join(f"{float(value):g}" for value in hkl) + ")"


def plot_mdhisto_kpath(
    data: MDHistoData,
    nodes: Sequence[KPathNode | Mapping[str, Any]] | None = None,
    *,
    step_inv_angstrom: float = 0.03,
    transverse_width_inv_angstrom: float = 0.08,
    minimum_voxels: int = 1,
    lattice_parameters: Mapping[str, float] | None = None,
    label_mode: str = "names",
    show_guides: bool = True,
    guide_color: str = "white",
    guide_linewidth: float = 1.2,
    guide_alpha: float = 0.9,
    cmap: str = "viridis",
    color_scale: str = "linear",
    color_alpha: float = 0.0,
    figsize: tuple[float, float] = (10.0, 6.0),
    ax: Any | None = None,
):
    """Plot an experimental intensity map along an absolute-HKL momentum path."""

    import matplotlib.pyplot as plt

    from .plotting_core import MDHistoSliceViewer

    prepared = (
        prepare_mdhisto_kpath(
            data,
            nodes or (),
            step_inv_angstrom=step_inv_angstrom,
            transverse_width_inv_angstrom=transverse_width_inv_angstrom,
            minimum_voxels=minimum_voxels,
            lattice_parameters=lattice_parameters,
        )
        if nodes is not None
        else data
    )
    if prepared.metadata.get("analysis_kind") != "kpath_visualization":
        raise ValueError("nodes are required unless data is a prepared k-path dataset")
    if label_mode not in {"names", "coordinates", "both"}:
        raise ValueError("label_mode must be names, coordinates, or both")
    model = MDHistoSliceViewer(
        prepared, x_dim=0, y_dim=1, cmap=cmap, color_scale=color_scale,
        color_alpha=color_alpha,
    )
    view = model.slice_arrays()
    values = model._display_values(view)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure
    artist = ax.pcolormesh(
        view["x_edges"], view["y_edges"], values,
        shading="auto", cmap=model._display_cmap(), norm=model._color_norm(values),
    )
    path = prepared.metadata["kpath"]
    nodes_payload = path["nodes"]
    positions = np.asarray(path["node_distances_inv_angstrom"], dtype=float)
    labels = []
    for node in nodes_payload:
        name = str(node.get("label", ""))
        coordinate = format_hkl(node["hkl"])
        if label_mode == "names":
            labels.append(name)
        elif label_mode == "coordinates":
            labels.append(coordinate)
        else:
            labels.append(f"{name}\n{coordinate}")
    ax.set_xticks(positions, labels)
    if show_guides:
        for position in positions:
            ax.axvline(
                position, color=guide_color, linestyle="--",
                linewidth=float(guide_linewidth), alpha=float(guide_alpha), zorder=4,
            )
    ax.set_xlim(float(view["x_edges"][0]), float(view["x_edges"][-1]))
    ax.set_xlabel("Momentum path")
    ax.set_ylabel(model._axis_label(1))
    colorbar = fig.colorbar(artist, ax=ax)
    colorbar.set_label(model._channel_label())
    fig._nfit_kpath_data = prepared
    fig._nfit_kpath_axis = ax
    return fig
