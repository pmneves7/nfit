"""Renderer-independent first-Brillouin-zone geometry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pprint import pformat
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BrillouinZoneSlice:
    """Line segments where reciprocal-lattice Voronoi cells meet a 2D view."""

    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


@dataclass(frozen=True)
class BrillouinZoneNode:
    """One labelled wavevector on a configured band path."""

    label: str
    reduced: tuple[float, float, float]
    cartesian_inv_angstrom: tuple[float, float, float]
    break_before: bool = False


@dataclass(frozen=True)
class BrillouinZoneScene:
    """First-zone polyhedron, reciprocal basis, and labelled path."""

    reciprocal_vectors: tuple[tuple[float, float, float], ...]
    vertices_inv_angstrom: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    path_nodes: tuple[BrillouinZoneNode, ...]


@dataclass(frozen=True)
class BrillouinZoneViewOptions:
    """Scriptable presentation settings for a Brillouin-zone scene."""

    basis_vector_color_mode: str = "single"
    basis_vector_color: str = "#496A9B"
    basis_vector_thickness: float = 0.006
    basis_vector_inside_style: str = "solid"
    path_color: str = "#7A1F1F"
    path_thickness: float = 4.0
    path_point_size: float = 16.0
    label_font_size: int = 18
    label_bold: bool = False
    cell_surface_color: str = "#B8C7D9"
    cell_surface_opacity: float = 0.10
    cell_outline_color: str = "#202020"
    cell_outline_thickness: float = 4.0
    show_basis_vectors: bool = True
    show_path: bool = True
    show_path_labels: bool = True
    show_compass: bool = False
    projection: str = "orthographic"

    def __post_init__(self) -> None:
        if self.basis_vector_color_mode not in {"single", "rgb"}:
            raise ValueError("basis_vector_color_mode must be 'single' or 'rgb'")
        if self.basis_vector_inside_style not in {"solid", "dashed", "hidden"}:
            raise ValueError(
                "basis_vector_inside_style must be 'solid', 'dashed', or 'hidden'"
            )
        if self.projection not in {"orthographic", "perspective"}:
            raise ValueError("projection must be 'orthographic' or 'perspective'")
        for name in (
            "basis_vector_color",
            "path_color",
            "cell_surface_color",
            "cell_outline_color",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be a nonempty color specification")
        if not 0.0 <= float(self.cell_surface_opacity) <= 1.0:
            raise ValueError("cell_surface_opacity must be between zero and one")
        if float(self.basis_vector_thickness) <= 0.0:
            raise ValueError("basis_vector_thickness must be positive")
        if float(self.path_thickness) <= 0.0:
            raise ValueError("path_thickness must be positive")
        if float(self.path_point_size) <= 0.0:
            raise ValueError("path_point_size must be positive")
        if float(self.cell_outline_thickness) <= 0.0:
            raise ValueError("cell_outline_thickness must be positive")
        if int(self.label_font_size) <= 0:
            raise ValueError("label_font_size must be positive")


def _first_zone_faces(
    reciprocal_lattice: FloatArray,
) -> tuple[FloatArray, tuple[tuple[int, ...], ...]]:
    from scipy.spatial import Voronoi

    for shell in range(2, 6):
        indices = np.asarray(
            [
                (i, j, k)
                for i in range(-shell, shell + 1)
                for j in range(-shell, shell + 1)
                for k in range(-shell, shell + 1)
            ],
            dtype=float,
        )
        points = indices @ reciprocal_lattice.T
        origin = int(np.flatnonzero(np.all(indices == 0.0, axis=1))[0])
        voronoi = Voronoi(points)
        region = voronoi.regions[voronoi.point_region[origin]]
        if not region or -1 in region:
            continue
        region_set = set(region)
        raw_faces = []
        for pair, ridge in zip(
            voronoi.ridge_points,
            voronoi.ridge_vertices,
            strict=True,
        ):
            if origin not in pair or -1 in ridge:
                continue
            face = tuple(int(value) for value in ridge if value in region_set)
            if len(face) >= 3:
                raw_faces.append(face)
        used = sorted({index for face in raw_faces for index in face})
        remap = {old: new for new, old in enumerate(used)}
        vertices = np.asarray([voronoi.vertices[index] for index in used])
        faces = tuple(tuple(remap[index] for index in face) for face in raw_faces)
        if len(vertices) >= 4 and faces:
            return vertices, faces
    raise ValueError("could not construct a bounded first Brillouin zone")


@lru_cache(maxsize=32)
def _cached_first_zone_faces(
    reciprocal_values: tuple[float, ...],
) -> tuple[FloatArray, tuple[tuple[int, ...], ...]]:
    reciprocal = np.asarray(reciprocal_values, dtype=float).reshape(3, 3)
    vertices, faces = _first_zone_faces(reciprocal)
    vertices.setflags(write=False)
    return vertices, faces


def _clip_polygon_half_plane(
    polygon: FloatArray, normal: FloatArray, bound: float, *, tolerance: float
) -> FloatArray:
    """Clip a two-dimensional polygon to ``normal @ point <= bound``."""

    if len(polygon) == 0:
        return polygon
    output: list[FloatArray] = []
    previous = polygon[-1]
    previous_value = float(normal @ previous - bound)
    for current in polygon:
        current_value = float(normal @ current - bound)
        previous_inside = previous_value <= tolerance
        current_inside = current_value <= tolerance
        if previous_inside != current_inside:
            fraction = previous_value / (previous_value - current_value)
            output.append(previous + fraction * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_value = current_value
    return np.asarray(output, dtype=float).reshape((-1, 2))


def build_brillouin_zone_slice(
    *,
    hkl_origin: ArrayLike,
    x_hkl_vector: ArrayLike,
    y_hkl_vector: ArrayLike,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    q_matrix: ArrayLike,
    spacegroup: str,
) -> BrillouinZoneSlice:
    """Intersect repeated Brillouin zones with an affine HKL plotting plane.

    The reciprocal lattice is selected from the conventional-cell centering in
    ``spacegroup``.  ``q_matrix`` supplies the reciprocal metric, so oblique and
    non-cubic cells are drawn in the actual plotted HKL coordinates.
    """

    from .analysis.zones import generate_zone_centers, reciprocal_basis_hkl

    origin = np.asarray(hkl_origin, dtype=float)
    x_vector = np.asarray(x_hkl_vector, dtype=float)
    y_vector = np.asarray(y_hkl_vector, dtype=float)
    metric = np.asarray(q_matrix, dtype=float)
    if any(value.shape != (3,) for value in (origin, x_vector, y_vector)):
        raise ValueError("the HKL origin and plotting vectors must contain three values")
    if metric.shape != (3, 3) or not np.all(np.isfinite(metric)):
        raise ValueError("q_matrix must be a finite 3x3 matrix")
    limits = np.asarray((*xlim, *ylim), dtype=float)
    if not np.all(np.isfinite(limits)) or xlim[0] == xlim[1] or ylim[0] == ylim[1]:
        raise ValueError("xlim and ylim must be finite, nonempty intervals")
    plane = metric @ np.column_stack((x_vector, y_vector))
    if np.linalg.matrix_rank(plane, tol=1.0e-12) < 2:
        raise ValueError("Brillouin-zone boundaries require two independent HKL axes")

    basis_hkl = reciprocal_basis_hkl(spacegroup, metric)
    reciprocal = metric @ basis_hkl.T
    vertices, _faces = _cached_first_zone_faces(
        tuple(float(value) for value in reciprocal.ravel())
    )
    zone_radius = float(np.max(np.linalg.norm(vertices, axis=1)))
    corners_xy = np.asarray(
        [(x, y) for x in xlim for y in ylim], dtype=float
    )
    corners_hkl = origin + corners_xy @ np.vstack((x_vector, y_vector))
    centers_hkl = generate_zone_centers(
        basis_hkl, corners_hkl.min(axis=0), corners_hkl.max(axis=0), shell=2
    )
    centers_q = centers_hkl @ metric.T
    origin_q = metric @ origin
    viewport = np.asarray(
        [(xlim[0], ylim[0]), (xlim[1], ylim[0]),
         (xlim[1], ylim[1]), (xlim[0], ylim[1])],
        dtype=float,
    )
    scale = max(float(np.ptp(limits)), 1.0)
    tolerance = 1.0e-9 * scale
    unique: dict[tuple[float, ...], tuple[tuple[float, float], tuple[float, float]]] = {}
    for center_q in centers_q:
        # A cell cannot intersect the plane when its center is farther away than
        # the circumscribed radius of the first zone.
        offset = center_q - origin_q
        projection, *_ = np.linalg.lstsq(plane, offset, rcond=None)
        if np.linalg.norm(offset - plane @ projection) > zone_radius + tolerance:
            continue
        polygon = viewport.copy()
        differences = centers_q - center_q
        neighbor_mask = (
            (np.linalg.norm(differences, axis=1) > tolerance)
            & (np.linalg.norm(differences, axis=1) <= 2.01 * zone_radius + tolerance)
        )
        for other_q, difference in zip(
            centers_q[neighbor_mask], differences[neighbor_mask], strict=True
        ):
            normal = 2.0 * plane.T @ difference
            bound = float(other_q @ other_q - center_q @ center_q - 2.0 * difference @ origin_q)
            polygon = _clip_polygon_half_plane(
                polygon, normal, bound, tolerance=tolerance
            )
            if len(polygon) == 0:
                break
        for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
            midpoint = 0.5 * (start + end)
            on_frame = (
                abs(midpoint[0] - xlim[0]) <= tolerance
                or abs(midpoint[0] - xlim[1]) <= tolerance
                or abs(midpoint[1] - ylim[0]) <= tolerance
                or abs(midpoint[1] - ylim[1]) <= tolerance
            )
            if on_frame or np.linalg.norm(end - start) <= tolerance:
                continue
            endpoints = sorted((tuple(map(float, start)), tuple(map(float, end))))
            key = tuple(round(value, 9) for point in endpoints for value in point)
            unique[key] = (endpoints[0], endpoints[1])
    return BrillouinZoneSlice(tuple(unique[key] for key in sorted(unique)))


def draw_brillouin_zone_slice(
    ax: Any,
    zone_slice: BrillouinZoneSlice,
    *,
    color: str = "#e57373",
    linewidth: float = 1.0,
    alpha: float = 0.75,
) -> Any:
    """Add a precomputed zone slice to a Matplotlib axes."""

    from matplotlib.collections import LineCollection

    if linewidth <= 0.0:
        raise ValueError("Brillouin-zone line width must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Brillouin-zone opacity must be between zero and one")
    artist = LineCollection(
        zone_slice.segments, color=color, linewidths=linewidth, alpha=alpha,
        zorder=5, clip_on=True,
    )
    artist.set_gid("nfit-brillouin-zone-boundaries")
    ax.add_collection(artist)
    return artist


def build_brillouin_zone_scene(
    direct_lattice: ArrayLike,
    band_path: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    primitive_lattice: ArrayLike | None = None,
) -> BrillouinZoneScene:
    """Build the first Brillouin zone and a labelled reduced-coordinate path.

    ``primitive_lattice`` defines both the reciprocal translation lattice and
    the reduced coordinates of ``band_path``. It defaults to
    ``direct_lattice``. Supplying both supports a Hamiltonian represented in a
    centered conventional cell while keeping path nodes in primitive
    reciprocal coordinates.
    """

    direct = np.asarray(direct_lattice, dtype=float)
    if direct.shape != (3, 3) or not np.all(np.isfinite(direct)):
        raise ValueError("direct_lattice must be a finite 3x3 column-vector matrix")
    primitive = np.asarray(
        direct if primitive_lattice is None else primitive_lattice,
        dtype=float,
    )
    if primitive.shape != (3, 3) or not np.all(np.isfinite(primitive)):
        raise ValueError(
            "primitive_lattice must be a finite 3x3 column-vector matrix"
        )
    primitive_reciprocal = 2.0 * np.pi * np.linalg.inv(primitive).T
    vertices, faces = _first_zone_faces(primitive_reciprocal)
    nodes = []
    for item in band_path:
        reduced = np.asarray(item.get("k", ()), dtype=float)
        if reduced.shape != (3,) or not np.all(np.isfinite(reduced)):
            raise ValueError("every band-path node requires three finite k coordinates")
        cartesian = primitive_reciprocal @ reduced
        nodes.append(
            BrillouinZoneNode(
                label=str(item.get("label", "")),
                reduced=tuple(float(value) for value in reduced),
                cartesian_inv_angstrom=tuple(float(value) for value in cartesian),
                break_before=bool(item.get("break_before", False)),
            )
        )
    return BrillouinZoneScene(
        reciprocal_vectors=tuple(
            tuple(float(value) for value in primitive_reciprocal[:, axis])
            for axis in range(3)
        ),
        vertices_inv_angstrom=tuple(map(tuple, vertices)),
        faces=faces,
        path_nodes=tuple(nodes),
    )


def _component_lattices(component: Any) -> tuple[FloatArray, FloatArray | None]:
    from .crystal import lattice_vectors, primitive_lattice_vectors
    from .electronic_structure import ElectronicModel, import_wannier90

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("Brillouin-zone viewing requires a tight_binding component")
    config = component.config
    crystal = config.get("crystal")
    primitive = None
    if (
        isinstance(crystal, dict)
        and crystal.get("sites")
        and config.get("orbital_manifolds")
    ):
        direct = lattice_vectors(crystal["lattice"])
        primitive = primitive_lattice_vectors(
            crystal["lattice"],
            str(crystal.get("spacegroup", "P 1")),
        )
    elif isinstance(config.get("model_data"), dict) and config["model_data"]:
        model = ElectronicModel.from_dict(config["model_data"])
        direct = model.direct_lattice
    elif str(config.get("source_path", "")).strip():
        direct = import_wannier90(
            str(config["source_path"]),
            periodic_axes=tuple(config.get("periodic_axes") or (0, 1, 2)),
        ).direct_lattice
    else:
        raise ValueError("the tight-binding model has no electronic lattice")
    return np.asarray(direct, dtype=float), primitive


def standard_band_path(
    crystal: Any,
    *,
    convention: str = "hinuma",
    symprec: float = 1.0e-5,
) -> tuple[dict[str, Any], ...]:
    """Return a standard labelled path in nfit's primitive reciprocal basis.

    ``"hinuma"`` selects the Hinuma--Pizzi--Kumagai--Oba--Tanaka (HPKOT)
    convention implemented by Seek-path. ``"setyawan_curtarolo"`` selects
    ASE's Setyawan--Curtarolo convention. Disconnected path sections carry
    ``break_before=True`` so plots and the 3D viewer do not join them.
    """

    selected = str(convention).strip().lower()
    if selected in {"hinuma", "hpkot", "seekpath"}:
        selected = "hinuma"
    elif selected in {"setyawan_curtarolo", "setyawan-curtarolo", "sc", "ase"}:
        selected = "setyawan_curtarolo"
    else:
        raise ValueError(
            "standard path convention must be hinuma or setyawan_curtarolo"
        )

    from .crystal import (
        expand_crystal_sites,
        lattice_vectors,
        primitive_lattice_vectors,
        validate_crystal,
    )

    validate_crystal(crystal)
    nfit_primitive = primitive_lattice_vectors(
        crystal["lattice"],
        str(crystal.get("spacegroup", "P 1")),
    )

    def display_label(label: str) -> str:
        return "Γ" if label.upper() in {"G", "GAMMA"} else label

    if selected == "setyawan_curtarolo":
        try:
            from ase.cell import Cell
            from ase.dft.kpoints import parse_path_string
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise ImportError(
                "Setyawan-Curtarolo paths require ASE"
            ) from exc

        bravais = Cell(nfit_primitive.T).get_bravais_lattice(
            eps=float(symprec)
        )
        ase_path = bravais.bandpath()
        ase_reciprocal = np.asarray(
            ase_path.cell.reciprocal(),
            dtype=float,
        )
        nfit_reciprocal = np.linalg.inv(nfit_primitive).T
        point_coordinates = {
            str(label): (
                np.asarray(coordinate, dtype=float)
                @ ase_reciprocal
                @ np.linalg.inv(nfit_reciprocal.T)
            )
            for label, coordinate in ase_path.special_points.items()
        }
        nodes: list[dict[str, Any]] = []
        for section in parse_path_string(ase_path.path):
            for index, label in enumerate(section):
                nodes.append(
                    {
                        "label": display_label(str(label)),
                        "k": np.round(
                            point_coordinates[str(label)],
                            12,
                        ).tolist(),
                        **(
                            {"break_before": True}
                            if nodes and index == 0
                            else {}
                        ),
                    }
                )
        if len(nodes) < 2:
            raise ValueError("ASE returned no connected path segments")
        return tuple(nodes)

    try:
        import seekpath
    except ImportError as exc:  # pragma: no cover - declared dependency
        raise ImportError("standard high-symmetry paths require seekpath") from exc

    labels = [str(site["label"]) for site in crystal.get("sites", ())]
    sites = expand_crystal_sites(crystal, labels)
    if not sites:
        raise ValueError("a standard band path requires at least one crystal site")
    species_ids: dict[str, int] = {}
    types = []
    for site in sites:
        species = str(site.element or site.label)
        species_ids.setdefault(species, len(species_ids) + 1)
        types.append(species_ids[species])
    conventional = lattice_vectors(crystal["lattice"])
    result = seekpath.get_path(
        (
            conventional.T,
            np.asarray([site.position for site in sites], dtype=float),
            np.asarray(types, dtype=int),
        ),
        with_time_reversal=True,
        recipe="hpkot",
        symprec=float(symprec),
    )
    seek_primitive = np.asarray(result["primitive_lattice"], dtype=float).T
    seek_reciprocal = 2.0 * np.pi * np.linalg.inv(seek_primitive).T
    nfit_reciprocal = 2.0 * np.pi * np.linalg.inv(nfit_primitive).T
    point_coordinates = {
        str(label): np.linalg.solve(
            nfit_reciprocal,
            seek_reciprocal @ np.asarray(coordinate, dtype=float),
        )
        for label, coordinate in result["point_coords"].items()
    }

    nodes: list[dict[str, Any]] = []
    previous_end = ""
    for start, stop in result["path"]:
        if not nodes or str(start) != previous_end:
            nodes.append(
                {
                    "label": display_label(str(start)),
                    "k": np.round(point_coordinates[str(start)], 12).tolist(),
                    "break_before": bool(nodes),
                }
            )
        nodes.append(
            {
                "label": display_label(str(stop)),
                "k": np.round(point_coordinates[str(stop)], 12).tolist(),
            }
        )
        previous_end = str(stop)
    if len(nodes) < 2:
        raise ValueError("Seek-path returned no connected path segments")
    return tuple(nodes)


def set_tight_binding_standard_path(
    component: Any,
    convention: str = "hinuma",
    *,
    symprec: float = 1.0e-5,
) -> tuple[dict[str, Any], ...]:
    """Install a standard path without resolving the electronic Hamiltonian."""

    if getattr(component, "type", None) != "tight_binding":
        raise TypeError("standard band paths require a tight_binding component")
    selected = str(convention).strip().lower()
    if selected in {"hinuma", "hpkot", "seekpath"}:
        normalized = "hinuma"
        import seekpath

        provider = "seekpath"
        provider_version = str(getattr(seekpath, "__version__", "unknown"))
        convention_label = "HPKOT"
    elif selected in {
        "setyawan_curtarolo",
        "setyawan-curtarolo",
        "sc",
        "ase",
    }:
        normalized = "setyawan_curtarolo"
        import ase

        provider = "ASE"
        provider_version = str(getattr(ase, "__version__", "unknown"))
        convention_label = "Setyawan-Curtarolo"
    else:
        raise ValueError(
            "standard path convention must be hinuma or setyawan_curtarolo"
        )

    path = standard_band_path(
        component.config.get("crystal", {}),
        convention=normalized,
        symprec=symprec,
    )
    component.config["band_path"] = [dict(node) for node in path]
    component.config["band_path_convention"] = normalized
    component.config["band_path_metadata"] = {
        "provider": provider,
        "provider_version": provider_version,
        "convention": convention_label,
        "symprec": float(symprec),
    }
    return path


def brillouin_zone_scene(component: Any) -> BrillouinZoneScene:
    """Build a Brillouin-zone scene from a tight-binding component."""

    direct, primitive = _component_lattices(component)
    config = component.config
    return build_brillouin_zone_scene(
        direct,
        list(config.get("band_path", ())),
        primitive_lattice=primitive,
    )


def band_path_reciprocal_lattice(component: Any) -> FloatArray:
    """Return the reciprocal basis used by configured electronic paths."""

    direct, primitive = _component_lattices(component)
    path_lattice = direct if primitive is None else primitive
    return 2.0 * np.pi * np.linalg.inv(path_lattice).T


def brillouin_zone_script(
    component: Any,
    *,
    view_options: BrillouinZoneViewOptions | None = None,
) -> str:
    """Return editable Python reproducing the component's 3D zone view."""

    scene = brillouin_zone_scene(component)
    options = (
        BrillouinZoneViewOptions()
        if view_options is None
        else view_options
    )
    direct, _primitive = _component_lattices(component)
    reciprocal = np.asarray(scene.reciprocal_vectors, dtype=float).T
    primitive = 2.0 * np.pi * np.linalg.inv(reciprocal).T
    band_path = [
        {
            "label": node.label,
            "k": list(node.reduced),
            **({"break_before": True} if node.break_before else {}),
        }
        for node in scene.path_nodes
    ]
    return "\n".join(
        [
            '"""Inspect a labelled first Brillouin zone without project widgets."""',
            "",
            "from nfit import BrillouinZoneViewOptions, build_brillouin_zone_scene",
            "from nfit.qt_brillouin_zone_viewer import show_brillouin_zone_scene",
            "from PySide6 import QtWidgets",
            "",
            f"direct_lattice = {pformat(direct.tolist())}",
            f"primitive_lattice = {pformat(primitive.tolist())}",
            f"band_path = {pformat(band_path)}",
            "scene = build_brillouin_zone_scene(",
            "    direct_lattice,",
            "    band_path,",
            "    primitive_lattice=primitive_lattice,",
            ")",
            f"view_options = {options!r}",
            "app = QtWidgets.QApplication.instance()",
            "owns_app = app is None",
            "if owns_app:",
            "    app = QtWidgets.QApplication([])",
            "window = show_brillouin_zone_scene(scene, options=view_options)",
            "if owns_app:",
            "    app.exec()",
            "",
        ]
    )
