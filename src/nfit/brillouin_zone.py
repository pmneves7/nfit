"""Renderer-independent first-Brillouin-zone geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pprint import pformat
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BrillouinZoneNode:
    """One labelled wavevector on a configured band path."""

    label: str
    reduced: tuple[float, float, float]
    cartesian_inv_angstrom: tuple[float, float, float]


@dataclass(frozen=True)
class BrillouinZoneScene:
    """First-zone polyhedron, reciprocal basis, and labelled path."""

    reciprocal_vectors: tuple[tuple[float, float, float], ...]
    vertices_inv_angstrom: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]
    path_nodes: tuple[BrillouinZoneNode, ...]


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


def build_brillouin_zone_scene(
    direct_lattice: ArrayLike,
    band_path: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    primitive_lattice: ArrayLike | None = None,
) -> BrillouinZoneScene:
    """Build the first Brillouin zone and a labelled reduced-coordinate path.

    ``direct_lattice`` defines the reduced coordinates of ``band_path``.
    ``primitive_lattice`` generates the reciprocal translation lattice and
    defaults to ``direct_lattice``. Supplying both supports a Hamiltonian
    represented in a centered conventional cell.
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
    path_reciprocal = 2.0 * np.pi * np.linalg.inv(direct).T
    primitive_reciprocal = 2.0 * np.pi * np.linalg.inv(primitive).T
    vertices, faces = _first_zone_faces(primitive_reciprocal)
    nodes = []
    for item in band_path:
        reduced = np.asarray(item.get("k", ()), dtype=float)
        if reduced.shape != (3,) or not np.all(np.isfinite(reduced)):
            raise ValueError("every band-path node requires three finite k coordinates")
        cartesian = path_reciprocal @ reduced
        nodes.append(
            BrillouinZoneNode(
                label=str(item.get("label", "")),
                reduced=tuple(float(value) for value in reduced),
                cartesian_inv_angstrom=tuple(float(value) for value in cartesian),
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
    if isinstance(config.get("model_data"), dict) and config["model_data"]:
        model = ElectronicModel.from_dict(config["model_data"])
        direct = model.direct_lattice
        if (
            isinstance(crystal, dict)
            and crystal.get("sites")
            and config.get("orbital_manifolds")
        ):
            crystal_direct = lattice_vectors(crystal["lattice"])
            if np.allclose(crystal_direct, direct, atol=1e-9):
                primitive = primitive_lattice_vectors(
                    crystal["lattice"],
                    str(crystal.get("spacegroup", "P 1")),
                )
    elif isinstance(crystal, dict) and crystal.get("sites"):
        direct = lattice_vectors(crystal["lattice"])
        primitive = primitive_lattice_vectors(
            crystal["lattice"],
            str(crystal.get("spacegroup", "P 1")),
        )
    elif str(config.get("source_path", "")).strip():
        direct = import_wannier90(
            str(config["source_path"]),
            periodic_axes=tuple(config.get("periodic_axes") or (0, 1, 2)),
        ).direct_lattice
    else:
        raise ValueError("the tight-binding model has no electronic lattice")
    return np.asarray(direct, dtype=float), primitive


def brillouin_zone_scene(component: Any) -> BrillouinZoneScene:
    """Build a Brillouin-zone scene from a tight-binding component."""

    direct, primitive = _component_lattices(component)
    config = component.config
    return build_brillouin_zone_scene(
        direct,
        list(config.get("band_path", ())),
        primitive_lattice=primitive,
    )


def brillouin_zone_script(component: Any) -> str:
    """Return editable Python reproducing the component's 3D zone view."""

    scene = brillouin_zone_scene(component)
    direct, _primitive = _component_lattices(component)
    reciprocal = np.asarray(scene.reciprocal_vectors, dtype=float).T
    primitive = 2.0 * np.pi * np.linalg.inv(reciprocal).T
    band_path = [
        {"label": node.label, "k": list(node.reduced)}
        for node in scene.path_nodes
    ]
    return "\n".join(
        [
            '"""Inspect a labelled first Brillouin zone without project widgets."""',
            "",
            "from nfit import build_brillouin_zone_scene",
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
            "app = QtWidgets.QApplication.instance()",
            "owns_app = app is None",
            "if owns_app:",
            "    app = QtWidgets.QApplication([])",
            "window = show_brillouin_zone_scene(scene)",
            "if owns_app:",
            "    app.exec()",
            "",
        ]
    )
