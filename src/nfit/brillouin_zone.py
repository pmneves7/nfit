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
