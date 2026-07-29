"""Public, renderer-independent geometry scenes for model inspection."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pprint import pformat
from typing import Any, Literal

import numpy as np

from .crystal import (
    cartesian_rotation,
    expand_crystal_sites,
    lattice_vectors,
    orbits_from_config,
)
from .electronic_builder import OrbitalManifold


@dataclass(frozen=True)
class GeometrySite:
    """One atom in the displayed unit cell."""

    identifier: str
    representative_label: str
    site_label: str
    element: str
    fractional: tuple[float, float, float]
    cartesian: tuple[float, float, float]
    active: bool
    color: tuple[float, float, float]
    display_radius: float


@dataclass(frozen=True)
class GeometryFrame:
    """One local orbital frame attached to an expanded site."""

    identifier: str
    site_identifier: str
    manifold_label: str
    origin_cartesian: tuple[float, float, float]
    axes_cartesian: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class GeometryOrbital:
    """An abstract orbital token; it is not a wavefunction isosurface."""

    identifier: str
    site_identifier: str
    manifold_label: str
    orbital_label: str
    preset: str
    center_cartesian: tuple[float, float, float]
    display_center_cartesian: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass(frozen=True)
class GeometryPathway:
    """One displayed hopping or exchange pathway."""

    identifier: str
    orbit_label: str
    start_cartesian: tuple[float, float, float]
    end_cartesian: tuple[float, float, float]
    representative: bool
    kind: Literal["exchange", "hopping"]


@dataclass(frozen=True)
class ModelGeometryScene:
    """Complete model-geometry scene consumed by GUI or external renderers."""

    model_name: str
    model_type: str
    lattice_vectors: tuple[tuple[float, float, float], ...]
    cell_vertices: tuple[tuple[float, float, float], ...]
    cell_edges: tuple[tuple[int, int], ...]
    sites: tuple[GeometrySite, ...]
    frames: tuple[GeometryFrame, ...]
    orbitals: tuple[GeometryOrbital, ...]
    pathways: tuple[GeometryPathway, ...]
    pathway_labels: tuple[str, ...]


def _stable_color(label: str) -> tuple[float, float, float]:
    digest = hashlib.sha256(str(label).encode()).digest()
    values = np.frombuffer(digest[:3], dtype=np.uint8).astype(float) / 255.0
    values = 0.25 + 0.65 * values
    return tuple(float(value) for value in values)


_ELEMENT_COLORS = {
    "H": (1.00, 1.00, 1.00),
    "C": (0.35, 0.35, 0.35),
    "N": (0.19, 0.31, 0.97),
    "O": (1.00, 0.05, 0.05),
    "F": (0.56, 0.88, 0.31),
    "Na": (0.67, 0.36, 0.95),
    "Mg": (0.54, 1.00, 0.00),
    "Al": (0.75, 0.65, 0.65),
    "Si": (0.94, 0.78, 0.63),
    "P": (1.00, 0.50, 0.00),
    "S": (1.00, 1.00, 0.19),
    "Cl": (0.12, 0.94, 0.12),
    "K": (0.56, 0.25, 0.83),
    "Ca": (0.24, 1.00, 0.00),
    "Ti": (0.75, 0.76, 0.78),
    "V": (0.65, 0.65, 0.67),
    "Cr": (0.54, 0.60, 0.78),
    "Mn": (0.61, 0.48, 0.78),
    "Fe": (0.88, 0.40, 0.20),
    "Co": (0.94, 0.56, 0.63),
    "Ni": (0.31, 0.82, 0.31),
    "Cu": (0.78, 0.50, 0.20),
    "Zn": (0.49, 0.50, 0.69),
    "Br": (0.65, 0.16, 0.16),
    "Ag": (0.75, 0.75, 0.75),
    "I": (0.58, 0.00, 0.58),
    "Ba": (0.00, 0.79, 0.00),
    "Ce": (1.00, 1.00, 0.78),
    "Nd": (0.78, 1.00, 0.78),
    "Sm": (0.56, 1.00, 0.78),
    "Gd": (0.27, 1.00, 0.78),
    "Yb": (0.00, 0.75, 0.22),
    "Pt": (0.82, 0.82, 0.88),
    "Au": (1.00, 0.82, 0.14),
    "Pb": (0.34, 0.35, 0.38),
}


def element_display_color(element: str) -> tuple[float, float, float]:
    """Return a stable CPK-like display color for a chemical element."""

    symbol = str(element).strip()
    if symbol:
        symbol = symbol[:1].upper() + symbol[1:].lower()
    return _ELEMENT_COLORS.get(symbol, _stable_color(f"element:{symbol}"))


def element_display_radius(element: str, lattice_scale: float) -> float:
    """Return a compact sphere radius in Å for a unit-cell model view."""

    try:
        import gemmi

        covalent_radius = float(gemmi.Element(str(element)).covalent_r)
    except (ValueError, RuntimeError):
        covalent_radius = 1.0
    if not np.isfinite(covalent_radius) or covalent_radius <= 0.0:
        covalent_radius = 1.0
    scale = float(lattice_scale)
    return float(np.clip(0.055 * scale * covalent_radius, 0.035 * scale, 0.10 * scale))


def _cell_geometry(
    direct_lattice: np.ndarray,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, int], ...],
]:
    fractional = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (1, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (0, 1, 1),
            (1, 1, 1),
        ],
        dtype=float,
    )
    cartesian = fractional @ direct_lattice.T
    edges = tuple(
        (start, stop)
        for start in range(8)
        for stop in range(start + 1, 8)
        if np.count_nonzero(fractional[start] != fractional[stop]) == 1
    )
    return tuple(map(tuple, cartesian)), edges


def _expanded_by_representative(
    crystal: Mapping[str, Any], representative_labels: Sequence[str]
) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for label in representative_labels:
        for site in expand_crystal_sites(crystal, [str(label)]):
            key = tuple(float(value) for value in site.position)
            if key in seen:
                continue
            seen.add(key)
            result.append((str(label), site))
    return result


def _pathways(
    component: Any,
    direct_lattice: np.ndarray,
    *,
    selected_pathway: str | None,
    pathway_mode: str,
) -> tuple[GeometryPathway, ...]:
    config = component.config
    raw_orbits = (
        config.get("orbits", ())
        if component.type == "heisenberg_rpa"
        else config.get("spatial_orbits", ())
    )
    if not raw_orbits:
        return ()
    positions = np.asarray(config.get("site_positions", ()), dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        return ()
    kind = "exchange" if component.type == "heisenberg_rpa" else "hopping"
    result: list[GeometryPathway] = []
    for orbit in orbits_from_config(raw_orbits):
        if selected_pathway and orbit.label != selected_pathway:
            continue
        bonds = orbit.bonds if pathway_mode == "all" else orbit.bonds[:1]
        for index, bond in enumerate(bonds):
            start = direct_lattice @ positions[bond.site_i]
            end_fractional = positions[bond.site_j] + np.asarray(bond.offset)
            end = direct_lattice @ end_fractional
            result.append(
                GeometryPathway(
                    identifier=f"{kind}:{orbit.label}:{index}",
                    orbit_label=orbit.label,
                    start_cartesian=tuple(float(value) for value in start),
                    end_cartesian=tuple(float(value) for value in end),
                    representative=index == 0,
                    kind=kind,
                )
            )
    return tuple(result)


def model_geometry_scene(
    component: Any,
    *,
    include_ghost_sites: bool = True,
    show_orbitals: bool = True,
    show_local_frames: bool = True,
    selected_pathway: str | None = None,
    selected_hopping_term: str | None = None,
    pathway_mode: Literal["representative", "all"] = "representative",
) -> ModelGeometryScene:
    """Build a shared unit-cell scene for tight-binding or Heisenberg models."""

    if component.type not in {"tight_binding", "heisenberg_rpa"}:
        raise ValueError("model geometry is available for tight_binding and heisenberg_rpa")
    if pathway_mode not in {"representative", "all"}:
        raise ValueError("pathway_mode must be 'representative' or 'all'")
    if selected_hopping_term:
        matched = next(
            (
                item
                for item in component.config.get("hopping_terms", ())
                if str(item.get("identifier", "")) == selected_hopping_term
                or str(item.get("label", "")) == selected_hopping_term
            ),
            None,
        )
        if matched is None:
            raise ValueError(
                f"unknown hopping term {selected_hopping_term!r}"
            )
        selected_pathway = str(matched.get("orbit_label", ""))
    crystal = component.config.get("crystal")
    if not isinstance(crystal, Mapping):
        raise ValueError("the model has no crystal geometry")
    direct = lattice_vectors(crystal["lattice"])
    lattice_scale = min(
        float(np.linalg.norm(direct[:, axis])) for axis in range(3)
    )
    all_labels = [str(site["label"]) for site in crystal.get("sites", ())]
    if component.type == "tight_binding":
        manifolds = tuple(
            OrbitalManifold.from_dict(item)
            for item in component.config.get("orbital_manifolds", ())
        )
        active_labels = tuple(dict.fromkeys(item.site_label for item in manifolds))
    else:
        manifolds = ()
        active_labels = tuple(
            str(value) for value in component.config.get("magnetic_sites", ())
        )
    expanded = _expanded_by_representative(crystal, all_labels)
    sites: list[GeometrySite] = []
    active_site_ids: dict[tuple[str, tuple[float, float, float]], str] = {}
    for representative, site in expanded:
        active = representative in active_labels
        if not active and not include_ghost_sites:
            continue
        cartesian = direct @ np.asarray(site.position, dtype=float)
        identifier = f"{representative}:{site.label}"
        sites.append(
            GeometrySite(
                identifier=identifier,
                representative_label=representative,
                site_label=site.label,
                element=site.element,
                fractional=tuple(float(value) for value in site.position),
                cartesian=tuple(float(value) for value in cartesian),
                active=active,
                color=element_display_color(site.element),
                display_radius=element_display_radius(site.element, lattice_scale),
            )
        )
        if active:
            active_site_ids[(representative, tuple(site.position))] = identifier

    frames: list[GeometryFrame] = []
    orbitals: list[GeometryOrbital] = []
    token_radius = 0.06 * lattice_scale
    for manifold in manifolds:
        expanded_sites = expand_crystal_sites(crystal, [manifold.site_label])
        for site in expanded_sites:
            position_key = (manifold.site_label, tuple(site.position))
            site_identifier = active_site_ids.get(position_key)
            if site_identifier is None:
                continue
            origin = direct @ np.asarray(site.position, dtype=float)
            generator = (
                np.eye(3)
                if site.rotation is None
                else cartesian_rotation(site.rotation, crystal["lattice"])
            )
            frame = generator @ np.asarray(manifold.local_frame, dtype=float)
            if show_local_frames:
                frames.append(
                    GeometryFrame(
                        identifier=f"frame:{site_identifier}:{manifold.label}",
                        site_identifier=site_identifier,
                        manifold_label=manifold.label,
                        origin_cartesian=tuple(float(value) for value in origin),
                        axes_cartesian=tuple(
                            tuple(float(value) for value in frame[:, axis])
                            for axis in range(3)
                        ),
                    )
                )
            if not show_orbitals:
                continue
            count = len(manifold.orbitals)
            for index, orbital in enumerate(manifold.orbitals):
                angle = 2.0 * np.pi * index / max(count, 1)
                local_offset = token_radius * np.asarray(
                    [np.cos(angle), np.sin(angle), 0.35 * ((index % 3) - 1)]
                )
                display = origin + frame @ local_offset
                orbitals.append(
                    GeometryOrbital(
                        identifier=f"orbital:{site_identifier}:{manifold.label}:{orbital}",
                        site_identifier=site_identifier,
                        manifold_label=manifold.label,
                        orbital_label=orbital,
                        preset=manifold.preset,
                        center_cartesian=tuple(float(value) for value in origin),
                        display_center_cartesian=tuple(
                            float(value) for value in display
                        ),
                        color=_stable_color(f"{manifold.preset}:{orbital}"),
                    )
                )
    vertices, edges = _cell_geometry(direct)
    raw_orbits = (
        component.config.get("orbits", ())
        if component.type == "heisenberg_rpa"
        else component.config.get("spatial_orbits", ())
    )
    pathway_labels = tuple(
        str(item.get("label", ""))
        for item in raw_orbits
        if isinstance(item, Mapping) and item.get("label")
    )
    return ModelGeometryScene(
        model_name=str(component.name),
        model_type=str(component.type),
        lattice_vectors=tuple(
            tuple(float(value) for value in direct[:, axis]) for axis in range(3)
        ),
        cell_vertices=vertices,
        cell_edges=edges,
        sites=tuple(sites),
        frames=tuple(frames),
        orbitals=tuple(orbitals),
        pathways=_pathways(
            component,
            direct,
            selected_pathway=selected_pathway,
            pathway_mode=pathway_mode,
        ),
        pathway_labels=pathway_labels,
    )


def model_geometry_script(component: Any) -> str:
    """Return editable Python reproducing a model-geometry viewer scene."""

    config = copy.deepcopy(component.config)
    lines = [
        '"""Inspect model geometry without constructing nfit project widgets."""',
        "",
        "from nfit import ModelComponentSpec, model_geometry_scene",
        "from nfit.qt_model_geometry_viewer import show_model_geometry_scene",
        "from PySide6 import QtWidgets",
        "",
        (
            f"component = ModelComponentSpec(name={str(component.name)!r}, "
            f"type={str(component.type)!r}, config={pformat(config, sort_dicts=True)})"
        ),
        "scene = model_geometry_scene(component)",
        "app = QtWidgets.QApplication.instance()",
        "owns_app = app is None",
        "if owns_app:",
        "    app = QtWidgets.QApplication([])",
        "window = show_model_geometry_scene(scene)",
        "if owns_app:",
        "    app.exec()",
        "",
    ]
    return "\n".join(lines)
