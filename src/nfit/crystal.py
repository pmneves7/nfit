"""Model-independent crystal structures, expanded sites, and spatial bonds.

This module reads CIF files, expands selected crystallographic sites through a
space group, and enumerates neighbor bonds grouped into symmetry orbits. The
geometry is shared by magnetic exchange and electronic tight-binding models;
model-specific interactions are attached in their own layers.

The generic spatial-orbit API labels shells ``B1``, ``B2``, ... . The
Heisenberg wrapper retains ``J1``, ``J2``, ... . Symmetry-inequivalent orbits
at the same distance receive suffixes such as ``B3a``/``B3b`` or
``J3a``/``J3b``.

All public entry points accept and return plain JSON-serializable dicts so
crystal and bond configuration can be stored directly in project files and
model-component configuration.

Space-group operations and CIF parsing use `gemmi <https://gemmi.readthedocs.io>`_,
imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

_POSITION_DECIMALS = 6
_POSITION_TOL = 10.0**-_POSITION_DECIMALS
_DISTANCE_DECIMALS = 5


def _require_gemmi():
    try:
        import gemmi
    except ImportError as exc:
        raise ImportError(
            "crystal symmetry and CIF import require the 'gemmi' package; "
            "install it with 'pip install gemmi' or 'conda install -c "
            "conda-forge gemmi'"
        ) from exc
    return gemmi


@dataclass(frozen=True)
class CrystalSite:
    """One selected site of the expanded unit cell in fractional coordinates.

    ``rotation`` is the fractional rotation matrix of the space-group operation
    that generated this site from its Wyckoff representative (``None`` for
    legacy payloads). ``ion`` remains available for magnetic form factors;
    ``element`` carries the crystallographic species independently.
    """

    label: str
    position: tuple[float, float, float]
    ion: str = ""
    element: str = ""
    rotation: tuple[tuple[float, float, float], ...] | None = None


@dataclass(frozen=True)
class BondSymmetry:
    """How an orbit's representative bond maps onto one member bond.

    ``rotation`` is the fractional rotation of the space-group operation;
    ``reverses`` records that the operation mapped the representative onto the
    member with opposite orientation (the stored bond is the canonicalized
    form), in which case exchange tensors transpose.
    """

    rotation: tuple[tuple[float, float, float], ...]
    reverses: bool = False


@dataclass(frozen=True)
class Bond:
    """A bond from site ``site_i`` in the home cell to ``site_j`` offset by ``offset`` cells."""

    site_i: int
    site_j: int
    offset: tuple[int, int, int]

    def reversed(self) -> Bond:
        return Bond(
            self.site_j,
            self.site_i,
            (-self.offset[0], -self.offset[1], -self.offset[2]),
        )

    def canonical(self) -> Bond:
        other = self.reversed()
        return min(self, other, key=lambda b: (b.site_i, b.site_j, b.offset))


@dataclass(frozen=True)
class BondOrbit:
    """All spatial bonds equivalent under the space group.

    ``operations`` (when present) is aligned with ``bonds`` and records, per
    bond, the symmetry operation carrying the orbit's representative bond
    (``bonds[0]``, whose entry is the identity) onto it. Interaction layers use
    this information to rotate exchange tensors or orbital hopping matrices.
    """

    label: str
    distance_angstrom: float
    bonds: tuple[Bond, ...]
    operations: tuple[BondSymmetry, ...] | None = None

    @property
    def multiplicity(self) -> int:
        return len(self.bonds)


def crystal_from_cif(path: str) -> dict[str, Any]:
    """Read a CIF file into a plain crystal dict.

    Returns ``{"lattice": {"a", "b", "c", "alpha", "beta", "gamma"},
    "spacegroup": str, "sites": [{"label", "element", "position", "ion"}]}``
    with positions fractional. The ``ion`` field (magnetic form factor key) is
    left empty for the user to assign.
    """

    gemmi = _require_gemmi()
    source = Path(path)
    content = source.read_bytes()
    structure = gemmi.read_small_structure(str(source))
    if not structure.sites:
        raise ValueError(f"CIF file {path!r} defines no atomic sites")
    cell = structure.cell
    # Some legacy CIFs append a setting marker such as ``Z`` to the
    # Hermann-Mauguin field. Gemmi's parsed space-group object incorporates
    # the CIF symmetry operations and retains any genuine origin choice, so it
    # is more authoritative than that raw text field. Fall back to the IT
    # number only when the parsed object is unavailable.
    try:
        spacegroup = structure.spacegroup.xhm()
    except (AttributeError, RuntimeError, ValueError):
        try:
            spacegroup = gemmi.get_spacegroup_reference_setting(structure.spacegroup_number).xhm()
        except (RuntimeError, ValueError):
            spacegroup = structure.spacegroup_hm or "P 1"
    sites = [
        {
            "label": site.label or site.type_symbol,
            "element": site.type_symbol,
            "position": [
                round(float(site.fract.x), _POSITION_DECIMALS),
                round(float(site.fract.y), _POSITION_DECIMALS),
                round(float(site.fract.z), _POSITION_DECIMALS),
            ],
            "ion": "",
        }
        for site in structure.sites
    ]
    return {
        "lattice": {
            "a": float(cell.a),
            "b": float(cell.b),
            "c": float(cell.c),
            "alpha": float(cell.alpha),
            "beta": float(cell.beta),
            "gamma": float(cell.gamma),
        },
        "spacegroup": str(spacegroup),
        "sites": sites,
        "provenance": {
            "source": "cif",
            "path": str(source.resolve()),
            "size_bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        },
    }


def lattice_vectors(lattice: Mapping[str, Any]) -> FloatArray:
    """Return the 3x3 matrix whose columns are the lattice vectors in Angstrom."""

    a = float(lattice["a"])
    b = float(lattice["b"])
    c = float(lattice["c"])
    alpha = np.radians(float(lattice.get("alpha", 90.0)))
    beta = np.radians(float(lattice.get("beta", 90.0)))
    gamma = np.radians(float(lattice.get("gamma", 90.0)))
    avec = np.array([a, 0.0, 0.0])
    bvec = np.array([b * np.cos(gamma), b * np.sin(gamma), 0.0])
    cx = c * np.cos(beta)
    cy = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz_sq = c * c - cx * cx - cy * cy
    if cz_sq <= 0.0:
        raise ValueError("lattice parameters produce a non-positive unit-cell volume")
    cvec = np.array([cx, cy, np.sqrt(cz_sq)])
    return np.column_stack([avec, bvec, cvec])


def primitive_lattice_vectors(
    lattice: Mapping[str, Any],
    spacegroup: str,
) -> FloatArray:
    """Return primitive direct-lattice vectors for a conventional crystal cell.

    Gemmi supplies the centering translations of the selected space-group
    setting. The returned columns generate that complete translation lattice,
    including A-, B-, C-, I-, F-, and rhombohedrally centered cells.
    """

    conventional = lattice_vectors(lattice)
    gemmi = _require_gemmi()
    group = _resolve_spacegroup(gemmi, spacegroup)
    denominator = float(gemmi.Op.DEN)
    centerings = tuple(
        np.asarray(value, dtype=float) / denominator
        for value in group.operations().cen_ops
    )
    transforms = {
        "P": np.eye(3),
        "A": np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.5, -0.5], [0.0, 0.5, 0.5]]
        ),
        "B": np.asarray(
            [[0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]
        ),
        "C": np.asarray(
            [[0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]
        ),
        "I": np.asarray(
            [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]
        ),
        "F": np.asarray(
            [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
        ),
        "R": np.asarray(
            [
                [2.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0],
                [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            ]
        ),
    }
    centering_type = str(group.centring_type())
    if centering_type in transforms:
        transform = transforms[centering_type]
        expected = 1.0 / len(centerings)
        if np.isclose(abs(np.linalg.det(transform)), expected, atol=1e-10):
            return conventional @ transform

    # Fallback for an uncommon future setting: search the centering translation
    # lattice for a shortest right-handed primitive basis.
    candidates: dict[tuple[float, float, float], FloatArray] = {}
    for centering in centerings:
        for offset in product((-1, 0, 1), repeat=3):
            vector = centering + np.asarray(offset, dtype=float)
            if np.linalg.norm(vector) <= 1e-12:
                continue
            key = tuple(float(value) for value in np.round(vector, 12))
            candidates[key] = vector
    ordered = sorted(
        candidates.values(),
        key=lambda value: (
            float(np.linalg.norm(conventional @ value)),
            tuple(float(item) for item in value),
        ),
    )
    target_volume = 1.0 / len(centerings)
    best: tuple[tuple[float, float, tuple[float, ...]], FloatArray] | None = None
    for vectors in combinations(ordered, 3):
        transform = np.column_stack(vectors)
        determinant = float(np.linalg.det(transform))
        if not np.isclose(abs(determinant), target_volume, atol=1e-10):
            continue
        if determinant < 0.0:
            transform = transform[:, [1, 0, 2]]
        physical = conventional @ transform
        lengths = np.linalg.norm(physical, axis=0)
        score = (
            float(np.sum(lengths * lengths)),
            float(np.linalg.cond(physical)),
            tuple(float(value) for value in transform.ravel(order="F")),
        )
        if best is None or score < best[0]:
            best = (score, physical)
    if best is None:
        raise ValueError(
            f"could not construct a primitive lattice for space group {spacegroup!r}"
        )
    return best[1]


def validate_crystal(crystal: Mapping[str, Any]) -> None:
    """Validate the shared JSON crystal geometry contract."""

    if not isinstance(crystal, Mapping):
        raise TypeError("crystal must be a mapping")
    lattice = crystal.get("lattice")
    if not isinstance(lattice, Mapping):
        raise ValueError("crystal must define a lattice mapping")
    basis = lattice_vectors(lattice)
    if np.any(~np.isfinite(basis)):
        raise ValueError("crystal lattice must be finite")
    if not str(crystal.get("spacegroup", "")).strip():
        raise ValueError("crystal spacegroup cannot be empty")
    sites = crystal.get("sites")
    if not isinstance(sites, Sequence) or isinstance(sites, (str, bytes)):
        raise ValueError("crystal sites must be a sequence")
    labels: set[str] = set()
    for site in sites:
        if not isinstance(site, Mapping):
            raise ValueError("every crystal site must be a mapping")
        label = str(site.get("label", "")).strip()
        if not label or label in labels:
            raise ValueError("crystal site labels must be nonempty and unique")
        labels.add(label)
        position = np.asarray(site.get("position", []), dtype=float)
        if position.shape != (3,) or np.any(~np.isfinite(position)):
            raise ValueError(
                f"crystal site {label!r} must have three finite fractional coordinates"
            )


def _lattice_vectors(lattice: Mapping[str, Any]) -> FloatArray:
    """Compatibility alias for the former private lattice helper."""

    return lattice_vectors(lattice)


def _symmetry_operations(spacegroup: str) -> list[tuple[FloatArray, FloatArray]]:
    """Return ``(rotation, translation)`` pairs in fractional coordinates."""

    gemmi = _require_gemmi()
    group = _resolve_spacegroup(gemmi, spacegroup)
    den = float(gemmi.Op.DEN)
    operations = []
    for op in group.operations():
        rotation = np.asarray(op.rot, dtype=float) / den
        translation = np.asarray(op.tran, dtype=float) / den
        operations.append((rotation, translation))
    return operations


def spacegroup_operations(
    crystal: Mapping[str, Any],
) -> tuple[tuple[FloatArray, FloatArray], ...]:
    """Return all fractional-coordinate operations for a validated crystal."""

    validate_crystal(crystal)
    return tuple(
        (np.array(rotation, copy=True), np.array(translation, copy=True))
        for rotation, translation in _symmetry_operations(
            str(crystal.get("spacegroup", "P 1"))
        )
    )


def site_symmetry_operations(
    crystal: Mapping[str, Any], site_label: str
) -> list[dict[str, Any]]:
    """Return the space-group operations that leave one representative site fixed.

    Rotations and translations are returned in fractional coordinates together
    with the corresponding Cartesian rotation. An operation belongs to the site
    stabilizer when it maps the representative position onto itself modulo a
    lattice translation.
    """

    validate_crystal(crystal)
    entries = {
        str(site["label"]): site for site in crystal.get("sites", [])
    }
    label = str(site_label)
    if label not in entries:
        raise ValueError(
            f"site label {site_label!r} not found; crystal defines {sorted(entries)}"
        )
    base = _wrap_fractional(
        np.asarray(entries[label]["position"], dtype=float)
    )
    result: list[dict[str, Any]] = []
    for rotation, translation in _symmetry_operations(
        str(crystal.get("spacegroup", "P 1"))
    ):
        image = _wrap_fractional(rotation @ base + translation)
        if not np.allclose(image, base, atol=_POSITION_TOL):
            continue
        result.append(
            {
                "rotation_fractional": np.asarray(rotation, dtype=float).tolist(),
                "translation_fractional": np.asarray(
                    translation, dtype=float
                ).tolist(),
                "rotation_cartesian": cartesian_rotation(
                    rotation, crystal["lattice"]
                ).tolist(),
            }
        )
    return result


def _resolve_spacegroup(gemmi: Any, spacegroup: str):
    """Resolve a user-entered space group, preferring reference origin choices.

    Gemmi resolves bare ``"227"`` and ``"F d -3 m"`` to origin choice 1, but
    the reference setting is origin choice 2. Bare names should use the
    reference setting; explicit entries such as ``"F d -3 m:1"`` are preserved.
    """

    name = str(spacegroup).strip()
    if name.isdigit():
        try:
            return gemmi.get_spacegroup_reference_setting(int(name))
        except (RuntimeError, ValueError) as exc:
            raise ValueError(f"unknown space group {spacegroup!r}") from exc

    group = gemmi.find_spacegroup_by_name(name)
    if group is None:
        raise ValueError(f"unknown space group {spacegroup!r}")

    if ":" not in name and not group.is_reference_setting():
        reference = gemmi.get_spacegroup_reference_setting(group.number)
        if reference.hm == group.hm and reference.qualifier == group.qualifier:
            return reference
    return group


def _wrap_fractional(position: FloatArray) -> FloatArray:
    wrapped = np.mod(np.round(position, _POSITION_DECIMALS), 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=_POSITION_TOL)] = 0.0
    return np.round(wrapped, _POSITION_DECIMALS)


def expand_crystal_sites(
    crystal: Mapping[str, Any], site_labels: Sequence[str]
) -> list[CrystalSite]:
    """Expand the chosen Wyckoff sites through the space group.

    Returns the full list of symmetry-equivalent magnetic sites in the unit
    cell, deduplicated modulo lattice translations, in a deterministic order.
    Site indices in bonds and orbits refer to this list.
    """

    sites = {str(site["label"]): site for site in crystal.get("sites", [])}
    missing = [label for label in site_labels if str(label) not in sites]
    if missing:
        raise ValueError(
            f"site label(s) {missing!r} not found; crystal defines "
            f"{sorted(sites)}"
        )
    operations = _symmetry_operations(crystal.get("spacegroup", "P 1"))

    expanded: list[CrystalSite] = []
    seen: set[tuple[float, ...]] = set()
    for label in site_labels:
        site = sites[str(label)]
        base = np.asarray(site["position"], dtype=float)
        ion = str(site.get("ion", "") or "")
        element = str(site.get("element", "") or "")
        count = 0
        for rotation, translation in operations:
            position = _wrap_fractional(rotation @ base + translation)
            key = tuple(position.tolist())
            if key in seen:
                continue
            seen.add(key)
            count += 1
            expanded.append(
                CrystalSite(
                    label=f"{label}_{count}",
                    position=(float(position[0]), float(position[1]), float(position[2])),
                    ion=ion,
                    element=element,
                    rotation=_rotation_tuple(rotation),
                )
            )
    return expanded


def expand_magnetic_sites(
    crystal: Mapping[str, Any], site_labels: Sequence[str]
) -> list[CrystalSite]:
    """Compatibility wrapper for magnetic models using the shared site expansion."""

    return expand_crystal_sites(crystal, site_labels)


def _rotation_tuple(rotation: FloatArray) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(x) for x in row) for row in np.asarray(rotation, dtype=float))


def _site_index_and_offset(
    position: FloatArray, site_positions: FloatArray
) -> tuple[int, FloatArray]:
    """Match a fractional position to an expanded site plus an integer cell offset."""

    for index, reference in enumerate(site_positions):
        delta = position - reference
        offset = np.round(delta)
        if np.allclose(delta, offset, atol=1e-4):
            return index, offset.astype(int)
    raise ValueError(
        "a symmetry operation mapped a magnetic site onto a position that is "
        "not in the expanded site list; the site expansion and space group "
        "are inconsistent"
    )


def _transform_bond(
    bond: Bond,
    rotation: FloatArray,
    translation: FloatArray,
    site_positions: FloatArray,
) -> tuple[Bond, bool]:
    """Map a bond through a symmetry op; return ``(canonical bond, reversed)``.

    ``reversed`` is True when canonicalization flipped the transformed bond's
    orientation relative to the op's image of the input bond — exchange
    tensors carried onto the stored bond must then be transposed.
    """

    start = site_positions[bond.site_i]
    end = site_positions[bond.site_j] + np.asarray(bond.offset, dtype=float)
    new_start = rotation @ start + translation
    new_end = rotation @ end + translation
    index_i, shift_i = _site_index_and_offset(new_start, site_positions)
    index_j, shift_j = _site_index_and_offset(new_end, site_positions)
    offset = shift_j - shift_i
    image = Bond(index_i, index_j, (int(offset[0]), int(offset[1]), int(offset[2])))
    canonical = image.canonical()
    return canonical, canonical != image


def generate_spatial_bond_orbits(
    crystal: Mapping[str, Any],
    site_labels: Sequence[str],
    cutoff_angstrom: float,
    *,
    label_prefix: str = "B",
) -> tuple[list[CrystalSite], list[BondOrbit]]:
    """Enumerate bonds up to a cutoff and group them into symmetry orbits.

    Returns ``(sites, orbits)`` where ``sites`` is the expanded selected-site
    list (bond indices refer to it) and ``orbits`` are sorted by bond length.
    ``label_prefix`` selects stable labels such as ``B1`` for generic spatial
    orbits or ``J1`` for Heisenberg exchange.
    """

    if cutoff_angstrom <= 0.0:
        raise ValueError("cutoff_angstrom must be positive")
    prefix = str(label_prefix).strip()
    if not prefix:
        raise ValueError("label_prefix cannot be empty")
    sites = expand_crystal_sites(crystal, site_labels)
    if not sites:
        raise ValueError("no selected sites to bond")
    positions = np.asarray([site.position for site in sites], dtype=float)
    lattice = lattice_vectors(crystal["lattice"])
    operations = _symmetry_operations(crystal.get("spacegroup", "P 1"))

    # Cell offsets to search: |n_i| up to cutoff over the spacing of lattice
    # planes perpendicular to each axis (rows of the inverse lattice matrix).
    plane_spacing = 1.0 / np.linalg.norm(np.linalg.inv(lattice), axis=1)
    max_offsets = np.ceil(cutoff_angstrom / plane_spacing).astype(int) + 1

    bonds_by_distance: dict[float, list[Bond]] = {}
    offsets = [
        (n1, n2, n3)
        for n1 in range(-max_offsets[0], max_offsets[0] + 1)
        for n2 in range(-max_offsets[1], max_offsets[1] + 1)
        for n3 in range(-max_offsets[2], max_offsets[2] + 1)
    ]
    seen: set[Bond] = set()
    for i in range(len(sites)):
        for j in range(len(sites)):
            for offset in offsets:
                if i == j and offset == (0, 0, 0):
                    continue
                delta_frac = positions[j] + np.asarray(offset, dtype=float) - positions[i]
                distance = float(np.linalg.norm(lattice @ delta_frac))
                if distance > cutoff_angstrom or distance < _POSITION_TOL:
                    continue
                bond = Bond(i, j, offset).canonical()
                if bond in seen:
                    continue
                seen.add(bond)
                bonds_by_distance.setdefault(round(distance, _DISTANCE_DECIMALS), []).append(bond)

    # Group bonds into orbits under the space group, shell by shell. For each
    # orbit member remember the (first) operation carrying the representative
    # onto it, so anisotropic exchange tensors can be rotated onto every bond.
    shells: list[tuple[float, list[tuple[list[Bond], list[BondSymmetry]]]]] = []
    identity = np.eye(3)
    for distance in sorted(bonds_by_distance):
        remaining = set(bonds_by_distance[distance])
        orbits: list[tuple[list[Bond], list[BondSymmetry]]] = []
        while remaining:
            seed = min(remaining, key=lambda b: (b.site_i, b.site_j, b.offset))
            mapped: dict[Bond, BondSymmetry] = {
                seed: BondSymmetry(rotation=_rotation_tuple(identity), reverses=False)
            }
            for rotation, translation in operations:
                image, reverses = _transform_bond(seed, rotation, translation, positions)
                if image not in mapped:
                    mapped[image] = BondSymmetry(
                        rotation=_rotation_tuple(rotation), reverses=reverses
                    )
            orbit = set(mapped)
            if not orbit <= remaining:
                stray = sorted(orbit - remaining)[0]
                raise ValueError(
                    f"symmetry maps bond {seed} onto {stray}, which is not in "
                    "the same distance shell; inconsistent crystal input"
                )
            remaining -= orbit
            ordered = [seed] + sorted(
                orbit - {seed}, key=lambda b: (b.site_i, b.site_j, b.offset)
            )
            orbits.append((ordered, [mapped[bond] for bond in ordered]))
        shells.append((distance, orbits))

    labeled: list[BondOrbit] = []
    for shell_number, (distance, orbits) in enumerate(shells, start=1):
        for orbit_number, (orbit, symmetries) in enumerate(orbits):
            suffix = ""
            if len(orbits) > 1:
                suffix = chr(ord("a") + orbit_number)
            labeled.append(
                BondOrbit(
                    label=f"{prefix}{shell_number}{suffix}",
                    distance_angstrom=distance,
                    bonds=tuple(orbit),
                    operations=tuple(symmetries),
                )
            )
    return sites, labeled


def bond_stabilizer_symmetries(
    crystal: Mapping[str, Any],
    sites: Sequence[CrystalSite],
    bond: Bond,
) -> tuple[BondSymmetry, ...]:
    """Return every space-group operation that preserves an undirected bond."""

    positions = np.asarray([site.position for site in sites], dtype=float)
    canonical = bond.canonical()
    result = []
    for rotation, translation in _symmetry_operations(
        crystal.get("spacegroup", "P 1")
    ):
        image, reverses = _transform_bond(
            canonical,
            rotation,
            translation,
            positions,
        )
        if image == canonical:
            result.append(
                BondSymmetry(
                    rotation=_rotation_tuple(rotation),
                    reverses=reverses,
                )
            )
    return tuple(result)


def generate_bond_orbits(
    crystal: Mapping[str, Any],
    site_labels: Sequence[str],
    cutoff_angstrom: float,
) -> tuple[list[CrystalSite], list[BondOrbit]]:
    """Generate Heisenberg-compatible ``J``-labeled spatial bond orbits."""

    return generate_spatial_bond_orbits(
        crystal,
        site_labels,
        cutoff_angstrom,
        label_prefix="J",
    )


def cartesian_rotation(
    rotation: FloatArray | Sequence[Sequence[float]], lattice: Mapping[str, Any]
) -> FloatArray:
    """Convert a fractional-coordinate rotation to a Cartesian rotation matrix.

    ``R_cart = L R_frac L^{-1}`` with ``L`` the lattice-vector matrix. For a
    crystallographic operation the result is orthogonal (proper or improper).
    """

    basis = lattice_vectors(lattice)
    return basis @ np.asarray(rotation, dtype=float) @ np.linalg.inv(basis)


def _tensor_action_matrix(rotation_cart: FloatArray, transpose: bool) -> FloatArray:
    """9x9 matrix of ``T -> R (T or T^T) R^T`` acting on row-major ``vec(T)``.

    This is the transformation of a rank-2 spin-spin coupling tensor under a
    (possibly improper) point operation: spins are axial vectors, but the two
    ``det(R)`` factors cancel for rank-2, so ``R`` is used directly. The
    ``transpose`` form applies when the operation reverses the bond.
    """

    action = np.zeros((9, 9))
    for k in range(9):
        basis_tensor = np.zeros(9)
        basis_tensor[k] = 1.0
        tensor = basis_tensor.reshape(3, 3)
        image = rotation_cart @ (tensor.T if transpose else tensor) @ rotation_cart.T
        action[:, k] = image.reshape(9)
    return action


def _invariant_tensor_basis(
    actions: Sequence[FloatArray],
    *,
    remove_isotropic: bool,
    symmetric_only: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Orthonormal bases of the tensor subspace fixed by all ``actions``.

    Averages the group action (a projector, symmetric because the actions are
    orthogonal in the Frobenius inner product), extracts the eigenvalue-one
    subspace, optionally removes the isotropic direction, and splits the result
    into symmetric and antisymmetric parts. Returns ``(symmetric_basis,
    antisymmetric_basis)`` as ``(9, k)`` column matrices with deterministic
    ordering and sign.
    """

    projector = np.zeros((9, 9))
    for action in actions:
        projector += action
    projector /= len(actions)
    projector = 0.5 * (projector + projector.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projector)
    allowed = eigenvectors[:, eigenvalues > 0.99]

    if remove_isotropic:
        iso = np.eye(3).reshape(9) / np.sqrt(3.0)
        allowed = allowed - np.outer(iso, iso @ allowed)

    transpose_map = _tensor_action_matrix(np.eye(3), transpose=True)
    sym = 0.5 * (allowed + transpose_map @ allowed)
    antisym = 0.5 * (allowed - transpose_map @ allowed)

    def orthonormal(columns: FloatArray) -> FloatArray:
        if columns.size == 0:
            return np.zeros((9, 0))
        left, singular, _ = np.linalg.svd(columns, full_matrices=False)
        basis = left[:, singular > 1e-8]
        # Deterministic sign: first significant component positive.
        for index in range(basis.shape[1]):
            column = basis[:, index]
            lead = column[np.argmax(np.abs(column) > 1e-8)]
            if lead < 0:
                basis[:, index] = -column
        return np.round(basis, 12)

    if symmetric_only:
        return orthonormal(sym), np.zeros((9, 0))
    return orthonormal(sym), orthonormal(antisym)


def _bond_stabilizer_actions(
    bond: Bond,
    positions: FloatArray,
    operations: Sequence[tuple[FloatArray, FloatArray]],
    lattice: Mapping[str, Any],
) -> list[FloatArray]:
    """Tensor-space actions of every op mapping ``bond`` onto itself."""

    actions = []
    for rotation, translation in operations:
        image, reverses = _transform_bond(bond, rotation, translation, positions)
        if image == bond.canonical():
            rotation_cart = cartesian_rotation(rotation, lattice)
            actions.append(_tensor_action_matrix(rotation_cart, transpose=reverses))
    return actions


def symmetry_allowed_exchange_basis(
    crystal: Mapping[str, Any],
    sites: Sequence[CrystalSite],
    orbit: BondOrbit,
) -> list[dict[str, Any]]:
    """Symmetry-allowed anisotropic exchange basis for one bond orbit.

    Projects the 9-dimensional rank-2 tensor space onto the subspace invariant
    under the representative bond's stabilizer (operations mapping the bond
    onto itself, with the transpose constraint when they reverse it), removes
    the isotropic component (that is the orbit's Heisenberg parameter), and
    returns unit-Frobenius basis matrices in **Cartesian** coordinates:
    ``[{"name": "S1"|"D1"..., "kind": "symmetric"|"dm", "matrix": [[...]]}]``.
    Symmetric-traceless elements are named ``S1..``, antisymmetric
    (Dzyaloshinskii–Moriya) elements ``D1..``. A bond whose midpoint is an
    inversion center gets no DM elements, per the Moriya rules.
    """

    positions = np.asarray([site.position for site in sites], dtype=float)
    operations = _symmetry_operations(crystal.get("spacegroup", "P 1"))
    representative = orbit.bonds[0]
    actions = _bond_stabilizer_actions(
        representative, positions, operations, crystal["lattice"]
    )
    symmetric, antisymmetric = _invariant_tensor_basis(actions, remove_isotropic=True)
    basis: list[dict[str, Any]] = []
    for index in range(symmetric.shape[1]):
        basis.append(
            {
                "name": f"S{index + 1}",
                "kind": "symmetric",
                "matrix": symmetric[:, index].reshape(3, 3).tolist(),
            }
        )
    for index in range(antisymmetric.shape[1]):
        basis.append(
            {
                "name": f"D{index + 1}",
                "kind": "dm",
                "matrix": antisymmetric[:, index].reshape(3, 3).tolist(),
            }
        )
    return basis


def symmetry_allowed_sia_basis(
    crystal: Mapping[str, Any],
    site_label: str,
) -> list[dict[str, Any]]:
    """Symmetry-allowed single-ion anisotropy basis for one Wyckoff site.

    ``site_label`` names an entry of ``crystal["sites"]`` (the Wyckoff
    representative). Returns unit-Frobenius **symmetric-traceless** Cartesian
    basis matrices invariant under the site's point group:
    ``[{"name": "K1"..., "matrix": [[...]]}]``. Cubic site symmetry allows
    none (rank-2 anisotropy vanishes); tensors for symmetry-equivalent sites
    are obtained by rotating these with each site's recorded generator.
    """

    entries = {str(site["label"]): site for site in crystal.get("sites", [])}
    if str(site_label) not in entries:
        raise ValueError(
            f"site label {site_label!r} not found; crystal defines {sorted(entries)}"
        )
    base = np.asarray(entries[str(site_label)]["position"], dtype=float)
    operations = _symmetry_operations(crystal.get("spacegroup", "P 1"))
    actions = []
    for rotation, translation in operations:
        image = _wrap_fractional(rotation @ base + translation)
        if np.allclose(image, _wrap_fractional(base.copy()), atol=1e-6):
            rotation_cart = cartesian_rotation(rotation, crystal["lattice"])
            actions.append(_tensor_action_matrix(rotation_cart, transpose=False))
    symmetric, _ = _invariant_tensor_basis(
        actions, remove_isotropic=True, symmetric_only=True
    )
    return [
        {"name": f"K{index + 1}", "matrix": symmetric[:, index].reshape(3, 3).tolist()}
        for index in range(symmetric.shape[1])
    ]


def sites_to_config(sites: Sequence[CrystalSite]) -> list[list[float]]:
    """Serialize expanded site positions for model-component configuration."""

    return [[float(x) for x in site.position] for site in sites]


def site_rotations_to_config(sites: Sequence[CrystalSite]) -> list[Any]:
    """Serialize per-site generator rotations (fractional; ``None`` if absent)."""

    return [
        [[float(x) for x in row] for row in site.rotation]
        if site.rotation is not None
        else None
        for site in sites
    ]


def orbits_to_config(orbits: Sequence[BondOrbit]) -> list[dict[str, Any]]:
    """Serialize bond orbits into the plain-dict form models consume.

    When an orbit carries symmetry operations, each bond dict additionally has
    ``"rotation"`` (fractional 3x3) and ``"reversed"`` (bool) describing how
    the representative bond maps onto it.
    """

    payload: list[dict[str, Any]] = []
    for orbit in orbits:
        bonds = []
        for index, bond in enumerate(orbit.bonds):
            entry: dict[str, Any] = {
                "site_i": bond.site_i,
                "site_j": bond.site_j,
                "offset": list(bond.offset),
            }
            if orbit.operations is not None:
                symmetry = orbit.operations[index]
                entry["rotation"] = [[float(x) for x in row] for row in symmetry.rotation]
                entry["reversed"] = bool(symmetry.reverses)
            bonds.append(entry)
        payload.append(
            {
                "label": orbit.label,
                "distance_angstrom": float(orbit.distance_angstrom),
                "bonds": bonds,
            }
        )
    return payload


def orbits_from_config(payload: Sequence[Mapping[str, Any]]) -> list[BondOrbit]:
    """Rebuild :class:`BondOrbit` objects from their plain-dict form."""

    orbits: list[BondOrbit] = []
    for entry in payload:
        bonds = []
        symmetries: list[BondSymmetry] = []
        have_operations = True
        for bond in entry.get("bonds", []):
            bonds.append(
                Bond(
                    int(bond["site_i"]),
                    int(bond["site_j"]),
                    (
                        int(bond["offset"][0]),
                        int(bond["offset"][1]),
                        int(bond["offset"][2]),
                    ),
                )
            )
            rotation = bond.get("rotation")
            if rotation is None:
                have_operations = False
            else:
                symmetries.append(
                    BondSymmetry(
                        rotation=_rotation_tuple(np.asarray(rotation, dtype=float)),
                        reverses=bool(bond.get("reversed", False)),
                    )
                )
        orbits.append(
            BondOrbit(
                label=str(entry["label"]),
                distance_angstrom=float(entry.get("distance_angstrom", 0.0)),
                bonds=tuple(bonds),
                operations=tuple(symmetries) if have_operations and symmetries else None,
            )
        )
    return orbits
