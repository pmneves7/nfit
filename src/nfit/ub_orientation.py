"""GUI-independent Mantid/Shiver sample orientation and UB file operations.

UB maps column HKL vectors to Q/(2 pi), in inverse angstroms, in the
zero-goniometer sample frame: +z along the beam and +y vertically upward.
The u and v vectors are reciprocal-lattice (HKL) directions, not real-space
directions. Their inverse mapping follows Mantid ``getuVector/getvVector``.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .fitting import reciprocal_basis_from_lattice_parameters

_LATTICE_NAMES = ("a", "b", "c", "alpha", "beta", "gamma")
_ISAW_TO_MANTID_COORDINATES = np.asarray(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=float
)


def validate_ub_matrix(ub: Any) -> np.ndarray:
    """Return an independent finite, nonsingular 3 x 3 matrix.

    No transpose, 2 pi scaling, frame conversion, or handedness change is
    inferred from the numerical entries.
    """
    matrix = np.array(ub, dtype=float, copy=True)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("UB must be a finite 3 x 3 matrix")
    if np.linalg.matrix_rank(matrix) != 3:
        raise ValueError("UB matrix must be invertible")
    return matrix


def mantid_ub_from_ipns(ub: Any) -> np.ndarray:
    """Convert IPNS/DAVE UB (beam +x, vertical +z) to Mantid sample axes.

    Only Cartesian rows are permuted: HKL columns, 2 pi scaling, and
    handedness stay unchanged. MACS's native import option uses IPNS axes.
    """
    return _ISAW_TO_MANTID_COORDINATES @ validate_ub_matrix(ub)


def ipns_ub_from_mantid(ub: Any) -> np.ndarray:
    """Convert Mantid UB (beam +z, vertical +y) to IPNS/DAVE sample axes."""
    return _ISAW_TO_MANTID_COORDINATES.T @ validate_ub_matrix(ub)


def validate_lattice_parameters(lattice: Mapping[str, Any]) -> dict[str, float]:
    """Validate lengths in angstroms, angles in degrees, and positive volume."""
    try:
        result = {name: float(lattice[name]) for name in _LATTICE_NAMES}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Lattice requires a, b, c, alpha, beta, and gamma") from exc
    if not np.all(np.isfinite(list(result.values()))):
        raise ValueError("Lattice parameters must be finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        basis = reciprocal_basis_from_lattice_parameters(*result.values(), include_2pi=False)
    if not np.all(np.isfinite(basis)) or np.linalg.matrix_rank(basis) != 3:
        raise ValueError("Lattice parameters must produce a finite, nonzero unit-cell volume")
    return result


def _orientation_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite HKL components")
    # Normalize before converting to Cartesian coordinates: orientation vector
    # scale carries no physical information, and large scales can overflow.
    scale = np.max(np.abs(vector))
    if scale == 0.0:
        raise ValueError(f"{name} must be nonzero")
    return vector / scale


def ub_from_lattice_orientation(lattice: Mapping[str, Any], u: Any, v: Any) -> np.ndarray:
    """Build UB with Mantid/Shiver ``setUFromVectors`` semantics.

    u points along +z (incident beam). v defines the horizontal plane and
    its part perpendicular to u points toward +x; u cross v points toward
    +y (vertical). Positive rescaling of either input has no effect.
    """
    parameters = validate_lattice_parameters(lattice)
    basis = reciprocal_basis_from_lattice_parameters(*parameters.values(), include_2pi=False)
    q_u = basis @ _orientation_vector(u, "u")
    q_v = basis @ _orientation_vector(v, "v")
    q_u /= np.linalg.norm(q_u)
    q_v /= np.linalg.norm(q_v)
    vertical = np.cross(q_u, q_v)
    norm = np.linalg.norm(vertical)
    if norm <= 1e-12:
        raise ValueError("u and v must be nonparallel and define a plane")
    vertical /= norm
    horizontal = np.cross(vertical, q_u)
    return np.vstack((horizontal, vertical, q_u)) @ basis


def lattice_orientation_from_ub(
    ub: Any, *, normalize_vectors: bool = False
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Derive the unit cell and Mantid/Shiver u/v directions from UB.

    Returns lengths in angstroms and angles in degrees. By default u is
    ``inverse(UB) @ [0,0,1]`` and v is ``inverse(UB) @ [1,0,0]``, exactly
    as returned by Mantid, including their scale. ``normalize_vectors``
    optionally gives Euclidean unit HKL vectors describing the same directions.
    For a right-handed UB these outputs reconstruct the original matrix.
    """
    matrix = validate_ub_matrix(ub)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        direct_rows = np.linalg.inv(matrix)
        lengths = np.hypot.reduce(direct_rows, axis=1)
    if not np.all(np.isfinite(direct_rows)) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("UB must determine finite, positive unit-cell lengths")
    directions = direct_rows / lengths[:, None]
    angles = [
        np.rad2deg(np.arccos(np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0)))
        for i, j in ((1, 2), (0, 2), (0, 1))
    ]
    lattice = validate_lattice_parameters(
        dict(zip(_LATTICE_NAMES, map(float, [*lengths, *angles]), strict=True))
    )
    u = direct_rows[:, 2].copy()
    v = direct_rows[:, 0].copy()
    if normalize_vectors:
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
    return lattice, u, v


def ub_setup_from_matrix(ub: Any) -> dict[str, Any]:
    """Return a serializable, coherent Mantid-frame cell/orientation setup."""
    matrix = validate_ub_matrix(ub)
    lattice, u, v = lattice_orientation_from_ub(matrix)
    return {"ub_matrix": matrix.tolist(), "lattice_parameters": lattice,
            "u": u.tolist(), "v": v.tolist()}


def ub_matrix_from_metadata(metadata: Any) -> np.ndarray | None:
    """Read a no-2 pi UB from explicit setup or recognized matrix metadata.

    Full HKL-to-Q matrices take precedence over UB/orientation aliases in
    each metadata container. Declared 2 pi factors are removed; Cartesian
    frame is preserved for instrument-specific conversion by the caller.
    """
    if not isinstance(metadata, Mapping):
        return None
    setup = metadata.get("ub_setup")
    if isinstance(setup, Mapping) and setup.get("ub_matrix") is not None:
        return validate_ub_matrix(setup["ub_matrix"])
    containers = [metadata]
    if isinstance(metadata.get("oriented_lattice"), Mapping):
        containers.append(metadata["oriented_lattice"])
    for container in containers:
        for key in ("rlu_to_inv_angstrom_matrix", "ub_matrix", "orientation_matrix"):
            if container.get(key) is None:
                continue
            matrix = validate_ub_matrix(container[key])
            flags = (f"{key}_includes_2pi", "q_matrix_includes_2pi", "include_2pi", "includes_2pi")
            includes = next((bool(container[name]) for name in flags if name in container), False)
            lattice = container.get("lattice_parameters")
            if not any(name in container for name in flags) and isinstance(lattice, Mapping):
                includes = bool(lattice.get("include_2pi", False))
            return matrix / (2.0 * np.pi) if key == "rlu_to_inv_angstrom_matrix" or includes else matrix
    return None


def ub_setup_metadata(setup: Mapping[str, Any]) -> dict[str, Any]:
    """Return coherent explicit orientation metadata with physical Q scaling."""
    canonical = ub_setup_from_matrix(setup["ub_matrix"])
    matrix = np.asarray(canonical["ub_matrix"])
    return {"ub_setup": canonical, "ub_setup_override": True,
            "ub_matrix": canonical["ub_matrix"], "ub_matrix_includes_2pi": False,
            "orientation_matrix": canonical["ub_matrix"], "orientation_matrix_includes_2pi": False,
            "rlu_to_inv_angstrom_matrix": (2.0 * np.pi * matrix).tolist(),
            "lattice_parameters": canonical["lattice_parameters"],
            "orientation_u": canonical["u"], "orientation_v": canonical["v"]}


def with_ub_setup_metadata(data, metadata):
    """Overlay an explicit setup on an immutable data view, preserving arrays.

    The saved setup uses Mantid axes. MACS payload matrices retain their
    native IPNS/DAVE axes, matching the numerical import and script contract.
    """
    setup = metadata.get("ub_setup") if isinstance(metadata, Mapping) else None
    if data is None or not isinstance(setup, Mapping) or setup.get("ub_matrix") is None:
        return data
    updated = dict(getattr(data, "metadata", {}) or {})
    changes = ub_setup_metadata(setup)
    if updated.get("importer") == "macs_nexus":
        native_matrix = ipns_ub_from_mantid(setup["ub_matrix"])
        changes.update(ub_matrix=native_matrix.tolist(), orientation_matrix=native_matrix.tolist(),
                       rlu_to_inv_angstrom_matrix=(2 * np.pi * native_matrix).tolist())
    updated.update(changes)
    if isinstance(updated.get("oriented_lattice"), Mapping):
        updated["oriented_lattice"] = {**updated["oriented_lattice"], **changes}
    return data.with_updates(metadata=updated)


def read_isaw_ub(path: str | Path) -> tuple[np.ndarray, dict[str, float]]:
    """Read transposed IPNS-frame ISAW UB, returning Mantid-frame UB and cell.

    As in Mantid, the matrix is authoritative for lattice parameters; the
    rounded lattice row is validated but does not override the matrix metric.
    Standard explanatory text after the numeric header is ignored.
    """
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            rows.append([float(value) for value in line.split()])
        except ValueError as exc:
            raise ValueError("Invalid numeric ISAW UB header") from exc
        if len(rows) == 4:
            break
    if len(rows) < 4 or any(len(row) < 3 for row in rows[:3]) or len(rows[3]) < 6:
        raise ValueError("ISAW UB files require three matrix rows and one six-value lattice row")
    validate_lattice_parameters(dict(zip(_LATTICE_NAMES, rows[3][:6], strict=True)))
    matrix = mantid_ub_from_ipns(np.asarray([row[:3] for row in rows[:3]]).T)
    return matrix, lattice_orientation_from_ub(matrix)[0]


def write_isaw_ub(
    path: str | Path, ub: Any, lattice: Mapping[str, Any] | None = None
) -> None:
    """Write UB in transposed IPNS/ISAW convention with a consistent cell.

    The optional lattice argument preserves the existing public call signature.
    If supplied, its metric must agree with UB (within ISAW rounding precision).
    """
    matrix = validate_ub_matrix(ub)
    parameters = lattice_orientation_from_ub(matrix)[0]
    if lattice is not None:
        supplied = validate_lattice_parameters(lattice)
        if not np.allclose(list(supplied.values()), list(parameters.values()), rtol=1e-4, atol=1e-4):
            raise ValueError("Lattice parameters do not agree with the UB matrix")
    matrix_ipns = ipns_ub_from_mantid(matrix)
    lines = [" ".join(f"{value: .12g}" for value in row) for row in matrix_ipns.T]
    volume = 1.0 / abs(np.linalg.det(matrix))
    lines.append(" ".join(f"{value:.12g}" for value in [*parameters.values(), volume]))
    lines.append(" ".join("0" for _ in range(7)))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _matrix_from_text(raw: str) -> np.ndarray:
    """Parse numeric literals and the array(...) wrapper used by SNS logs."""
    try:
        node = ast.parse(raw.strip(), mode="eval").body
        if isinstance(node, ast.Call):
            function = node.func
            allowed = (isinstance(function, ast.Name) and function.id in {"array", "matrix"}) or (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id in {"np", "numpy"}
                and function.attr in {"array", "matrix"}
            )
            if not allowed or len(node.args) != 1:
                raise ValueError("Unsupported UB array expression")
            node = node.args[0]
        result = np.asarray(ast.literal_eval(node), dtype=float)
    except (ValueError, TypeError, SyntaxError) as exc:
        raise ValueError("UB log must contain a numeric 3 x 3 matrix") from exc
    if result.shape == (9,):
        result = result.reshape(3, 3)
    return validate_ub_matrix(result)


def _read_nexus_matrix(dataset: h5py.Dataset, *, log: bool) -> np.ndarray:
    # This editor reads orientation metadata only. Never materialize a signal,
    # event table, or an unexpectedly unbounded log payload.
    if dataset.size > 4096 or dataset.id.get_storage_size() > 1024 * 1024:
        raise ValueError(f"Orientation metadata is too large: {dataset.name}")
    values = np.asarray(dataset[()])
    if values.dtype.kind in {"S", "U", "O"}:
        matrices = []
        for value in values.reshape(-1):
            text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            if not text.strip():
                continue
            if len(text) > 1024 * 1024:
                raise ValueError(f"Orientation metadata is too large: {dataset.name}")
            matrices.append(_matrix_from_text(text))
        if not matrices:
            raise ValueError(f"Empty UB matrix log: {dataset.name}")
    elif values.shape in {(3, 3), (9,)}:
        matrices = [validate_ub_matrix(values.reshape(3, 3))]
    elif log and values.ndim >= 2 and values.shape[-2:] == (3, 3):
        matrices = [validate_ub_matrix(matrix) for matrix in values.reshape(-1, 3, 3)]
    else:
        raise ValueError(f"UB must be a 3 x 3 matrix: {dataset.name}")
    first = matrices[0]
    if any(not np.allclose(first, matrix, rtol=1e-10, atol=1e-12) for matrix in matrices[1:]):
        raise ValueError(f"Ambiguous changing UB matrix log: {dataset.name}")
    return first


def read_nexus_ub(
    path: str | Path, *, matrix_path: str | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    """Read UB metadata from processed Mantid or unprocessed SNS NeXus.

    Processed ``sample/oriented_lattice/orientation_matrix`` values are read
    directly, including MD experiment trees. Unprocessed ``DASlogs/*UBMatrix*/
    value`` entries follow Mantid DGSPlanner's raw UB importer. No event or
    histogram data are loaded, and absent UB metadata never implies identity.

    Processed lattices take precedence over acquisition logs that can predate
    refinement. Multiple different matrices require an explicit ``matrix_path``
    to select a dataset; equal repeated experiment matrices are accepted.
    The matrix determines all returned lattice parameters.
    """
    with h5py.File(path, "r") as handle:
        if matrix_path is not None:
            item = handle.get(matrix_path)
            if isinstance(item, h5py.Group):
                item = item.get("value")
            if not isinstance(item, h5py.Dataset):
                raise ValueError(f"No orientation dataset at {matrix_path}")
            matrix = _read_nexus_matrix(item, log="ubmatrix" in item.name.lower())
            return matrix, lattice_orientation_from_ub(matrix)[0]

        processed = []
        raw_logs = []

        def collect(name, item):
            if not isinstance(item, h5py.Dataset):
                return
            parts = name.lower().split("/")
            if len(parts) >= 3 and parts[-2:] == ["oriented_lattice", "orientation_matrix"]:
                processed.append(item.name)
            elif (
                parts[-1] == "value" and len(parts) >= 3
                and "ubmatrix" in parts[-2]
                and any(part in {"daslogs", "das_logs", "logs"} for part in parts[:-2])
            ):
                raw_logs.append(item.name)

        handle.visititems(collect)
        candidates = processed or raw_logs
        if not candidates:
            raise ValueError("NeXus file does not contain a recognized UB/orientation matrix")
        matrices = [_read_nexus_matrix(handle[name], log=not processed) for name in candidates]
        matrix = matrices[0]
        if any(not np.allclose(matrix, other, rtol=1e-10, atol=1e-12) for other in matrices[1:]):
            raise ValueError(
                "Ambiguous NeXus orientation matrices; select matrix_path from: "
                + ", ".join(candidates)
            )
        return matrix, lattice_orientation_from_ub(matrix)[0]
