from __future__ import annotations

from itertools import combinations, product

import numpy as np

_CENTERING_TRANSLATIONS = {
    "P": ((0.0, 0.0, 0.0),),
    "I": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    "F": ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),
    "A": ((0.0, 0.0, 0.0), (0.0, 0.5, 0.5)),
    "B": ((0.0, 0.0, 0.0), (0.5, 0.0, 0.5)),
    "C": ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0)),
    "R": ((0.0, 0.0, 0.0), (2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)),
}


def centering_translations(spacegroup: str) -> np.ndarray:
    symbol = str(spacegroup).strip().upper()[:1]
    try:
        return np.asarray(_CENTERING_TRANSLATIONS[symbol], dtype=float)
    except KeyError as exc:
        raise ValueError(f"unsupported lattice centering in {spacegroup!r}") from exc


def is_reciprocal_lattice_vector(hkl: np.ndarray, translations: np.ndarray) -> bool:
    phases = np.asarray(translations, dtype=float) @ np.asarray(hkl, dtype=float)
    return bool(np.allclose(phases, np.rint(phases), atol=1e-9))


def reciprocal_basis_hkl(spacegroup: str, q_matrix: np.ndarray | None = None) -> np.ndarray:
    translations = centering_translations(spacegroup)
    multiplicity = len(translations)
    metric = np.eye(3) if q_matrix is None else np.asarray(q_matrix, dtype=float)
    candidates = [
        np.asarray(v, dtype=int)
        for v in product(range(-2, 3), repeat=3)
        if v != (0, 0, 0) and is_reciprocal_lattice_vector(np.asarray(v), translations)
    ]
    candidates.sort(key=lambda v: (float(np.linalg.norm(metric @ v)), *v.tolist()))
    choices = []
    for triple in combinations(candidates, 3):
        matrix = np.stack(triple, axis=0)
        determinant = int(round(np.linalg.det(matrix)))
        if abs(determinant) != multiplicity:
            continue
        if determinant < 0:
            matrix = matrix[[0, 2, 1]]
        score = sum(float(np.dot(metric @ row, metric @ row)) for row in matrix)
        choices.append((score, tuple(matrix.ravel().tolist()), matrix))
    if not choices:
        raise ValueError(f"could not construct reciprocal basis for {spacegroup!r}")
    return min(choices, key=lambda item: (item[0], item[1]))[2]


def generate_zone_centers(
    basis_hkl: np.ndarray, lower_hkl: np.ndarray, upper_hkl: np.ndarray, *, shell: int = 1
) -> np.ndarray:
    basis = np.asarray(basis_hkl, dtype=float)
    corners = np.asarray(list(product(*zip(lower_hkl, upper_hkl, strict=True))), dtype=float)
    coefficients = corners @ np.linalg.inv(basis)
    lo = np.floor(coefficients.min(axis=0)).astype(int) - shell
    hi = np.ceil(coefficients.max(axis=0)).astype(int) + shell
    centers = [np.asarray(index) @ basis for index in product(*(range(a, b + 1) for a, b in zip(lo, hi, strict=True)))]
    return np.asarray(sorted(centers, key=lambda row: tuple(row)), dtype=float)


def nearest_zone_indices(
    points_hkl: np.ndarray, centers_hkl: np.ndarray, q_matrix: np.ndarray
) -> np.ndarray:
    points_q = np.asarray(points_hkl, dtype=float) @ np.asarray(q_matrix, dtype=float).T
    centers = np.asarray(centers_hkl, dtype=float)
    centers_q = centers @ np.asarray(q_matrix, dtype=float).T
    distances = np.sum((points_q[:, None, :] - centers_q[None, :, :]) ** 2, axis=-1)
    order = np.lexsort((centers[:, 2], centers[:, 1], centers[:, 0]))
    return order[np.argmin(distances[:, order], axis=1)]
