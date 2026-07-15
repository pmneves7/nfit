import numpy as np
import pytest

from nfit.analysis.zones import (
    centering_translations,
    is_reciprocal_lattice_vector,
    nearest_zone_indices,
    reciprocal_basis_hkl,
)


@pytest.mark.parametrize("centering,multiplicity", [("P", 1), ("I", 2), ("F", 4), ("A", 2), ("B", 2), ("C", 2), ("R", 3)])
def test_centering_basis_has_correct_index_and_allowed_rows(centering, multiplicity):
    basis = reciprocal_basis_hkl(centering)
    assert round(np.linalg.det(basis)) == multiplicity
    translations = centering_translations(centering)
    assert all(is_reciprocal_lattice_vector(row, translations) for row in basis)


def test_nearest_zone_ties_follow_lexicographic_center_order():
    centers = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 0, 0]])
    indices = nearest_zone_indices(np.array([[0.5, 0, 0]]), centers, np.eye(3))
    assert np.array_equal(centers[indices[0]], [0, 0, 0])
